#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see htt"://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#
"""
Auxiliary functions to run.py file
"""

import logging
import math
import os
from collections import Counter, defaultdict
from dataclasses import replace
from functools import partial
from hashlib import blake2b

import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset, load_from_disk
from datasets.dataset_dict import DatasetDict
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, matthews_corrcoef
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    PretrainedConfig,
    Trainer,
    TrainerCallback,
)

from nupic.research.frameworks.pytorch.model_utils import count_nonzero_params

__all__ = [
    "evaluate_language_model",
    "evaluate_tasks",
    "get_labels",
    "init_config",
    "init_datasets_mlm",
    "init_datasets_task",
    "init_model",
    "init_teacher_models",
    "init_tokenizer",
    "init_trainer",
    "preprocess_datasets_mlm",
    "preprocess_datasets_task",
    "run_hyperparameter_search",
    "test_tasks",
    "train",
]

"""
GLUE:
- SentencePair: (entailment, question answering)
    - 2 classes: QQP (question pairing), QNLI (adapted q&A), MRPC (paraphrase),
      RTE (adapted entailment from 3 to 2), WNLI (adapted multiple choice)
    - 3 classes: MNLI (entailment, neutral or contradiction)
    - 1 variable regresssion: STS-B  (similarity, from 0 to 5)
- Single Sentence, 2 classes: SST-2 (sentiment), COLA (grammatically correct)
"""

# Tasks ordered by training time
TASK_TO_KEYS = {
    "wnli": ("sentence1", "sentence2"),
    "rte": ("sentence1", "sentence2"),
    "stsb": ("sentence1", "sentence2"),
    "mrpc": ("sentence1", "sentence2"),
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "mnli": ("premise", "hypothesis"),  # includes matched and mismatched
}


def train(trainer, output_dir, last_checkpoint=None):
    """Trainig function applicable to pretraining language models and finetuning."""

    logging.info("Before training: total params: {:,} non zero params: {:,}".format(
        *count_nonzero_params(trainer.model)
    ))

    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload

    output_train_file = os.path.join(output_dir, "train_results.txt")
    if trainer.is_world_process_zero():
        with open(output_train_file, "w") as writer:
            logging.info("***** Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logging.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

        # Need to save the state, since Trainer.save_model saves only the tokenizer
        # with the model
        trainer.state.save_to_json(
            os.path.join(output_dir, "trainer_state.json")
        )

    logging.info("After training: total params: {:,} non zero params: {:,}".format(
        *count_nonzero_params(trainer.model)
    ))


def evaluate_tasks(trainer, output_dir, tasks, eval_datasets):
    """
    Evaluate tasks after finetuning.
    Returns evaluation dict with results.
    """
    eval_results = {}
    for eval_dataset, task in zip(eval_datasets, tasks):
        eval_result = trainer.evaluate(eval_dataset=eval_dataset)
        if task == "mnli-mm":
            eval_result = {f"mm_{k}": v for k, v in eval_result.items()}
        eval_results.update(eval_result)

        output_eval_file = os.path.join(
            output_dir, f"eval_results_{task}.txt"
        )
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logging.info(f"***** Eval results {task} *****")
                for key, value in sorted(eval_result.items()):
                    logging.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

    return eval_results


def test_tasks(trainer, output_dir, tasks, test_datasets, is_regression, label_list):
    """
    Test tasks after finetuning.
    """
    for test_dataset, task in zip(test_datasets, tasks):
        # Removing the `label` columns because it contains -1
        # and Trainer won't like that.
        test_dataset.remove_columns_("label")
        predictions = trainer.predict(test_dataset=test_dataset).predictions
        predictions = (np.squeeze(predictions) if is_regression
                       else np.argmax(predictions, axis=1))

        output_test_file = os.path.join(
            output_dir, f"test_results_{task}.txt"
        )
        if trainer.is_world_process_zero():
            with open(output_test_file, "w") as writer:
                logging.info(f"***** Test results {task} *****")
                writer.write("index\tprediction\n")
                for index, item in enumerate(predictions):
                    if is_regression:
                        writer.write(f"{index}\t{item:3.3f}\n")
                    else:
                        item = label_list[item]
                        writer.write(f"{index}\t{item}\n")


def evaluate_language_model(trainer, eval_dataset, output_dir):
    """Evaluate language model. Returns dict with results on perplexity metric. """
    results = {}
    eval_output = trainer.evaluate(eval_dataset)

    perplexity = math.exp(eval_output["eval_loss"])
    results["perplexity"] = perplexity

    output_eval_file = os.path.join(output_dir, "eval_results_mlm.txt")
    if trainer.is_world_process_zero():
        with open(output_eval_file, "w") as writer:
            logging.info("***** Eval results *****")
            for key, value in sorted(results.items()):
                logging.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

    return results


def load_and_concatenate_datasets(data_args):
    """Load and concatenate multiple compatible datasets"""
    train_datasets, validation_datasets = [], []
    for name, config in zip(data_args.dataset_name, data_args.dataset_config_name):

        dataset = load_dataset(name, config)
        if "validation" not in dataset.keys():
            validation_ds = load_dataset(
                name, config,
                split=f"train[:{data_args.validation_split_percentage}%]",
            )
            train_ds = load_dataset(
                name, config,
                split=f"train[{data_args.validation_split_percentage}%:]",
            )
        else:
            validation_ds = dataset["validation"]
            train_ds = dataset["train"]

        # Some specific preprocessing to align fields on known datasets
        # extraneous fields not used in language modeling are also removed
        # after preprocessing
        if name == "wikipedia":
            train_ds.remove_columns_("title")
            validation_ds.remove_columns_("title")
        elif name == "ptb_text_only":
            train_ds.rename_column_("sentence", "text")
            validation_ds.rename_column_("sentence", "text")

        train_datasets.append(train_ds)
        validation_datasets.append(validation_ds)

    for ds_idx in range(1, len(train_datasets)):
        assert train_datasets[ds_idx].features.type == \
            train_datasets[ds_idx - 1].features.type, \
            "Features name and type must match between all datasets"

    datasets = DatasetDict()
    datasets["train"] = concatenate_datasets(train_datasets)
    datasets["validation"] = concatenate_datasets(validation_datasets)

    return datasets


def preprocess_datasets_mlm(datasets, tokenizer, data_args, column_names,
                            text_column_name):
    """Tokenize datasets and applies remaining preprocessing steps"""

    if data_args.line_by_line:
        # When using line_by_line, we just tokenize each nonempty line.
        padding = "max_length" if data_args.pad_to_max_length else False

        def tokenize_function(examples):
            # Remove empty lines
            examples["text"] = [
                line for line in examples["text"]
                if len(line) > 0 and not line.isspace()
            ]
            return tokenizer(
                examples["text"],
                padding=padding,
                truncation=True,
                max_length=data_args.max_seq_length,
                # We use this option because DataCollatorForLanguageModeling (see below)
                # is more efficient when it receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )

        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=[text_column_name],
            load_from_cache_file=not data_args.overwrite_cache,
        )
    else:
        # Otherwise, we tokenize every text, then concatenate them together before
        # splitting them in smaller parts. We use `return_special_tokens_mask=True`
        # because DataCollatorForLanguageModeling (see below) is more
        # efficient when it receives the `special_tokens_mask`.
        def tokenize_function(examples):
            return tokenizer(
                examples[text_column_name], return_special_tokens_mask=True
            )

        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

        if data_args.max_seq_length is None:
            max_seq_length = tokenizer.model_max_length
            if max_seq_length > 1024:
                logging.warning(
                    f"The tokenizer picked seems to have a very large "
                    f"`model_max_length` ({tokenizer.model_max_length}). "
                    f"Picking 1024 instead. You can change that default value by "
                    f"passing --max_seq_length xxx. "
                )
                max_seq_length = 1024
        else:
            if data_args.max_seq_length > tokenizer.model_max_length:
                logging.warning(
                    f"The max_seq_length passed ({data_args.max_seq_length}) is larger "
                    f"than the maximum length for the model "
                    f"({tokenizer.model_max_length}). "
                    f"Using max_seq_length={tokenizer.model_max_length}. "
                )
            max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

        # Main data processing function that will concatenate all texts from our dataset
        # and generate chunks of max_seq_length.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported
            # it instead of this drop, you can customize this part to your needs.
            total_length = (total_length // max_seq_length) * max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + max_seq_length] for i in
                    range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so
        # group_texts throws away a remainder for each of those groups of 1,000 texts.
        # You can adjust that batch_size here but a higher value might be slower
        # to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the
        # map method for more information:
        tokenized_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    return tokenized_datasets


def preprocess_datasets_task(datasets, tokenizer, data_args, model,
                             num_labels, label_list, is_regression):
    """Preprocess datasets for finetuning"""

    if data_args.task_name is not None:
        sentence1_key, sentence2_key = TASK_TO_KEYS[data_args.task_name]
    else:
        # Defaults, but don't hesitate to tweak to your use case.
        non_label_column_names = [
            name for name in datasets["train"].column_names if name != "label"
        ]
        if ("sentence1" in non_label_column_names
           and "sentence2" in non_label_column_names):
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # Pad later, dynamically at batch creation, to the max sequence length per batch
        padding = False

    # Some models have set the order of the labels to use, make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {
                i: label_name_to_id[label_list[i]] for i in range(num_labels)
            }
        else:
            logging.warn(
                "Your model seems to have been trained with labels, but they don't "
                "match the dataset. ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, " ,
                f"dataset labels: {list(sorted(label_list))}.",
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logging.warn(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than ",
            f"the maximum length for the model ({tokenizer.model_max_length}). ",
            f"Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(
            *args, padding=padding, max_length=max_seq_length, truncation=True
        )

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [label_to_id[label] for label in examples["label"]]
        return result

    tokenized_datasets = datasets.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=not data_args.overwrite_cache
    )

    return tokenized_datasets


def init_datasets_mlm(data_args):
    """
    Initialize datasets.
    Returns a regular version of dataset and a tokenized version if exists.

    Get the datasets: you can either provide your own CSV/JSON/TXT training
    and evaluation files (see below) or just provide the name of one of the public
    datasets available on the hub at https://huggingface.co/datasets/
    (the dataset will be downloaded automatically from the datasets Hub

    For CSV/JSON files, this script will use the column called 'text' or the first
    column. You can easily tweak this behavior (see below)

    In distributed training, the load_dataset function guarantee that only one local
    process can concurrently download the dataset.

    See more about loading any type of standard or custom dataset
    (from files, python dict, pandas DataFrame, etc) at
    https://huggingface.co/docs/datasets/loading_datasets.html.
    """

    datasets, tokenized_datasets, dataset_path = None, None, None

    if data_args.dataset_name is not None:
        # Load from dataset instead
        if not(isinstance(data_args.dataset_name, tuple)):
            # account for when a single dataset is passed as argument
            data_args.dataset_name = (data_args.dataset_name, )
        if not (isinstance(data_args.dataset_config_name, tuple)):
            data_args.dataset_config_name = (data_args.dataset_config_name, )

        assert len(data_args.dataset_name) == len(data_args.dataset_config_name), \
            ("If using more than one dataset, length of dataset_name and "
             "dataset_config_name must match")

        # Verifies if dataset is already saved
        dataset_folder = hash_dataset_folder_name(data_args)
        dataset_path = os.path.join(
            os.path.abspath(data_args.tokenized_data_cache_dir),
            str(dataset_folder)
        )
        logging.info(f"Tokenized dataset cache folder: {dataset_path}")

        if os.path.exists(dataset_path) and data_args.reuse_tokenized_data:
            logging.info(f"Loading cached tokenized data ...")
            tokenized_datasets = load_from_disk(dataset_path)
        else:
            datasets = load_and_concatenate_datasets(data_args)

    else:
        # If dataset not available in Hub, look for train and validation files.
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = data_args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        datasets = load_dataset(extension, data_files=data_files)

    return datasets, tokenized_datasets, dataset_path


def init_datasets_task(data_args, training_args):
    """
    Get the datasets for finetuning on tasks. Returns dataset.

    You can either provide your own CSV/JSON training and evaluation files (see below)
    or specify a GLUE benchmark task (the dataset will be downloaded automatically
    from the datasets Hub).

    For CSV/JSON files, this script will use as labels the column called 'label' and as
    pair of sentences the sentences in columns called 'sentence1' and 'sentence2' if
    such column exists or the first two columns not named label if at least two columns
    are provided.

    If the CSVs/JSONs contain only one non-label column, the script does single sentence
    classification on this single column. You can easily tweak this behavior (see below)

    In distributed training, the load_dataset function guarantee that only one local
    process can concurrently download the dataset.

    See more about loading any type of standard or custom dataset at
    https://huggingface.co/docs/datasets/loading_datasets.html.

    TODO: implement recovering tokenized version
    """

    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset("glue", data_args.task_name)
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {
            "train": data_args.train_file, "validation": data_args.validation_file
        }

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert test_extension == train_extension, \
                    "`test_file` should have the same extension (csv or json) " \
                    + "as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError(
                    "Need either a GLUE task or a test file for `do_predict`."
                )

        for key in data_files.keys():
            logging.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            datasets = load_dataset("csv", data_files=data_files)
        else:
            # Loading a dataset from local json files
            datasets = load_dataset("json", data_files=data_files)

    return datasets


def get_labels(datasets, data_args):
    label_list = None
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = (
            datasets["train"].features["label"].dtype in ["float32", "float64"]
        )
        if is_regression:
            num_labels = 1
        else:
            label_list = datasets["train"].unique("label")
            label_list.sort()  # sort it for determinism
            num_labels = len(label_list)

    return is_regression, label_list, num_labels


def init_config(model_args, extra_config_kwargs=None):
    """
    Distributed training:
    The .from_pretrained methods guarantee that only one local process can
    concurrently download model & vocab.
    """
    config_kwargs = dict(
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if extra_config_kwargs is not None:
        config_kwargs.update(extra_config_kwargs)

    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, **config_kwargs
        )
    else:
        config = CONFIG_MAPPING[model_args.model_type](**model_args.config_kwargs)
        logging.warning("You are instantiating a new config instance from scratch.")

    return config


def init_tokenizer(model_args):
    """
    Distributed training:
    The .from_pretrained methods guarantee that only one local process can
    concurrently download model & vocab.
    """
    tokenizer_kwargs = dict(
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, **tokenizer_kwargs
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, **tokenizer_kwargs
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported "
            "by this script. You can do it from another script, save it, and load it "
            "from here, using --tokenizer_name."
        )

    return tokenizer


def init_model(model_args, config, tokenizer, finetuning=False):
    """"
    Initialize a model for pretraining or finetuning
    """

    # Load model
    if model_args.model_name_or_path is not None:
        model_kwargs = dict(
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        if finetuning:
            logging.info("Loading a pretrained model for finetuning")
            model = AutoModelForSequenceClassification.from_pretrained(
                model_args.model_name_or_path, **model_kwargs
            )
        else:
            logging.info("Loading a pretrained model to continue pretraining")
            model = AutoModelForMaskedLM.from_pretrained(
                model_args.model_name_or_path, **model_kwargs
            )
            model.resize_token_embeddings(len(tokenizer))
    else:
        if finetuning:
            raise ValueError(
                "Finetuning models must be loaded from pretrained models."
            )
        else:
            logging.info("Pretraining new model from scratch")
            model = AutoModelForMaskedLM.from_config(config)
            model.resize_token_embeddings(len(tokenizer))

    logging.info(f"Initialized model: {model}")
    return model


def init_trainer(
    tokenizer,
    data_collator,
    training_args,
    train_dataset,
    eval_dataset,
    model=None,
    trainer_callbacks=None,
    finetuning=False,
    task_name=None,
    is_regression=False,
    trainer_class=Trainer,
    trainer_extra_kwargs=None,
):
    """Initialize Trainer, main class that controls the experiment"""
    if trainer_callbacks is not None:
        for cb in trainer_callbacks:
            assert isinstance(cb, TrainerCallback), \
                "Trainer callbacks must be an instance of TrainerCallback"

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=trainer_callbacks,
    )
    trainer_kwargs.update(trainer_extra_kwargs)

    # Add specific metrics for finetuning task
    if finetuning:
        compute_metrics = partial(
            compute_metrics_task, task_name=task_name, is_regression=is_regression,
        )
        trainer_kwargs.update(compute_metrics=compute_metrics)

    trainer = trainer_class(**trainer_kwargs)

    return trainer


def compute_metrics_task(ep: EvalPrediction, metric=None,
                         task_name=None, is_regression=None):
    """
    You can define your custom compute_metrics function. It takes an
    `EvalPrediction` object (a namedtuple with a predictions and label_ids
    field) and has to return a dictionary string to float.
    """
    preds = (ep.predictions[0] if isinstance(ep.predictions, tuple) else ep.predictions)
    preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)

    if not is_regression:
        logging.info(f"Label distribution for {task_name} before cleaning")
        logging.info(f"Predictions: {Counter(preds).most_common()}")
        logging.info(f"Labels: {Counter(ep.label_ids).most_common()}")

    # Ignore the -100 labels - not able to tokenize?
    # TODO: investigate why a few labels are -100 in all tasks
    label_ids = ep.label_ids[np.where(ep.label_ids != -100)]
    preds = preds[np.where(ep.label_ids != -100)]
    logging.info(f"Removing {1-(len(label_ids) / len(ep.label_ids)):.2%} samples "
                 "from evaluation set where label == -100")

    if not is_regression:
        logging.info(f"Label distribution for {task_name} after cleaning")
        logging.info(f"Predictions: {Counter(preds).most_common()}")
        logging.info(f"Labels: {Counter(label_ids).most_common()}")

    if task_name is not None:
        if task_name == "cola":
            result = {"matthews_correlation": matthews_corrcoef(label_ids, preds)}
        elif task_name == "stsb":
            result = pearson_and_spearman(preds, label_ids)
        elif task_name in ["mrpc", "qqp"]:
            result = acc_and_f1(preds, label_ids)
        elif task_name in ["sst2", "mnli", "mnli-mm", "mnli_mismatched", "mnli_matched",
                           "qnli", "rte", "wnli", "hans"]:
            result = {"accuracy": simple_accuracy(preds, label_ids)}
        # Consolidate if more than one metric
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result
    elif is_regression:
        return {"mse": ((preds - label_ids) ** 2).mean().item()}
    else:
        return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "accuracy": acc,
        "f1": f1,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
    }


class TaskResults():
    """
    Collects and reports results for each finetuning task
    """

    reporting_metrics_per_task = {
        "cola": ["eval_matthews_correlation"],
        "mnli": ["eval_accuracy", "mm_eval_accuracy"],
        "mrpc": ["eval_f1", "eval_accuracy"],
        "qnli": ["eval_accuracy"],
        "qqp": ["eval_accuracy", "eval_f1"],
        "rte": ["eval_accuracy"],
        "sst2": ["eval_accuracy"],
        "stsb": ["eval_pearson", "eval_spearmanr"],
        "wnli": ["eval_accuracy"]
    }

    def __init__(self, task_name, training_args=None):
        self.task_name = task_name
        self.reporting_metrics = self.reporting_metrics_per_task[task_name]
        self.all_results = []
        self._results = None
        self.training_args = training_args

    def append(self, results):
        self.all_results.append(results)

    def reduce_metrics(self, reduction="mean"):
        aggregated_results = defaultdict(list)
        for results in self.all_results:
            for metric, value in results.items():
                aggregated_results[metric].append(value)

        if reduction == "mean":
            self._results = {k: np.average(v) for k, v in aggregated_results.items()}

        elif reduction == "max":
            argmax_idx = np.argmax(aggregated_results[self.reporting_metrics[0]])
            self._results = self.all_results[argmax_idx]

    @property
    def results(self):
        if self._results is not None:
            return self._results
        elif len(self.all_results) > 0:
            return self.all_results[0]
        else:
            raise AttributeError(f"Results not available for {self.task_name}")

    def consolidate(self):
        return np.average([self.results[m] for m in self.reporting_metrics])

    def to_string(self):
        results_to_string = [
            f"{self.results[m]*100:.2f}" for m in self.reporting_metrics
        ]
        return "/".join(results_to_string)


def hash_dataset_folder_name(data_args):
    """
    Creates a hashed name for the dataset folder comprised of the dataset_name and
    dataset_name_config. As well, the following data_args are included unless their
    default values are used.
        - max_seq_length (default None)

    More arguments can be added to the hashed name as needed.
    """
    defaults = dict(
        max_seq_length=None,
    )

    dataset_folder = "-".join([
        f"{name}_{config}" for name, config in
        zip(data_args.dataset_name, data_args.dataset_config_name)
    ])

    for arg, default in defaults.items():
        if getattr(data_args, arg) != default:
            non_default = getattr(data_args, arg)
            dataset_folder += f" ({arg}={non_default})"

    hashed_folder_name = blake2b(dataset_folder.encode(), digest_size=20).hexdigest()
    print(f"Hashing dataset folder name '{dataset_folder}' to '{hashed_folder_name}'")
    return hashed_folder_name


def init_teacher_models(model_args, tokenizer):
    """Initialize and return list of teacher models"""

    teacher_models = []
    for model_name_or_path in model_args.teacher_models_name_or_path:
        teacher_args = replace(model_args, model_name_or_path=model_name_or_path)
        teacher_config = init_config(teacher_args)
        teacher_models.append(init_model(teacher_args, teacher_config, tokenizer))

    logging.info(f"{len(teacher_models)} teacher models initialized.")

    return teacher_models


def run_hyperparameter_search(
    model_args,
    config,
    tokenizer,
    data_collator,
    training_args,
    train_dataset,
    eval_dataset,
):
    """
    Run hyperparameter search using Ray Tune
    Not tested when using multiple instances with torch.distributed.launch
    """

    model_args.trainer_extra_kwargs.update(
        model_init=partial(init_model, model_args, config, tokenizer, False)
    )
    training_args.load_best_model_at_end = True
    training_args.disable_tqdm = True
    # TODO: sync metric_for_best_model with compute_objective, should be same metric
    # and accept custom metrics defined by user
    training_args.metric_for_best_model = "eval_loss"
    # TODO: load best model and proceed to evaluation normally after hp search
    training_args.do_eval = False
    training_args.do_predict = False

    # Get fraction of the validation dataset to use in hp search
    hp_eval_dataset = eval_dataset.shard(
        index=1, num_shards=int(1 / model_args.hp_validation_dataset_pct)
    )

    trainer = init_trainer(
        tokenizer=tokenizer,
        data_collator=data_collator,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=hp_eval_dataset,
        trainer_class=model_args.trainer_class,
        trainer_extra_kwargs=model_args.trainer_extra_kwargs,
        trainer_callbacks=model_args.trainer_callbacks or None,
    )

    hp_search_kwargs = dict(
        direction="maximize",
        backend="ray",
        n_trials=model_args.hp_num_trials,
        hp_space=model_args.hp_space,
        compute_objective=model_args.hp_compute_objective,
        local_dir=training_args.output_dir,
        resources_per_trial=dict(
            cpu=os.cpu_count() / torch.cuda.device_count() - 1,
            gpu=1
        ),
        checkpoint_freq=0,
        keep_checkpoints_num=0,
        checkpoint_at_end=False,
    )
    # Update any extra kwargs defined in config
    hp_search_kwargs.update(**model_args.hp_extra_kwargs)

    # Run hp search and save results
    best_run = trainer.hyperparameter_search(**hp_search_kwargs)
    logging.info(f"Best run: {best_run}")

    hp_res_file = os.path.join(training_args.output_dir, "hp_search_results.txt")
    if trainer.is_world_process_zero():
        with open(hp_res_file, "w") as writer:
            writer.write("Hyperparameter search best run:\n")
            writer.write(f"run_id = {best_run.run_id}\n")
            writer.write(f"{training_args.metric_for_best_model}")
            writer.write(f"= {best_run.objective}\n")
            writer.write(f"\nHyperparameters:\n")
            for key, value in sorted(best_run.hyperparameters.items()):
                writer.write(f"{key} = {value}\n")


def compute_objective_eval_loss(metrics):
    return metrics["eval_loss"]
