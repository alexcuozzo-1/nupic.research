# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------
from copy import deepcopy

from ray import tune
from transformers import Trainer

from callbacks import PlotDensitiesCallback, RezeroWeightsCallback
from trainer_mixins import DistillationTrainerMixin, OneCycleLRMixin, RigLMixin

from .finetuning import finetuning_bert700k_glue
from .sparse_bert import fully_static_sparse_bert_100k_fp16
from .sparse_bertitos import small_bert_sparse_100k, tiny_bert_sparse_100k
from .trifecta import KDLRRangeTestTrainer


class RigLOneCycleLRTrainer(OneCycleLRMixin, RigLMixin, Trainer):
    pass


class RigLDistillationTrainer(DistillationTrainerMixin, RigLMixin, Trainer):
    pass


class KDOneCycleLRTrainer(DistillationTrainerMixin, OneCycleLRMixin, Trainer):
    pass


onecycle_args = dict(
    pct_start=0.3,
    anneal_strategy="linear",
    cycle_momentum=True,
    base_momentum=0.85,
    max_momentum=0.95,
    div_factor=25,
    final_div_factor=1e4,
    last_epoch=-1,
)

rigl_args = dict(
    prune_fraction=0.3,
    prune_freq=100,
)

kd_args = dict(
    teacher_model_names_or_paths=[
        "/mnt/efs/results/pretrained-models/transformers-local/bert_1mi",
    ],
)

# ----------
# Tiny BERT
# ----------

# |------------------------------------------------------------------|
# | model                                 | eval loss      | steps   |
# |---------------------------------------|:--------------:|:-------:|
# | tiny_bert                             | 4.021          | 100k    |
# | tiny_bert_sparse                      | 5.865          | 100k    |
# | tiny_bert_sparse KD + RigL + OneCycle | 3.578          | 100k    |
# | tiny_bert_sparse KD + OneCycle        | 3.827          | 100k    |
# | tiny_bert_sparse                      | 5.774          | 300k    |
# | tiny_bert_sparse KD + RigL + OneCycle | 3.507          | 300k    |
# | tiny_bert_sparse KD + OneCycle        | 3.938          | 300k    |
# |------------------------------------------------------------------|
#

# RigL + OneCycleLR
tiny_bert_rigl_100k_onecycle_lr = deepcopy(tiny_bert_sparse_100k)
tiny_bert_rigl_100k_onecycle_lr.update(
    max_steps=100000,
    trainer_class=RigLOneCycleLRTrainer,
    trainer_mixin_args=dict(
        max_lr=0.0075,
        **onecycle_args,
        **rigl_args,
    ),
    overwrite_output_dir=True,
    fp16=True,
)


# RigL + KD
tiny_bert_rigl_100k_kd = deepcopy(tiny_bert_sparse_100k)
tiny_bert_rigl_100k_kd.update(
    model_type="fully_static_sparse_bert",
    trainer_callbacks=[
        RezeroWeightsCallback(),
        PlotDensitiesCallback(plot_freq=1000),
    ],
    trainer_class=RigLDistillationTrainer,
    trainer_mixin_args=dict(
        **kd_args,
        **rigl_args,
    ),
    fp16=True,
    overwrite_output_dir=True,
)


# KD + OneCycleLR
tiny_bert_sparse_300k_onecycle_lr_kd = deepcopy(tiny_bert_sparse_100k)
tiny_bert_sparse_300k_onecycle_lr_kd.update(
    max_steps=300000,
    trainer_class=KDOneCycleLRTrainer,
    trainer_mixin_args=dict(
        max_lr=0.0075,
        **kd_args,
        **onecycle_args,
    ),
    overwrite_output_dir=True,
    fp16=True,
)


# KD + OneCycleLR (100k) (eval/loss=4.031)
tiny_bert_sparse_100k_onecycle_lr_kd = deepcopy(tiny_bert_sparse_300k_onecycle_lr_kd)
tiny_bert_sparse_100k_onecycle_lr_kd.update(
    max_steps=100000,
)


# Search for the best max_lr parameters for tiny BERT trained with KD and OneCycle LR
def max_lr_hp_space(trial):
    return dict(
        trainer_mixin_args=dict(
            max_lr=tune.grid_search([
                0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 0.011,
                0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018,
            ]),
        )
    )


tiny_bert_kd_onecycle_50k_maxlr_search = deepcopy(tiny_bert_sparse_300k_onecycle_lr_kd)
tiny_bert_kd_onecycle_50k_maxlr_search.update(
    max_steps=50000,

    # hyperparameter search
    hp_space=max_lr_hp_space,
    hp_num_trials=1,
    hp_validation_dataset_pct=0.05,  # default
    hp_extra_kwargs=dict()  # default
)


# Search for the best pct_start parameters for tiny BERT trained with KD and OneCycle LR
def pct_start_hp_space(trial):
    return dict(
        trainer_mixin_args=dict(
            # Vary percent-start as 10%, 20%, or 30%.
            # The lr will then peak at either 30k, 60k, 90k steps.
            pct_start=tune.grid_search([0.1, 0.2, 0.3]),

            # Use the same max_lr and KD args for each run.
            max_lr=0.01,
            **kd_args,
        )
    )


tiny_bert_kd_onecycle_300k_pct_start_search = deepcopy(tiny_bert_sparse_300k_onecycle_lr_kd)  # noqa E501
tiny_bert_kd_onecycle_300k_pct_start_search.update(
    # hyperparameter search
    hp_space=pct_start_hp_space,
    hp_num_trials=1,
    hp_validation_dataset_pct=0.05,  # default
    hp_extra_kwargs=dict(),  # default

    # Using batch_size of 16 instead of 128 since we're training on 8 GPUs.
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
)


tiny_bert_kd_onecycle_100k_pct_start_search = deepcopy(tiny_bert_kd_onecycle_300k_pct_start_search)  # noqa E501
tiny_bert_kd_onecycle_100k_pct_start_search.update(
    max_steps=100000,
)


# ----------
# Small BERT
# ----------


small_bert_rigl_100k_onecycle_lr = deepcopy(small_bert_sparse_100k)
small_bert_rigl_100k_onecycle_lr.update(
    model_type="fully_static_sparse_bert",
    overwrite_output_dir=True,

    # RigL
    trainer_callbacks=[
        RezeroWeightsCallback(),
        PlotDensitiesCallback(plot_freq=1000),
    ],
    fp16=True,

    # One cycle lr
    trainer_class=RigLOneCycleLRTrainer,
    trainer_mixin_args=dict(
        # One cycle lr
        max_lr=0.003,
        **onecycle_args,
        **rigl_args,
    ),
)


# ---------
# BERT Base
# ---------


# BERT Base with KD + OneCycle LR
# This achieves and eval-loss of 2.28, just slightly over 2.154 from its dense
# counterpart. See `sparse_v4_kd_100k` in the README for more details.
# This took 22h 17m to run on four ps.16xlarges
bert_sparse_100k_kd_oncycle_lr = deepcopy(fully_static_sparse_bert_100k_fp16)
bert_sparse_100k_kd_oncycle_lr.update(
    trainer_class=KDOneCycleLRTrainer,
    trainer_mixin_args=dict(

        # One cycle lr
        max_lr=0.0012,
        pct_start=0.3,
        anneal_strategy="linear",
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=25,
        final_div_factor=1e4,
        last_epoch=-1,

        # KD
        teacher_model_names_or_paths=[
            "/mnt/efs/results/pretrained-models/transformers-local/bert_1mi"
        ],
    ),
    overwrite_output_dir=True,
)


# This is an lr-range test for `bert_sparse_100k_kd_oncycle_lr` above.
# This test helped decide to set `max_lr=0.0012`.
# This took 20m to run on four ps.16xlarges
bert_sparse_100k_kd_lr_range_test = deepcopy(fully_static_sparse_bert_100k_fp16)
bert_sparse_100k_kd_lr_range_test.update(
    max_steps=100000,
    trainer_class=KDLRRangeTestTrainer,
    trainer_mixin_args=dict(
        # LR Range Test
        min_lr=0.0001,
        max_lr=0.005,
        test_mode="linear",

        # KD
        teacher_model_names_or_paths=[
            "/mnt/efs/results/pretrained-models/transformers-local/bert_1mi"
        ],
    ),
    overwrite_output_dir=True,
)


# This fine-tunes a pretrained model from `bert_sparse_100k_kd_oncycle_lr` above.
# This took 6h 20m to run on a p3.2xlarge
finetuning_bert_sparse_kd_oncycle_lr_100k_glue = deepcopy(finetuning_bert700k_glue)
finetuning_bert_sparse_kd_oncycle_lr_100k_glue.update(
    # Model arguments
    model_type="fully_static_sparse_bert",
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/bert_sparse_80%_kd_onecycle_lr_100k",  # noqa: E501
)

# ---------
# Deepspeed
# ---------

# This lr-range test is based on `bert_sparse_100k_kd_lr_range_test` and adapted
# for deepspeed training.
# With this test the best `max_lr` value found is `0.0017`.
# On four p3.16xlarge it takes ~20m to run
bert_sparse_100k_kd_lr_range_test_deepspeed = deepcopy(bert_sparse_100k_kd_lr_range_test)  # noqa: E501
bert_sparse_100k_kd_lr_range_test_deepspeed.update(
    max_steps=100,
    tokenized_data_cache_dir="/mnt/datasets/huggingface/preprocessed-datasets/text",
    fp16=False,  # Use deepspeed FP16 instead of apex
    deepspeed={
        "zero_optimization": {
            "stage": 1,
        },
        # When using fp16 dynamic loss scale, deepspeed will skip the optimizer
        # and LR scheduler steps whenever the loss value overflows (NaN/Inf).
        # Using deepspeed default values the loss will likely overflow on the
        # first few steps as the dynamic loss scale warms up. When the loss
        # overflows, huggingface will detect the LR scheduler step was skipped
        # and return zero as the current learning rate potentially affecting the
        # results of the LR range test. To avoid loss overflow during the LR
        # range test you could use static loss scale or use a smaller initial
        # scale power.
        # See https://www.deepspeed.ai/docs/config-json/#fp16-training-options
        "fp16": {
            "enabled": True,
            "initial_scale_power": 14,
        },
        "gradient_clipping": 1.0,
        "sparse_gradients": True,
        "steps_per_print": 1,
    }
)

CONFIGS = dict(
    # Tiny BERT
    tiny_bert_rigl_100k_onecycle_lr=tiny_bert_rigl_100k_onecycle_lr,
    tiny_bert_rigl_100k_kd=tiny_bert_rigl_100k_kd,
    tiny_bert_sparse_100k_onecycle_lr_kd=tiny_bert_sparse_100k_onecycle_lr_kd,
    tiny_bert_sparse_300k_onecycle_lr_kd=tiny_bert_sparse_300k_onecycle_lr_kd,
    tiny_bert_kd_onecycle_50k_maxlr_search=tiny_bert_kd_onecycle_50k_maxlr_search,
    tiny_bert_kd_onecycle_100k_pct_start_search=tiny_bert_kd_onecycle_100k_pct_start_search,  # noqa: E501
    tiny_bert_kd_onecycle_300k_pct_start_search=tiny_bert_kd_onecycle_300k_pct_start_search,  # noqa: E501

    # Small BERT
    small_bert_rigl_100k_onecycle_lr=small_bert_rigl_100k_onecycle_lr,

    # BERT Base
    bert_sparse_100k_kd_oncycle_lr=bert_sparse_100k_kd_oncycle_lr,
    bert_sparse_100k_kd_lr_range_test=bert_sparse_100k_kd_lr_range_test,
    finetuning_bert_sparse_kd_oncycle_lr_100k_glue=finetuning_bert_sparse_kd_oncycle_lr_100k_glue,  # noqa: E501

    # Deepspeed
    bert_sparse_100k_kd_lr_range_test_deepspeed=bert_sparse_100k_kd_lr_range_test_deepspeed,  # noqa: E501
)
