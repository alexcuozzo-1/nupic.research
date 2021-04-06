# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
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

import io
import sys
import time
from pprint import pformat

import torch
from torch.backends import cudnn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from nupic.research.frameworks.pytorch.lr_scheduler import ComposedLRScheduler
from nupic.research.frameworks.pytorch.model_utils import (
    deserialize_state_dict,
    evaluate_model,
    serialize_state_dict,
    train_model,
)
from nupic.research.frameworks.vernon.experiment_utils import create_lr_scheduler
from nupic.research.frameworks.vernon.experiments import SupervisedExperiment
from nupic.research.frameworks.vernon.network_utils import (
    create_model,
    get_compatible_state_dict,
)

try:
    from apex import amp
except ImportError:
    amp = None


__all__ = [
    "SelfSupervisedExperiment",
]


# Improves performance when using fixed size images (224) and CNN
cudnn.benchmark = True


class SelfSupervisedExperiment(SupervisedExperiment):
    """
    General experiment class used to train neural networks in self-supervised learning
    tasks.

    Self-supervised experiments have three important dataset splits: train, test,
    and supervised. The training set consists of unlabeled data for representation
    learning, the supervised set consists of a typically smaller amount of labeled
    data for which to train a classifier, and the test set is used to evaluate the
    classifier.

    The validation step trains a new classifier on top of the existing frozen model,
    and then proceeds to test this classifier on the test set. The number of
    supervised training epochs to train for each validation is given by
    supervised_training_epochs_per_validation.
    """

    def __init__(self, config):
        super(SelfSupervisedExperiment, self).__init__()
        self.unsupervised_loader = None
        self.supervised_loader = None
        self.supervised_training_epochs_per_validation = 3

    def create_loaders(self, config):
        super(SelfSupervisedExperiment, self).create_loaders(config)
        self.supervised_loader = self.create_supervised_loader(config)
        self.unsupervised_loader = self.create_unsupervised_loader(config)


    @classmethod
    def create_unsupervised_loader(cls, config, dataset=None):
        if dataset is None:
            dataset = cls.load_dataset(config, split='unsupervised')
        return super(SelfSupervisedExperiment, cls).create_train_dataloader(config,
                                                                         dataset)

    @classmethod
    def create_supervised_dataloader(cls, config, dataset=None):
        if dataset is None:
            dataset = cls.load_dataset(config, split='supervised')
        return super().create_train_dataloader(config, dataset)

    @classmethod
    def create_validation_dataloader(cls, config, dataset=None):
        if dataset is None:
            dataset = cls.load_dataset(config, split='test')
        return super().create_train_dataloader(config, dataset)


    @classmethod
    def load_dataset(cls, config, split="unsupervised"):
        dataset_class = config.get("dataset_class", None)
        if dataset_class is None:
            raise ValueError("Must specify 'dataset_class' in config.")

        if split == "unsupervised":
            dataset_args = dict(config.get("train_dataset_args", {"split":"unlabeled"}))
        elif split == "supervised":
            dataset_args = dict(config.get("supervised_dataset_args",
                                           {"split":"train"}))
        elif split == "test":
            dataset_args = dict(config.get("test_dataset_args", {"split":"test"}))
        return dataset_class(**dataset_args)

    def setup_experiment(self, config):
        """
        Configure the experiment for training
        :param config: Dictionary containing the configuration parameters
            - data: Dataset path
            - progress: Show progress during training
            - train_dir: Dataset training data relative path
            - batch_size: Training batch size
            - val_dir: Dataset validation data relative path
            - val_batch_size: Validation batch size
            - workers: how many data loading processes to use
            - train_loader_drop_last: Whether to skip last batch if it is
                                      smaller than the batch size
            - num_classes: Limit the dataset size to the given number of classes
            - model_class: Model class. Must inherit from "torch.nn.Module"
            - model_args: model model class arguments passed to the constructor
            - init_batch_norm: Whether or not to Initialize running batch norm
                               mean to 0.
            - optimizer_class: Optimizer class.
                               Must inherit from "torch.optim.Optimizer"
            - optimizer_args: Optimizer class class arguments passed to the
                              constructor
            - lr_scheduler_class: Learning rate scheduler class.
                                 Must inherit from "_LRScheduler"
            - lr_scheduler_args: Learning rate scheduler class class arguments
                                 passed to the constructor
            - lr_scheduler_step_every_batch: Whether to step the lr-scheduler after
                                             after every batch (e.g. for OneCycleLR)
            - loss_function: Loss function. See "torch.nn.functional"
            - epochs: Number of epochs to train
            - batches_in_epoch: Number of batches per epoch.
                                Useful for debugging
            - batches_in_epoch_val: Number of batches per epoch in validation.
                                   Useful for debugging
            - mixed_precision: Whether or not to enable apex mixed precision
            - mixed_precision_args: apex mixed precision arguments.
                                    See "amp.initialize"
            - sample_transform: Transform acting on the training samples. To be used
                                additively after default transform or auto-augment.
            - target_transform: Transform acting on the training targets.
            - replicas_per_sample: Number of replicas to create per sample in the batch.
                                   (each replica is transformed independently)
                                   Used in maxup.
            - train_model_func: Optional user defined function to train the model,
                                expected to behave similarly to `train_model`
                                in terms of input parameters and return values
            - evaluate_model_func: Optional user defined function to validate the model
                                   expected to behave similarly to `evaluate_model`
                                   in terms of input parameters and return values
            - checkpoint_file: if not None, will start from this model. The model
                               must have the same model_args and model_class as the
                               current experiment.
            - load_checkpoint_args: args to be passed to `load_state_from_checkpoint`
            - epochs_to_validate: list of epochs to run validate(). A -1 asks
                                  to run validate before any training occurs.
                                  Default: last three epochs.
            - launch_time: time the config was created (via time.time). Used to report
                           wall clock time until the first batch is done.
                           Default: time.time() in this setup_experiment().
        """

        self.launch_time = config.get("launch_time", time.time())
        super().setup_experiment(config)
        self.supervised_training_epochs_per_validation = config.get(
            "supervised_training_epochs_per_validation", 3)



