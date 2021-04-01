#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see http://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#


import os
from copy import deepcopy

import numpy as np
import torch

from nupic.research.frameworks.vernon.distributed import experiments, mixins
from torchvision.datasets import STL10

from .base import DEFAULT_BASE
from .repository_code import customTrainLoop


class SelfSupervisedGreedyInfoMaxExperiment(
    mixins.LogEveryLoss,
    #validateLinearProbe,
    customTrainLoop,
    experiments.SupervisedExperiment,
):
    pass


BATCH_SIZE = 32
DEFAULT_BASE = dict(
    experiment_class=SelfSupervisedGreedyInfoMaxExperiment,

    # Dataset
    dataset_class=STL10,
    dataset_args=dict(
        root="~/nta/data/STL10",
        download=False,
    ),

    # Seed
    seed=tune.sample_from(lambda spec: np.random.randint(1, 10000)),

    # Number of times to sample from the hyperparameter space. If `grid_search` is
    # provided the grid will be repeated `num_samples` of times.

    # Training batch size
    batch_size=BATCH_SIZE,
    # Validation batch size
    val_batch_size=10000,
    # Number of batches per epoch. Useful for debugging
    batches_in_epoch=sys.maxsize,

    # Update this to stop training when accuracy reaches the metric value
    # For example, stop=dict(mean_accuracy=0.75),
    stop=dict(),

    # Number of epochs
    epochs=30,
    epochs_to_validate=range(0, 30),

    # Which epochs to run and report inference over the validation dataset.
    # epochs_to_validate=range(-1, 30),  # defaults to the last 3 epochs

    # Model class. Must inherit from "torch.nn.Module"
    model_class=GIMResNet,

    # Optimizer class. Must inherit from "torch.optim.Optimizer"
    optimizer_class=torch.optim.Adam,

    # Optimizer class class arguments passed to the constructor
    optimizer_args=dict(
        lr=1.5e-4,
    ),

    # Learning rate scheduler class. Must inherit from "_LRScheduler"
    lr_scheduler_class=torch.optim.lr_scheduler.StepLR,

    # How often to checkpoint (epochs)
    checkpoint_freq=0,
    keep_checkpoints_num=1,
    checkpoint_at_end=True,
    checkpoint_score_attr="training_iteration",

    # How many times to try to recover before stopping the trial
    max_failures=3,

    # How many times to retry the epoch before stopping. This is useful when
    # using distributed training with spot instances.
    max_retries=3,

    # Python Logging level : "critical", "error", "warning", "info", "debug"
    log_level="debug",

    # Python Logging Format
    log_format="%(asctime)s:%(levelname)s:%(name)s:%(message)s",

    # Ray tune verbosity. When set to the default value of 2 it will log
    # iteration result dicts. This dict can flood the console if it contains
    # large data structures, so default to verbose=1. The SupervisedTrainable logs
    # a succinct version of the result dict.
    verbose=1,
)


CONFIGS = dict(
    default_base = DEFAULT_BASE
)
