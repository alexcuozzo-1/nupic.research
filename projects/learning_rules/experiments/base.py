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
Base GSC Experiment configuration.
"""

import ray.tune as tune

import os
import sys
from copy import deepcopy

import numpy as np
import torch

from nupic.research.frameworks.pytorch.datasets import preprocessed_gsc
from projects.learning_rules.synthetic_gradients.sample_network import SparseSyntheticMLP
from nupic.research.frameworks.vernon.distributed import experiments, mixins
from torchvision.datasets import MNIST
from torchvision import transforms




class SupervisedSyntheticGradientsExperiment(mixins.RezeroWeights,
                                    mixins.UpdateBoostStrength,
                                    mixins.LogEveryLoss,
                                    experiments.SupervisedExperiment):
    pass



# Batch size depends on the GPU memory.
BATCH_SIZE = 16

# Default configuration, uses a small MLP on MNIST
DEFAULT_BASE = dict(
    experiment_class=SupervisedSyntheticGradientsExperiment,

    # Results path
    local_dir=os.path.expanduser("~/nta/results/experiments/learning_rules"),

    # Dataset
    dataset_class=MNIST,
    dataset_args=dict(
        root="~/nta/data/mnist",
        download=True,
        transform=transforms.ToTensor(),
    ),

    wandb_args=dict(
        project="synthetic_gradients",
        name="mnist_initial_test",
    ),

    # Seed
    seed=tune.sample_from(lambda spec: np.random.randint(1, 10000)),

    # Number of times to sample from the hyperparameter space. If `grid_search` is
    # provided the grid will be repeated `num_samples` of times.
    num_samples=1,

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
    model_class=SparseSyntheticMLP,

    # Model class arguments passed to the constructor
    model_args=dict(
        input_shape=(1, 28, 28),
        num_classes=10,
        linear_units=(1000, 1000),
        linear_percent_on=(0.1, 0.1),
        linear_weight_density=(0.5, 0.5),
        use_batchnorm=True,
        use_softmax=True,
        kwinners_layers=(False, False),
        boost_strength=1.5,
        boost_strength_factor=0.9,
        k_inference_factor=1.5,
        duty_cycle_period=1000,
        synthetic_gradients=(True, True),
    ),

    # Optimizer class. Must inherit from "torch.optim.Optimizer"
    optimizer_class=torch.optim.SGD,

    # Optimizer class class arguments passed to the constructor
    optimizer_args=dict(
        lr=0.01,
        weight_decay=0.01,
        momentum=0.0,
    ),

    # Learning rate scheduler class. Must inherit from "_LRScheduler"
    lr_scheduler_class=torch.optim.lr_scheduler.StepLR,

    # Learning rate scheduler class class arguments passed to the constructor
    lr_scheduler_args=dict(
        gamma=0.9,
        step_size=1,
    ),

    # Loss function. See "torch.nn.functional"
    loss_function=torch.nn.functional.nll_loss,

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

GSC_BASE = deepcopy(DEFAULT_BASE)
GSC_BASE.update(
    # Dataset
    dataset_class=preprocessed_gsc,
    dataset_args=dict(
        root="~/nta/data/gsc_preprocessed",
        download=True,
    ),
    # Model class. Must inherit from "torch.nn.Module"
    model_class=SparseSyntheticMLP,

    # Model class arguments passed to the constructor
    model_args=dict(
        input_shape=(1, 32, 32),
        num_classes=12,
        linear_units=(1000, 1000),
        linear_percent_on=(0.1, 0.1),
        linear_weight_density=(0.5, 0.5),
        use_batchnorm=True,
        use_softmax=True,
        kwinners_layers=(False, False),
        boost_strength=1.5,
        boost_strength_factor=0.9,
        k_inference_factor=1.5,
        duty_cycle_period=1000,
        synthetic_gradients=(True, True),
    ),
)



# Export configurations in this file
CONFIGS = dict(
    default_base=DEFAULT_BASE,
    gsc_base = GSC_BASE,
)
