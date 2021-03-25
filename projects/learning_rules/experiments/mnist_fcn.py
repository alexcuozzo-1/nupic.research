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
from projects.learning_rules.synthetic_gradients.sample_network import \
    SparseSyntheticMLP, MNISTSyntheticFCN
from nupic.research.frameworks.vernon.distributed import experiments, mixins
from torchvision.datasets import MNIST
from torchvision import transforms
from .base import SupervisedSyntheticGradientsExperiment

"""
Common Details 

All experiments are run for 500k iterations and optimised with Adam (Kingma & Ba, 2014)
with batch size of 256. The learning rate was initialised at 3 × 10−5 and decreased 
by a factor of 10 at 300k and 400k steps. Note the number of iterations, learning rate,
and learning rate schedule was not optimised. We perform
a hyperparameter search over the number of hidden layers
in the synthetic gradient model (from 0 to 2, where 0 means
we use a linear model such that ˆδ = M(h) = φwh + φb)
and select the best number of layers for each experiment
type (given below) based on the final test performance. We
used cross entropy loss for classification and L2 loss for
synthetic gradient regression which was weighted by a factor of 1 with respect to 
the classification loss. All input data
was scaled to [0, 1] interval. The final regression layer of all
synthetic gradient models are initialised with zero weights
and biases, so initially, zero synthetic gradient is produced.
"""

"""
MNIST FCN 

Every hidden layer consists of fully connected layers with 256 units, 
followed by batchnormalisation and ReLU non-linearity. The synthetic gradient 
models consists of two (DNI) or zero (cDNI) hidden layers and with 1024 units (
linear, batch-normalisation, ReLU) followed by a final linear layer with 256 units.
"""

class SyntheticGradientsMNISTExperiment(mixins.LogEveryLoss,
                                        experiments.SupervisedExperiment):
    pass


# From original paper
BATCH_SIZE = 256

# Approximately 200 iterations per epoch
# Paper calls for 500k iterations, which would be
# 500k/200 = 2.5k epochs! that's too much, starting with 750
NUM_EPOCHS = 750

# Default configuration, uses a small MLP on MNIST
MNIST_FCN_3 = dict(
    experiment_class=SupervisedSyntheticGradientsExperiment,

    # Results path
    local_dir=os.path.expanduser("~/nta/results/experiments/learning_rules"),

    # Dataset
    dataset_class=MNIST,
    dataset_args=dict(
        root="~/nta/data/",
        download=False,
        transform=transforms.ToTensor(),
    ),

    wandb_args=dict(
        project="synthetic_gradients_MNIST_FCN",
        name="3-layer",
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
    epochs=NUM_EPOCHS,
    epochs_to_validate=range(0, NUM_EPOCHS),

    # Which epochs to run and report inference over the validation dataset.
    # epochs_to_validate=range(-1, 30),  # defaults to the last 3 epochs

    # Model class. Must inherit from "torch.nn.Module"
    model_class=MNISTSyntheticFCN,

    # Model class arguments passed to the constructor
    model_args=dict(
        input_shape=(1, 28, 28),
        num_classes=10,
        linear_units=(256, 256, 256),
        use_batchnorm=True,
        use_softmax=False,
        synthetic_gradients_layers=(True, True, True),
        synthetic_gradients_n_hidden=2,
        synthetic_gradients_hidden_dim=1024,
        use_context=False,
    ),

    # Optimizer class. Must inherit from "torch.optim.Optimizer"
    optimizer_class=torch.optim.Adam,

    # Optimizer class class arguments passed to the constructor
    optimizer_args=dict(
        lr=3 * 1e-5,
    ),

    # Learning rate scheduler class. Must inherit from "_LRScheduler"
    lr_scheduler_class=torch.optim.lr_scheduler.MultiStepLR,

    # Learning rate scheduler class class arguments passed to the constructor
    lr_scheduler_args=dict(
        gamma=1 / 3. ,
        milestones=[450, 600]
    ),

    # Loss function. See "torch.nn.functional"
    loss_function=torch.nn.functional.cross_entropy,

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

MNIST_FCN_3_NO_SYNTHETIC = deepcopy(MNIST_FCN_3)
MNIST_FCN_3_NO_SYNTHETIC.update(
    wandb_args=dict(
        project="synthetic_gradients_MNIST_FCN",
        name="3-layer-no-synthetic",
    ),

    model_args=dict(
            input_shape=(1, 28, 28),
            num_classes=10,
            linear_units=(256, 256, 256),
            use_batchnorm=True,
            use_softmax=False,
            synthetic_gradients_layers=(False, False, False),
            synthetic_gradients_n_hidden=2,
            synthetic_gradients_hidden_dim=1024,
            use_context=False,
        )
)

MNIST_FCN_4 = deepcopy(MNIST_FCN_3)
MNIST_FCN_4.update(
    wandb_args=dict(
        project="synthetic_gradients_MNIST_FCN",
        name="4-layer",
    ),
    model_args=dict(
            input_shape=(1, 28, 28),
            num_classes=10,
            linear_units=(256, 256, 256, 256),
            use_batchnorm=True,
            use_softmax=False,
            synthetic_gradients_layers=(True, True, True, True),
            synthetic_gradients_n_hidden=2,
            synthetic_gradients_hidden_dim=1024,
            use_context=False,
        ),
)


MNIST_FCN_5 = deepcopy(MNIST_FCN_3)
MNIST_FCN_5.update(
    wandb_args=dict(
        project="synthetic_gradients_MNIST_FCN",
        name="5-layer",
    ),
    model_args=dict(
            input_shape=(1, 28, 28),
            num_classes=10,
            linear_units=(256, 256, 256, 256, 256),
            use_batchnorm=True,
            use_softmax=False,
            synthetic_gradients_layers=(True, True, True, True, True),
            synthetic_gradients_n_hidden=2,
            synthetic_gradients_hidden_dim=1024,
            use_context=False,
        ),
)


MNIST_FCN_6 = deepcopy(MNIST_FCN_3)
MNIST_FCN_6.update(
    wandb_args=dict(
        project="synthetic_gradients_MNIST_FCN",
        name="6-layer",
    ),
    model_args=dict(
            input_shape=(1, 28, 28),
            num_classes=10,
            linear_units=(256, 256, 256, 256, 256, 256),
            use_batchnorm=True,
            use_softmax=False,
            synthetic_gradients_layers=(True, True, True, True, True, True),
            synthetic_gradients_n_hidden=2,
            synthetic_gradients_hidden_dim=1024,
            use_context=False,
        ),
)


# Export configurations in this file
CONFIGS = dict(
    #MNIST FCN w/ Synthetic Gradients, no context
    mnist_fcn_3=MNIST_FCN_3,
    mnist_fcn_4=MNIST_FCN_4,
    mnist_fcn_5=MNIST_FCN_5,
    mnist_fcn_6=MNIST_FCN_6,

    #MNIST FCT w/o Synthetic Gradients
    mnist_fcn_3_no_synthetic = MNIST_FCN_3_NO_SYNTHETIC,
)
