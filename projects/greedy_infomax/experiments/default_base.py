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
from torchvision.datasets import STL10, FakeData
from torchvision import transforms
from ray import tune
import sys
from nupic.research.frameworks.greedy_infomax.models.FullModel import FullVisionModel


class SelfSupervisedGreedyInfoMaxExperiment(
    mixins.LogEveryLoss,
    experiments.SelfSupervisedExperiment,
):
    def __init__(self, config):
        super(SelfSupervisedGreedyInfoMaxExperiment, self).__init__()
        self.supervised_loader = None
        self.prediction_step = config.get("prediction_step", 5)

    @classmethod
    def create_optimizer(cls, config, model):
        return super().create_optimizer(config, model.encoder)



# get transforms for the dataset
def get_transforms(eval=False, aug=None):
    trans = []

    if aug["randcrop"] and not eval:
        trans.append(transforms.RandomCrop(aug["randcrop"]))

    if aug["randcrop"] and eval:
        trans.append(transforms.CenterCrop(aug["randcrop"]))

    if aug["flip"] and not eval:
        trans.append(transforms.RandomHorizontalFlip())

    if aug["grayscale"]:
        trans.append(transforms.Grayscale())
        trans.append(transforms.ToTensor())
        trans.append(transforms.Normalize(mean=aug["bw_mean"], std=aug["bw_std"]))
    elif aug["mean"]:
        trans.append(transforms.ToTensor())
        trans.append(transforms.Normalize(mean=aug["mean"], std=aug["std"]))
    else:
        trans.append(transforms.ToTensor())

    trans = transforms.Compose(trans)
    return trans


aug = {
        "stl10": {
            "randcrop": 64,
            "flip": True,
            "grayscale": True,
            "mean": [0.4313, 0.4156, 0.3663],  # values for train+unsupervised combined
            "std": [0.2683, 0.2610, 0.2687],
            "bw_mean": [0.4120],  # values for train+unsupervised combined
            "bw_std": [0.2570],
        }  # values for labeled train set: mean [0.4469, 0.4400, 0.4069], std [0.2603, 0.2566, 0.2713]
}
transform_unsupervised = transforms.Compose([
    get_transforms(eval=False, aug=aug["stl10"])])
transform_valid = transforms.Compose([get_transforms(eval=True, aug=aug["stl10"])])


base_dataset_args=dict(
        root="~/nta/data/STL10",
        download=False,
)

#fake data class for debugging purposes
def fake_data(size=256, image_size=(3, 96, 96), num_classes = 10, train=True,
              transform=transform_valid):
    return FakeData(size=size, image_size=image_size, num_classes=num_classes,
    transform=transform)

base_dataset_args=dict(
        root="~/nta/data/",
        download=False,
)

unsupervised_dataset_args = deepcopy(base_dataset_args)
unsupervised_dataset_args.update(dict(transform=transform_unsupervised))
supervised_dataset_args = deepcopy(unsupervised_dataset_args)
test_dataset_args = deepcopy(base_dataset_args)
test_dataset_args.update(transform_valid)


BATCH_SIZE = 32
DEFAULT_BASE = dict(
    experiment_class=SelfSupervisedGreedyInfoMaxExperiment,

    # Dataset
    dataset_class=fake_data,
    # train_dataset_args=train_dataset_args,
    # supervised_dataset_args=supervised_dataset_args,
    # test_dataset_args=test_dataset_args,
    workers=1,

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
    epochs=300,
    epochs_to_validate=[],

    # Which epochs to run and report inference over the validation dataset.
    # epochs_to_validate=range(-1, 30),  # defaults to the last 3 epochs

    # Model class. Must inherit from "torch.nn.Module"
    model_class=FullVisionModel,
    #default model arguments
    model_args = dict(calc_loss=True,
                      negative_samples=16,
                      model_splits=3,
                      supervised=False,
                      resnet_50=False,
                      grayscale=True,),

    #Greedy InfoMax args
    greedy_infomax_args = dict(
        # time steps to predict into future
        prediction_step = 5,
        # number of module splits
        modules=3, # number of module splits
        ),

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
