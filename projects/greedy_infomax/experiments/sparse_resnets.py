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

import numpy as np
import ray.tune as tune
import torch
from torch.optim.lr_scheduler import OneCycleLR

from nupic.research.frameworks.greedy_infomax.models.ClassificationModel import (
    ClassificationModel,
)
from nupic.research.frameworks.greedy_infomax.models.FullModel import (
    SparseFullVisionModel,
    VDropSparseFullVisionModel,
    FullVisionModel
)
from nupic.research.frameworks.vernon.distributed import mixins
from nupic.torch.modules import SparseWeights2d

from .default_base import CONFIGS as DEFAULT_BASE_CONFIGS
from .default_base import GreedyInfoMaxExperiment


class GreedyInfoMaxExperimentSparse(
    mixins.RezeroWeights, mixins.LogBackpropStructure, GreedyInfoMaxExperiment
):
    pass


DEFAULT_BASE = DEFAULT_BASE_CONFIGS["default_base"]

BATCH_SIZE = 32
NUM_EPOCHS = 5
SPARSE_BASE = deepcopy(DEFAULT_BASE)
SPARSE_BASE.update(
    dict(
        wandb_args=dict(
            project="greedy_infomax-sparsity-tests", name="sparse_resnet_base"
        ),
        experiment_class=GreedyInfoMaxExperimentSparse,
        epochs=NUM_EPOCHS,
        epochs_to_validate=[NUM_EPOCHS - 1],
        supervised_training_epochs_per_validation=10,
        batch_size=BATCH_SIZE,
        model_class=SparseFullVisionModel,
        model_args=dict(
            negative_samples=16,
            k_predictions=5,
            resnet_50=False,
            grayscale=True,
            patch_size=16,
            overlap=2,
            sparse_weights_class=SparseWeights2d,
            sparsity=[0.5, 0.5, 0.5],
            percent_on=[0.9, 0.9, 0.9],
        ),
    )
)


def make_reg_schedule(
    epochs, pct_ramp_start, pct_ramp_end, peak_value, pct_drop, final_value
):
    def reg_schedule(epoch, batch_idx, steps_per_epoch):
        pct = (epoch + batch_idx / steps_per_epoch) / epochs

        if pct < pct_ramp_start:
            return 0.0
        elif pct < pct_ramp_end:
            progress = (pct - pct_ramp_start) / (pct_ramp_end - pct_ramp_start)
            return progress * peak_value
        elif pct < pct_drop:
            return peak_value
        else:
            return final_value

    return reg_schedule


class GreedyInfoMaxExperimentSparsePruning(
    mixins.NoiseRobustnessTest,
    mixins.RegularizeLoss,
    mixins.ConstrainParameters,
    mixins.LogBackpropStructure,
    mixins.PruneLowSNRGlobal,
    GreedyInfoMaxExperiment,
):
    pass


SPARSE_VDROP = deepcopy(SPARSE_BASE)
SPARSE_VDROP.update(
    dict(
        wandb_args=dict(
            project="greedy_infomax-sparsity-tests",
            name="sparse_resnet_vdrop_2"
        ),
        experiment_class=GreedyInfoMaxExperimentSparsePruning,
        epochs=30,
        epochs_to_validate=[0, 3, 6, 9, 12, 15, 19, 23, 27, 29],
        model_class=VDropSparseFullVisionModel,
        model_args=dict(
            negative_samples=16,
            k_predictions=5,
            resnet_50=False,
            block_dims=None,
            num_channels=None,
            grayscale=True,
            patch_size=16,
            overlap=2,
            # percent_on=None,
        ),
        prune_schedule=[
            (8, 0.8),
            (14, 0.6),
            (20, 0.4),
            (25, 0.2),
        ],
        log_module_sparsities=True,
        reg_scalar=make_reg_schedule(
            epochs=30,
            pct_ramp_start=2/20,
            pct_ramp_end=4/20,
            peak_value=0.001,
            pct_drop=18/20,
            final_value=0.0005,
        ),
        lr_scheduler_class=OneCycleLR,
        lr_scheduler_args=dict(
            max_lr=0.0005,
            div_factor=50,  # Optimized in Sig-Opt
            final_div_factor=2000,
            pct_start=0.2,  # Optimized in Sig-Opt
            epochs=20,
            anneal_strategy="linear",
            max_momentum=0.01,
            cycle_momentum=False,
        ),
        # batches_in_epoch=1,
        # batches_in_epoch_supervised=1,
        # batches_in_epoch_val=1,
        # supervised_training_epochs_per_validation=1,
        # batch_size=16,
    )
)

LARGE_SPARSE = deepcopy(SPARSE_BASE)
NUM_CLASSES = 10
LARGE_SPARSE.update(
    dict(
        wandb_args=dict(project="greedy_infomax-sparsity-tests", name="large_sparse"),
        experiment_class=GreedyInfoMaxExperimentSparse,
        epochs=NUM_EPOCHS,
        epochs_to_validate=[NUM_EPOCHS - 1],
        batch_size=16,
        supervised_training_epochs_per_validation=10,
        model_class=SparseFullVisionModel,
        model_args=dict(
            negative_samples=16,
            k_predictions=5,
            resnet_50=False,
            grayscale=True,
            patch_size=16,
            overlap=2,
            block_dims=[3, 4, 6],
            num_channels=[512, 512, 512],
            sparse_weights_class=SparseWeights2d,
            sparsity=[0.128] * 3,
            percent_on=[0.51] * 3,
        ),
        classifier_config=dict(
            model_class=ClassificationModel,
            model_args=dict(in_channels=512, num_classes=NUM_CLASSES),
            loss_function=torch.nn.functional.cross_entropy,
            # Classifier Optimizer class. Must inherit from "torch.optim.Optimizer"
            optimizer_class=torch.optim.Adam,
            # Optimizer class class arguments passed to the constructor
            optimizer_args=dict(lr=2e-4),
        ),
    )
)


LARGE_SPARSE_GRID_SEARCH = deepcopy(LARGE_SPARSE)
LARGE_SPARSE_GRID_SEARCH.update(
    dict(
        wandb_args=dict(project="greedy_infomax-large_sparse", name="gridsearch"),
        experiment_class=GreedyInfoMaxExperimentSparse,
        epochs=NUM_EPOCHS,
        epochs_to_validate=[NUM_EPOCHS - 1],
        batch_size=16,
        supervised_training_epochs_per_validation=10,
        model_class=VDropSparseFullVisionModel,
        model_args=dict(
            negative_samples=16,
            k_predictions=5,
            resnet_50=False,
            grayscale=True,
            patch_size=16,
            overlap=2,
            block_dims=[3, 4, 6],
            num_channels=tune.grid_search(
                [
                    [64, 128, 256],
                    [128, 128, 256],
                    [128, 256, 256],
                    [256, 256, 256],
                    [256, 512, 256],
                    [512, 512, 256],
                ]
            ),
            sparse_weights_class=SparseWeights2d,
            sparsity=None,
            percent_on=tune.grid_search(np.logspace(-2, -0.3, num=5)),
        ),
        classifier_config=dict(
            model_class=ClassificationModel,
            model_args=dict(in_channels=256, num_classes=NUM_CLASSES),
            loss_function=torch.nn.functional.cross_entropy,
            # Classifier Optimizer class. Must inherit from "torch.optim.Optimizer"
            optimizer_class=torch.optim.Adam,
            # Optimizer class class arguments passed to the constructor
            optimizer_args=dict(lr=2e-4),
        ),
    )
)


CONFIGS = dict(
    sparse_base=SPARSE_BASE,
    large_sparse=LARGE_SPARSE,
    sparse_vdrop=SPARSE_VDROP,
    large_sparse_grid_search=LARGE_SPARSE_GRID_SEARCH,
)