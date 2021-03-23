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

import numpy as np
import torch
from torch import nn

from nupic.torch.modules import Flatten, KWinners, KWinners2d, SparseWeights
from .decoupled_neural_interface import BackwardInterface, BasicSynthesizer


class SparseSyntheticMLP(nn.Sequential):
    """
    :param linear_units: a tuple of integers representing the number of units in each
                         hidden linear layer
    :param linear_percent_on: a tuple of floats representing the percent of units
                              allowed to remain on in linear layer
    :param linear_weight_density: a tuple of floats representing percent of weights
                                  that are allowed to be non-zero in each linear layer
    :param k_inference_factor: During inference (training=False) we increase
                               `percent_on` in all sparse layers by this factor
    :param boost_strength: boost strength (0.0 implies no boosting)
    :param boost_strength_factor: Boost strength factor to use [0..1]
    :param duty_cycle_period: The period used to calculate duty cycles
    :param synthetic_gradients: a tuple of booleans representing which layers have
                                synthetic gradients
    """

    def __init__(
        self,
        input_shape = (1, 32, 32),
        num_classes = 12,
        linear_units=(1000, 1000),
        linear_percent_on=(0.1, 0.1),
        linear_weight_density=(0.5, 0.5),
        use_batchnorm = True,
        use_softmax = True,
        kwinners_layers = (False, False),
        boost_strength=1.5,
        boost_strength_factor=0.9,
        k_inference_factor=1.5,
        duty_cycle_period=1000,
        synthetic_gradients=(True, True),
    ):
        super(SparseSyntheticMLP, self).__init__()

        self.add_module("flatten", Flatten())
        # Add Linear layers
        current_input_shape = int(torch.prod(torch.Tensor(input_shape)))
        for i in range(len(linear_units)):
            num_units =  linear_units[i]
            weight_density = linear_weight_density[i]
            percent_on = linear_percent_on[i]
            use_synthetic_gradients = synthetic_gradients[i]
            use_kwinners = kwinners_layers[i]

            self.add_module(f"linear_{i+1}", SparseWeights(
                nn.Linear(current_input_shape, num_units),
                weight_sparsity= 1 - weight_density))
            if use_batchnorm:
                self.add_module(f"linear_bn_{i+1}", nn.BatchNorm1d(num_units,
                                                                affine=False))
            if use_kwinners:
                self.add_module(f"linear_kwinner_{i+1}", KWinners(
                    n=num_units,
                    percent_on=percent_on,
                    k_inference_factor=k_inference_factor,
                    boost_strength=boost_strength,
                    boost_strength_factor=boost_strength_factor,
                    duty_cycle_period=duty_cycle_period))
            if use_synthetic_gradients:
                self.add_module(f"synthetic_gradients_{i+1}", BackwardInterface(
                    BasicSynthesizer(num_units, n_hidden=1, hidden_dim=num_units)
                ))
            current_input_shape = num_units

        # Classifier
        self.add_module("output", nn.Linear(current_input_shape, num_classes))
        if use_softmax:
            self.add_module("softmax", nn.LogSoftmax(dim=1))




class MNISTSyntheticFCN(nn.Sequential):
    """
    :param linear_units: a tuple of integers representing the number of units in each
                         hidden linear layer
    :param synthetic_gradients: a tuple of booleans representing which layers have
                                synthetic gradients
    :param use_context: a boolean, representing whether a context vector should be
                        used for conditioning the synthetic gradients modules
    :param synthetic_gradient_linear units:a tuple of integers representing the
                                           number of units in each hidden layer of
                                           the synthetic gradients module
    :param use_batchnorm: boolean, defaults to True
    :param use_softmax: boolean, defaults to False
    """

    def __init__(
        self,
        input_shape = (1, 28, 28),
        num_classes = 10,
        linear_units=(256, 256, 256),
        use_batchnorm = True,
        use_softmax = False,
        synthetic_gradients_layers=(True, True, True),
        synthetic_gradients_n_hidden=2,
        synthetic_gradients_hidden_dim = 1024,
        use_context = False,
    ):
        super(MNISTSyntheticFCN, self).__init__()

        assert len(linear_units) == len(synthetic_gradients_layers)
        self.add_module("flatten", Flatten())
        # Add Linear layers
        current_input_shape = int(torch.prod(torch.Tensor(input_shape)))
        for i in range(len(linear_units)):
            num_units =  linear_units[i]
            use_synthetic_gradients = synthetic_gradients_layers[i]

            self.add_module(f"linear_{i+1}", nn.Linear(current_input_shape, num_units))
            if use_batchnorm:
                self.add_module(f"linear_bn_{i+1}", nn.BatchNorm1d(num_units))
            self.add_module(f"relu_{i+1}", nn.ReLU())

            if use_synthetic_gradients:
                context_dim = num_classes if use_context else None
                self.add_module(f"synthetic_gradients_{i+1}", BackwardInterface(
                    BasicSynthesizer(num_units,
                                     n_hidden=synthetic_gradients_n_hidden,
                                     hidden_dim=synthetic_gradients_hidden_dim,
                                     context_dim=context_dim)
                ))
            current_input_shape = num_units

        # Classifier
        self.add_module("output", nn.Linear(current_input_shape, num_classes))
        if use_softmax:
            self.add_module("softmax", nn.LogSoftmax(dim=1))

