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
#
# This work was based on the original Greedy InfoMax codebase from Sindy Lowe:
# https://github.com/loeweX/Greedy_InfoMax
# The Greedy InfoMax paper can be found here:
# https://arxiv.org/abs/1905.11786
# ----------------------------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F
from nupic.research.frameworks.backprop_structure.modules import VDropConv2d

from nupic.research.frameworks.greedy_infomax.models.BilinearInfo import (
    BilinearInfo,
    SparseBilinearInfo,
    VDropSparseBilinearInfo
)
from nupic.torch.modules import KWinners2d, SparseWeights2d


class PreActBlockNoBN(nn.Module):
    """Pre-activation version of the BasicBlock."""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlockNoBN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1
        )
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes, kernel_size=1, stride=stride
                )
            )

    def forward(self, x):
        out = F.relu(x)
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += shortcut
        return out


class SparsePreActBlockNoBN(PreActBlockNoBN):
    """Sparse version of the PreActBlockNoBN block."""

    def __init__(self,
                 in_planes,
                 planes,
                 stride=1,
                 sparse_weights_class=SparseWeights2d,
                 sparsity=0.2,
                 percent_on=0.5):
        super(SparsePreActBlockNoBN, self).__init__(in_planes, planes, stride=stride)
        if sparsity > 0.3:
            self.conv1 = sparse_weights_class(self.conv1, sparsity=sparsity)
            self.conv2 = sparse_weights_class(self.conv2, sparsity=sparsity)
            if hasattr(self, "shortcut"):
                self.shortcut = nn.Sequential(
                    sparse_weights_class(self.shortcut._modules["0"], sparsity=sparsity)
                )
        if percent_on >= 0.5:
            self.nonlinearity1 = F.relu
            self.nonlinearity2 = F.relu
        else:
            self.nonlinearity1 = KWinners2d(in_planes, percent_on=percent_on)
            self.nonlinearity2 = KWinners2d(planes, percent_on=percent_on)

    def forward(self, x):
        out = self.nonlinearity1(x)
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.nonlinearity2(out)
        out = self.conv2(out)
        out += shortcut
        return out

class VDropSparsePreActBlockNoBN(nn.Module):
    """VDrop version of the PreActBlockNoBN block."""
    expansion = 1
    def __init__(self,
                 in_planes,
                 planes,
                 stride=1,
                 percent_on=0.5,
                 central_data=None):
        super(VDropSparsePreActBlockNoBN, self).__init__()
        self.conv1 = VDropConv2d(
            in_planes,
            planes,
            kernel_size=3,
            central_data=central_data,
            stride=stride,
            padding=1,
        )
        self.conv2 = VDropConv2d(planes,
                                 planes,
                                 kernel_size=3,
                                 central_data=central_data,
                                 stride=1,
                                 padding=1)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                VDropConv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    central_data=central_data,
                    stride=stride
                )
            )
        if percent_on >= 0.5:
            self.nonlinearity1 = F.relu
            self.nonlinearity2 = F.relu
        else:
            self.nonlinearity1 = KWinners2d(in_planes, percent_on=percent_on)
            self.nonlinearity2 = KWinners2d(planes, percent_on=percent_on)

    def forward(self, x):
        out = self.nonlinearity1(x)
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.nonlinearity2(out)
        out = self.conv2(out)
        out += shortcut
        return out




class PreActBottleneckNoBN(nn.Module):
    """Pre-activation version of the original Bottleneck module."""

    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneckNoBN, self).__init__()
        # self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        # self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1)
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes, kernel_size=1, stride=stride
                )
            )

    def forward(self, x):
        out = F.relu(x)
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.relu(out))
        out = self.conv3(F.relu(out))
        out += shortcut
        return out


class SparsePreActBottleneckNoBN(PreActBottleneckNoBN):
    """Pre-activation version of the original Bottleneck module."""

    def __init__(self,
                 in_planes,
                 planes,
                 stride=1,
                 sparse_weights_class=SparseWeights2d,
                 sparsity=0.5,
                 percent_on=0.2):
        super(SparsePreActBottleneckNoBN, self).__init__(
            in_planes, planes, stride=stride
        )

        if sparsity > 0.3:
            self.conv1 = sparse_weights_class(self.conv1, sparsity=sparsity)
            self.conv2 = sparse_weights_class(self.conv2, sparsity=sparsity)
            self.conv3 = sparse_weights_class(self.conv3, sparsity=sparsity)
            if hasattr(self, "shortcut"):
                self.shortcut = nn.Sequential(
                    sparse_weights_class(self.shortcut._modules["0"], sparsity=sparsity)
                )
        if percent_on >= 0.5:
            self.nonlinearity1 = F.relu
            self.nonlinearity2 = F.relu
            self.nonlinearity3 = F.relu
        else:
            self.nonlinearity1 = KWinners2d(in_planes, percent_on=percent_on)
            self.nonlinearity2 = KWinners2d(planes, percent_on=percent_on)
            self.nonlinearity3 = KWinners2d(planes, percent_on=percent_on)


    def forward(self, x):
        out = self.nonlinearity1(x)
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.nonlinearity1(out)
        out = self.conv2(out)
        out = self.nonlinearity3(out)
        out = self.conv3(out)
        out += shortcut
        return out


class VDropSparsePreActBottleneckNoBN(nn.Module):
    """Pre-activation version of the original Bottleneck module."""

    expansion = 4

    def __init__(self, in_planes, planes, stride=1, percent_on=0.5, central_data=None):
        super(VDropSparsePreActBottleneckNoBN, self).__init__()
        self.conv1 = VDropConv2d(in_planes,
                                 planes,
                                 kernel_size=1,
                                 central_data=central_data)
        self.conv2 = VDropConv2d(planes,
                               planes,
                               kernel_size=3,
                               central_data=central_data,
                               stride=stride,
                               padding=1)
        self.conv3 = VDropConv2d(planes,
                                 self.expansion * planes,
                                 kernel_size=1,
                                 central_data=central_data)
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                VDropConv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    central_data=central_data,
                    stride=stride
                )
            )
        if percent_on >= 0.5:
            self.nonlinearity1 = F.relu
            self.nonlinearity2 = F.relu
            self.nonlinearity3 = F.relu
        else:
            self.nonlinearity1 = KWinners2d(in_planes, percent_on=percent_on)
            self.nonlinearity2 = KWinners2d(planes, percent_on=percent_on)
            self.nonlinearity3 = KWinners2d(planes, percent_on=percent_on)


    def forward(self, x):
        out = self.nonlinearity1(x)
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.nonlinearity1(out)
        out = self.conv2(out)
        out = self.nonlinearity3(out)
        out = self.conv3(out)
        out += shortcut
        return out

class ResNetEncoder(nn.Module):
    """
    The main subcomponent of FullVisionModel. This encoder also implements both
    .forward() and .encode() to support different outputs for unsupervised and
    supervised training.
    """

    def __init__(
        self,
        block,
        num_blocks,
        filters,
        encoder_num,
        negative_samples=16,
        k_predictions=5,
        patch_size=16,
        input_dims=3,
        previous_input_dim=64,
        first_stride = 1
    ):
        super(ResNetEncoder, self).__init__()
        self.encoder_num = encoder_num

        self.overlap = 2

        self.patch_size = patch_size
        self.filters = filters

        self.model = nn.Sequential()
        self.in_planes = previous_input_dim
        self.first_stride = first_stride

        for idx in range(len(num_blocks)):
            self.model.add_module(
                "layer {}".format((idx)),
                self._make_layer(
                    block, self.filters[idx], num_blocks[idx], stride=self.first_stride
                ),
            )
            self.first_stride = 2

        self.bilinear_model = BilinearInfo(
            in_channels=self.in_planes,
            out_channels=self.in_planes,
            negative_samples=negative_samples,
            k_predictions=k_predictions,
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def encode(self, x, n_patches_x, n_patches_y):
        z = self.model(x)
        out = F.adaptive_avg_pool2d(z, 1)
        out = out.reshape(-1, n_patches_x, n_patches_y, out.shape[1])
        out = out.permute(0, 3, 1, 2).contiguous()
        return z, out

    def forward(self, x, n_patches_x, n_patches_y):
        z, out = self.encode(x, n_patches_x, n_patches_y)
        log_f_list, true_f_list = self.bilinear_model(out, out)
        return log_f_list, true_f_list, z


class SparseResNetEncoder(ResNetEncoder):
    """
    A sparse version of the above ResNetEncoder.
    """

    def __init__(
        self,
        block,
        num_blocks,
        filters,
        encoder_num,
        negative_samples=16,
        k_predictions=5,
        patch_size=16,
        input_dims=3,
        sparse_weights_class=SparseWeights2d,
        sparsity=0.5,
        percent_on=0.5,
        previous_input_dim=64,
        first_stride=1,
    ):
        super(SparseResNetEncoder, self).__init__(
            block,
            num_blocks,
            filters,
            encoder_num,
            negative_samples,
            k_predictions=k_predictions,
            patch_size=patch_size,
            input_dims=input_dims,
            previous_input_dim=previous_input_dim,
            first_stride=first_stride,
        )

        self.model = nn.Sequential()
        self.in_planes = previous_input_dim
        self.first_stride = first_stride
        for idx in range(len(num_blocks)):
            self.model.add_module(
                "sparse_layer {}".format((idx)),
                self._make_layer_sparse(
                    block,
                    self.filters[idx],
                    num_blocks[idx],
                    stride=self.first_stride,
                    sparse_weights_class=sparse_weights_class,
                    sparsity=sparsity,
                    percent_on=percent_on,
                ),
            )
            self.first_stride = 2

        self.bilinear_model = SparseBilinearInfo(
            in_channels=self.in_planes,
            out_channels=self.in_planes,
            negative_samples=negative_samples,
            k_predictions=k_predictions,
            sparse_weights_class=sparse_weights_class,
            sparsity=sparsity,
        )

    def _make_layer_sparse(self,
                           block,
                           planes,
                           num_blocks,
                           stride=1,
                           sparse_weights_class=SparseWeights2d,
                           sparsity=0.5,
                           percent_on=0.2,
    ):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    stride=stride,
                    sparse_weights_class=sparse_weights_class,
                    sparsity=sparsity,
                    percent_on=percent_on,
                )
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


class VDropSparseResNetEncoder(nn.Module):
    """
    A sparse version of the above ResNetEncoder.
    """

    def __init__(
        self,
        block,
        num_blocks,
        filters,
        encoder_num,
        negative_samples=16,
        k_predictions=5,
        patch_size=16,
        input_dims=3,
        percent_on=0.5,
        previous_input_dim=64,
        first_stride=1,
        central_data=None,
    ):
        super(VDropSparseResNetEncoder, self).__init__()

        self.encoder_num = encoder_num

        self.overlap = 2

        self.patch_size = patch_size
        self.filters = filters

        self.model = nn.Sequential()
        self.in_planes = previous_input_dim
        self.first_stride = first_stride
        for idx in range(len(num_blocks)):
            self.model.add_module(
                "sparse_layer {}".format((idx)),
                self._make_layer_vdrop_sparse(
                    block,
                    self.filters[idx],
                    num_blocks[idx],
                    stride=self.first_stride,
                    percent_on=percent_on,
                    central_data=central_data
                ),
            )
            self.first_stride = 2

        self.bilinear_model = VDropSparseBilinearInfo(
            in_channels=self.in_planes,
            out_channels=self.in_planes,
            negative_samples=negative_samples,
            k_predictions=k_predictions,
            central_data=central_data,
        )

    def _make_layer_vdrop_sparse(self,
                           block,
                           planes,
                           num_blocks,
                           stride=1,
                           percent_on=0.2,
                           central_data=None,
    ):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    central_data=central_data,
                    stride=stride,
                    percent_on=percent_on,
                )
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def encode(self, x, n_patches_x, n_patches_y):
        z = self.model(x)
        out = F.adaptive_avg_pool2d(z, 1)
        out = out.reshape(-1, n_patches_x, n_patches_y, out.shape[1])
        out = out.permute(0, 3, 1, 2).contiguous()
        return z, out

    def forward(self, x, n_patches_x, n_patches_y):
        z, out = self.encode(x, n_patches_x, n_patches_y)
        log_f_list, true_f_list = self.bilinear_model(out, out)
        return log_f_list, true_f_list, z