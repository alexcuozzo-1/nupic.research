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

import torch
import torch.nn.functional as F


def multiple_cross_entropy(data_lists, targets, reduction="mean"):
    log_f_module_list, true_f_module_list = data_lists
    device = log_f_module_list[0][0].device
    total_loss = torch.tensor(0.0, requires_grad=True, device=device)
    # Sum losses from each module
    for log_f_list, true_f_list in zip(log_f_module_list, true_f_module_list):
        # Sum losses for each k prediction
        for log_fk, true_fk in zip(log_f_list, true_f_list):
            total_loss = total_loss + F.cross_entropy(
                log_fk, true_fk, reduction=reduction
            )
    return total_loss


def multiple_log_softmax_nll_loss(data_lists, targets, reduction="mean"):
    log_f_module_list, true_f_module_list = data_lists
    device = log_f_module_list[0][0].device
    total_loss = torch.tensor(0.0, requires_grad=True, device=device)
    # Sum losses from each module
    for log_f_list, true_f_list in zip(log_f_module_list, true_f_module_list):
        # Sum losses for each k prediction
        for log_fk, true_fk in zip(log_f_list, true_f_list):
            softmax_fk = torch.softmax(log_fk, dim=1)
            log_softmax_fk = torch.log(softmax_fk + 1e-11)
            total_loss = total_loss + F.nll_loss(
                log_softmax_fk, true_fk, reduction=reduction
            )
    return total_loss


def true_gim_loss(data_lists, targets, reduction="mean"):
    log_f_module_list, true_f_module_list = data_lists
    device = log_f_module_list[0][0].device
    total_loss = torch.tensor(0.0, requires_grad=True, device=device)
    # Sum losses from each module
    for log_f_list, true_f_list in zip(log_f_module_list, true_f_module_list):
        # Sum losses for each k prediction
        for log_fk, _ in zip(log_f_list, true_f_list):
            numerator = log_fk[:, 0, :, :]
            denominator = torch.logsumexp(log_fk[:, 1:, :, :], dim=1).mean()
            total_loss = total_loss + (numerator - denominator).mean()
    return total_loss


def module_specific_cross_entropy(data_lists, targets, reduction="mean", module=-1):
    log_f_module_list, true_f_module_list = data_lists
    device = data_lists[0][0].device
    total_loss = torch.tensor(0.0, requires_grad=True, device=device)
    # Sum losses from each module
    log_f_list, true_f_list = log_f_module_list[module], true_f_module_list[module]
    for log_fk, true_fk in zip(log_f_list, true_f_list):
        total_loss = total_loss + F.cross_entropy(log_fk, true_fk, reduction=reduction)
    return total_loss