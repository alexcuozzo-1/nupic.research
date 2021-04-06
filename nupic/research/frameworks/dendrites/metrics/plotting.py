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

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from .metrics import (
    contexts_by_class,
    dendrite_activations_by_unit,
    dendrite_duty_cycle,
    dendrite_overlap,
    dendrite_overlap_matrix,
    entropy,
    hidden_activations_by_unit,
    mean_selected_activations,
    percent_active_dendrites,
    representation_overlap_matrix,
    representation_overlap_values,
    winning_segment_indices,
)


def plot_dendrite_activations(dendrite_activations, winning_mask, mask_values=None,
                              unit_to_plot=0):
    """
    Returns a heatmap of dendrite activations for a single unit, plotted using
    matplotlib.

    :param dendrite_activations: 3D torch tensor with shape (batch_size, num_units,
                                 num_segments) in which entry b, i, j gives the
                                 activation of the ith unit's jth dendrite segment for
                                 example b
    :param winning_mask: 3D torch tensor with shape (batch_size, num_units,
                         num_segments) in which entry b, i, j is 1 iff the ith unit's
                         jth dendrite segment won for example b, 0 otherwise
    :param mask_values: list of the routing function's mask values for output unit
                        `unit_to_plot`; unused if None
    :param unit_to_plot: index of the unit for which to plot dendrite activations;
                         plots activations of unit 0 by default
    """
    with torch.no_grad():

        num_examples, num_units, num_segments = dendrite_activations.size()

        x_labels = ["example {}".format(j) for j in range(num_examples)]
        if mask_values is not None:
            assert len(mask_values) == num_examples
            x_labels = ["{} [{}]".format(label, mask_values[j])
                        for j, label in enumerate(x_labels)]
        y_labels = ["segment {}".format(j) for j in range(num_segments)]

        # Find the range of activation values to anchor the colorbar
        vmax = dendrite_activations[:, unit_to_plot, :].abs().max().item()
        vmin = -1.0 * vmax

        # Use matplotlib to plot the activation heatmap
        plt.cla()
        fig, ax = plt.subplots()
        ax.imshow(dendrite_activations[:, unit_to_plot, :].T.detach().cpu().numpy(),
                  cmap="coolwarm_r", vmin=vmin, vmax=vmax)

        ax.set_xticks(np.arange(num_examples))
        ax.set_yticks(np.arange(num_segments))

        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.tight_layout()

        # Annotate the winning activation for each example
        winning_segments = torch.argmax(winning_mask[:, unit_to_plot, :], dim=1)
        for n, j in enumerate(winning_segments):
            val = round(dendrite_activations[n, unit_to_plot, j].item(), 2)
            ax.text(n, j, val, ha="center", va="center", color="w")

        figure = plt.gcf()
        return figure


def plot_percent_active_dendrites(winning_mask, targets, category_names=None,
                                  unit_to_plot=0, annotate=True):
    """
    Returns a heatmap with shape (number of segments, number of categories) where cell
    j, c in the heatmap gives the percentage of inputs in category c for which segment
    j is active (for a single unit).

    :param winning_mask: 3D torch tensor with shape (batch_size, num_units,
                         num_segments) in which entry b, i, j is 1 iff the ith unit's
                         jth dendrite segment won for example b, 0 otherwise
    :param targets: 1D torch tensor with shape (batch_size,) where entry b gives the
                    target label for example b
    :param category_names: list of category names to label each column of the heatmap;
                           unused if None or `annotate` is False
    :param unit_to_plot: index of the unit for which to plot percent active dendrites;
                         plots unit 0 by default
    :param annotate: boolean value indicating whether to annotate all items along the x
                     and y axes, as well as individual cell values
    """
    _, _, num_segments = winning_mask.size()
    num_categories = 1 + targets.max().item()

    percent_active = percent_active_dendrites(winning_mask, targets)
    percent_active = percent_active[unit_to_plot, :, :]
    percent_active = percent_active.detach().cpu().numpy()

    # Find the maximum percentage activation value to anchor the colorbar, and use
    # matplotlib to plot the heatmap
    vmax = np.max(percent_active)

    plt.cla()
    fig, ax = plt.subplots()
    ax.imshow(percent_active, cmap="copper", vmin=0.0, vmax=vmax)

    if annotate:

        x_labels = ["category {}".format(j) for j in range(num_categories)]
        if category_names is not None:
            x_labels = category_names
        y_labels = ["segment {}".format(j) for j in range(num_segments)]

        ax.set_xticks(np.arange(num_categories))
        ax.set_yticks(np.arange(num_segments))

        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Annotate all percentage activations
        for i in range(percent_active.shape[0]):
            for j in range(percent_active.shape[1]):
                val = np.round(percent_active[i, j], 2)
                ax.text(j, i, val, ha="center", va="center", color="w")

    else:
        ax.set_xlabel("category")
        ax.set_ylabel("segment")

    plt.tight_layout()
    figure = plt.gcf()
    return figure


def plot_mean_selected_activations(dendrite_activations, winning_mask, targets,
                                   category_names=None, unit_to_plot=0, annotate=True):
    """
    Returns a heatmap with shape (number of segments, number of categories) where cell
    j, c in the heatmap gives the mean activation of the segment j over all instances
    of category c for which segment j became active. As there are multiple dendrite
    segments, the heatmap is created just for the specified unit.

    :param dendrite_activations: 3D torch tensor with shape (batch_size, num_units,
                                 num_segments) in which entry b, i, j gives the
                                 activation of the ith unit's jth dendrite segment for
                                 example b
    :param winning_mask: 3D torch tensor with shape (batch_size, num_units,
                         num_segments) in which entry b, i, j is 1 iff the ith unit's
                         jth dendrite segment won for example b, 0 otherwise
    :param targets: 1D torch tensor with shape (batch_size,) where entry b gives the
                    target label for example b
    :param category_names: list of category names to label each column of the heatmap;
                           unused if None or `annotate` is False
    :param unit_to_plot: index of the unit for which to plot mean selected activations;
                         plots unit 0 by default
    :param annotate: boolean value indicating whether to annotate all items along the x
                     and y axes, as well as individual cell values
    """
    _, num_units, num_segments = dendrite_activations.size()
    num_categories = 1 + targets.max().item()

    assert 0 <= unit_to_plot < num_units

    msa = mean_selected_activations(dendrite_activations, winning_mask, targets)
    msa = msa[unit_to_plot, :, :]
    msa = msa.detach().cpu().numpy()

    # Find the largest absolute mean selected activation value to anchor the colorbar,
    # and use matplotlib to plot the heatmap
    vmax = np.nanmax(np.abs(msa))
    vmin = -vmax

    plt.cla()
    fig, ax = plt.subplots()
    ax.imshow(msa, cmap="coolwarm_r", vmin=vmin, vmax=vmax)

    if annotate:

        x_labels = ["category {}".format(j) for j in range(num_categories)]
        if category_names is not None:
            x_labels = category_names
        y_labels = ["segment {}".format(j) for j in range(num_segments)]

        ax.set_xticks(np.arange(num_categories))
        ax.set_yticks(np.arange(num_segments))

        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Annotate all mean selected activations
        for i in range(msa.shape[0]):
            for j in range(msa.shape[1]):
                val = round(msa[i, j], 2)
                ax.text(j, i, val, ha="center", va="center", color="w")

    else:
        ax.set_xlabel("category")
        ax.set_ylabel("segment")

    plt.tight_layout()
    figure = plt.gcf()
    return figure


def plot_dendrite_activations_by_unit(dendrite_activations, winning_mask, targets,
                                      category_names=None, num_units_to_plot=128):
    """
    Returns a heatmap with shape (num_categories, num_units) where cell c, i gives the
    mean value (post-sigmoid) of the selected dendrite activation for unit i over all
    given examples from category c.

    :param dendrite_activations: 3D torch tensor with shape (batch_size, num_units,
                                 num_segments) in which entry b, i, j gives the
                                 activation of the ith unit's jth dendrite segment for
                                 example b
    :param winning_mask: 3D torch tensor with shape (batch_size, num_units,
                         num_segments) in which entry b, i, j is 1 iff the ith unit's
                         jth dendrite segment won for example b, 0 otherwise
    :param targets: 1D torch tensor with shape (batch_size,) where entry b gives the
                    target label for example b
    :param category_names: list of category names to label each column of the heatmap;
                           unused if None
    :param num_units_to_plot: an integer which gives how many columns to show, for ease
                              of visualization; only the first num_units_to_plot units
                              are shown
    """
    activations = dendrite_activations_by_unit(dendrite_activations, winning_mask,
                                               targets)
    if num_units_to_plot is not None:
        activations = activations[:, :num_units_to_plot]
    activations = activations.detach().cpu().numpy()

    plt.cla()
    fig, ax = plt.subplots()
    ax.imshow(activations, cmap="coolwarm_r", vmin=0.0, vmax=1.0)

    ax.set_xlabel("hidden unit")
    ax.set_ylabel("category")

    plt.tight_layout()
    figure = plt.gcf()
    return figure


def plot_hidden_activations_by_unit(activations, targets, category_names=None,
                                    num_units_to_plot=128):
    """
    Returns a heatmap with shape (num_categories, num_units) where cell c, i gives the
    mean value of hidden activations for unit i over all given examples from category
    c.

    :param activations: 2D torch tensor with shape (batch_size, num_units) where entry
                        b, i gives the activation of unit i for example b
    :param targets: 1D torch tensor with shape (batch_size,) where entry b gives the
                    target label for example b
    :param category_names: list of category names to label each column of the heatmap;
                           unused if None
    :param num_units_to_plot: an integer which gives how many columns to show, for ease
                              of visualization; only the first num_units_to_plot units
                              are shown
    """
    activations = hidden_activations_by_unit(activations, targets)

    if num_units_to_plot is not None:
        activations = activations[:, :num_units_to_plot]
    activations = activations.detach().cpu().numpy()

    plt.cla()
    fig, ax = plt.subplots()
    max_val = np.abs(activations).max()
    ax.imshow(activations, cmap="PiYG", vmin=-max_val, vmax=max_val)

    ax.set_xlabel("hidden unit")
    ax.set_ylabel("category")

    plt.tight_layout()
    figure = plt.gcf()
    return figure


def plot_contexts_by_class(contexts, targets, dims_to_plot=64):
    """
    Returns a heatmap with shape (num_categories, dim_context) where cell c, k gives
    the fraction of instances of category c for which feature/dimension k of the
    context signal generated for an input of said class was non-zero. All values are in
    the range [0, 1].

    :param contexts: 2D torch tensor with shape (batch_size, dim_context) where row b
                     gives the context signal for the bth input
    :param targets: 1D torch tensor with shape (batch_size,) where entry b gives the
                    target label for example b
    :param dims_to_plot: an integer which gives how many columns/features to show, for
                         ease of visualization; only the first dims_to_plot units are
                         shown
    """
    contexts = contexts_by_class(contexts, targets)

    if dims_to_plot is not None:
        contexts = contexts[:, :dims_to_plot]
    contexts = contexts.detach().cpu().numpy()

    plt.cla()
    fig, ax = plt.subplots()
    ax.imshow(contexts, cmap="binary", vmin=0)

    ax.set_xlabel("context feature")
    ax.set_ylabel("category")

    plt.tight_layout()
    figure = plt.gcf()
    return figure


def plot_dendrite_overlap_matrix(winning_mask, targets, category_names=None,
                                 unit_to_plot=0, annotate=True):
    """
    Returns a heatmap with shape (number of categories, number of categories) where
    cell c, k gives the overlap in dendrite activations between categories c and k for
    the dendrite segments of the specified unit. The value in each cell can be
    interpreted as a similarity measure in dendrite activations between categories c
    and k; if the exact same segments are active for the same fraction of instances
    across both categories, the dendrite overlap is 1; if any segment that is active
    for category c and inactive for category k (and vice-versa), the dendrite overlap
    is 0. The resulting heatmap is symmetric.

    :param winning_mask: 3D torch tensor with shape (batch_size, num_units,
                         num_segments) in which entry b, i, j is 1 iff the ith unit's
                         jth dendrite segment won for example b, 0 otherwise
    :param targets: 1D torch tensor with shape (batch_size,) where entry b gives the
                    target label for example b
    :param category_names: list of category names to label each column of the heatmap;
                           unused if None or `annotate` is False
    :param unit_to_plot: index of the unit for which to plot the overlap matrix; plots
                         unit 0 by default
    :param annotate: boolean value indicating whether to annotate all items along the x
                     and y axes, as well as individual cell values
    """
    num_categories = 1 + targets.max().item()

    overlap_matrix = dendrite_overlap_matrix(winning_mask, targets)
    overlap_matrix = overlap_matrix[unit_to_plot, :, :]
    overlap_matrix = overlap_matrix.detach().cpu().numpy()

    # `overlap_matrix` is symmetric, hence we can set all values above the main
    # diagonal to np.NaN so they don't appear in the visualization
    for i in range(num_categories):
        for j in range(i + 1, num_categories):
            overlap_matrix[i, j] = np.nan

    # Anchor the colorbar to the range [0, 1]
    plt.cla()
    fig, ax = plt.subplots()
    ax.imshow(overlap_matrix, cmap="OrRd", vmin=0.0, vmax=1.0)

    if annotate:

        labels = ["category {}".format(j) for j in range(num_categories)]
        if category_names is not None:
            labels = category_names

        ax.set_xticks(np.arange(num_categories))
        ax.set_yticks(np.arange(num_categories))

        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Annotate all overlap values
        for i in range(num_categories):
            for j in range(i + 1):
                val = np.round(overlap_matrix[i, j].item(), 2)
                ax.text(j, i, val, ha="center", va="center", color="w")

    else:
        ax.set_xlabel("category")
        ax.set_ylabel("category")

    plt.tight_layout()
    figure = plt.gcf()
    return figure


def plot_overlap_scores_distribution(winning_mask, targets):
    """
    Returns a histogram which gives the distribution of dendrite overlap scores for all
    units over an input batch. Each data point in the histogram is the overlap score
    corresponding to the dendrite segments of a single unit. See `dendrite_overlap` for
    more details.

    :param winning_mask: 3D torch tensor with shape (batch_size, num_units,
                         num_segments) in which entry b, i, j is 1 iff the ith unit's
                         jth dendrite segment won for example b, 0 otherwise
    :param targets: 1D torch tensor with shape (batch_size,) where entry b gives the
                    target label for example b
    """
    overlap_scores = dendrite_overlap(winning_mask, targets)

    plt.cla()
    plt.hist(x=overlap_scores.tolist(), bins=np.arange(0.0, 1.0, 0.05), color="m",
             edgecolor="k")

    plt.xticks(np.arange(0.0, 1.0, 0.1))
    plt.xlabel("Overlap score")
    plt.ylabel("Segment frequency")
    plt.xlim(0.0, 1.0)
    plt.grid(True)
    plt.tight_layout()

    figure = plt.gcf()
    return figure


def plot_entropy_distribution(winning_mask, targets):
    """
    Returns a histogram which gives the distribution of entropy values of dendrite
    segments over an input batch. Each data point in the histogram is the observed
    entropy of a set of dendrite segments corresponding to a single unit. The entropy
    is the computed using the empirical distribution of the fraction of instances for
    which each segment became active.

    :param winning_mask: 3D torch tensor with shape (batch_size, num_units,
                         num_segments) in which entry b, i, j is 1 iff the ith unit's
                         jth dendrite segment won for example b, 0 otherwise
    """
    _, num_units, _ = winning_mask.size()

    duty_cycle = dendrite_duty_cycle(winning_mask)

    entropies = [entropy(duty_cycle[unit, :]) for unit in range(num_units)]
    max_entropy = entropies[0][1]
    entropies = [ent[0] for ent in entropies]

    plt.cla()
    plt.hist(x=entropies, bins=np.arange(0.0, max_entropy, 0.1), color="g",
             edgecolor="k")

    plt.xticks(np.arange(0.0, 1.0, 0.2))
    plt.xlabel("Entropy  (max entropy: {})".format(round(max_entropy, 2)))
    plt.ylabel("Segment frequency")
    plt.xlim(0.0, max_entropy)
    plt.grid(True)
    plt.tight_layout()

    figure = plt.gcf()
    return figure


def plot_representation_overlap_matrix(activations, targets, category_names=None,
                                       annotate=True):
    """
    Returns a heatmap with shape (num_categories, num_categories) where cell c1, c2
    gives the mean value of pairwise representation overlaps across all pairs of
    examples between classes c1 and c2.  Each individual pairwise representation
    overlap is simply the fraction of hidden units that are active in both examples.

    :param activations: 2D torch tensor with shape (batch_size, num_units) where entry
                        b, i gives the activation of unit i for example b
    :param targets: 1D torch tensor with shape (batch_size,) where entry b gives the
                    target label for example b
    :param category_names: list of category names to label each column of the heatmap;
                           unused if None or `annotate` is False
    :param annotate: boolean value indicating whether to annotate all items along the x
                     and y axes, as well as individual cell values
    """
    num_categories = 1 + targets.max().item()

    overlap_matrix = representation_overlap_matrix(activations, targets)
    overlap_matrix = overlap_matrix.detach().cpu().numpy()

    # `overlap_matrix` is symmetric, hence we can set all values above the main
    # diagonal to np.NaN so they don't appear in the visualization
    for i in range(num_categories):
        for j in range(i + 1, num_categories):
            overlap_matrix[i, j] = np.nan

    # Anchor the colorbar to the range [0, 1]
    plt.cla()
    fig, ax = plt.subplots()
    ax.imshow(overlap_matrix, cmap="YlOrBr", vmin=0.0, vmax=1.0)

    if annotate:

        labels = ["category {}".format(j) for j in range(num_categories)]
        if category_names is not None:
            labels = category_names

        ax.set_xticks(np.arange(num_categories))
        ax.set_yticks(np.arange(num_categories))

        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Annotate all overlap values
        for i in range(num_categories):
            for j in range(i + 1):
                val = np.round(overlap_matrix[i, j].item(), 2)
                ax.text(j, i, val, ha="center", va="center", color="w")

    else:
        ax.set_xlabel("category")
        ax.set_ylabel("category")

    plt.tight_layout()
    figure = plt.gcf()
    return figure


def plot_representation_overlap_distributions(activations, targets):
    """
    Returns a tuple of histograms that show pairwise representation overlaps between
    samples whose representations are given by `activations`. The first histogram
    includes only inter-class pairs, while the second histogram only intra-class pairs;
    self-paired examples are excluded in the latter case.

    :param activations: 2D torch tensor with shape (batch_size, num_units) where entry
                        b, i gives the activation of unit i for example b
    :param targets: 1D torch tensor with shape (batch_size,) where entry b gives the
                    target label for example b
    """
    inter_class_ol, intra_class_ol = representation_overlap_values(activations,
                                                                   targets)
    figures = []

    for fig_num, ol in enumerate((inter_class_ol, intra_class_ol)):

        plt.figure(fig_num)
        plt.hist(x=ol, bins=np.arange(0.0, 1.0, 0.02), color="tab:blue", edgecolor="k")

        plt.xticks(np.arange(0.0, 1.0, 0.2))
        plt.xlabel("Fraction of overlap")
        plt.ylabel("Number of pairs")
        plt.xlim(0.0, 1.0)
        plt.grid(True)
        plt.tight_layout()

        figures.append(plt.gcf())

    return tuple(figures)


def plot_winning_segment_distributions(winning_mask, num_units_to_plot=1, seed=0):
    """
    Plot the distribution of winning segments for the list of units (defaults to just
    the first):

    :param winning_mask: the winning mask of segments;
                         shape num_samples x num_units x num_segments
    :param num_units_to_plot: the number of units to plot
    :param seed: set the random seed for reproducibility.
    """

    # Randomly sample 'num_units_to_plot'.
    assert num_units_to_plot > 0
    num_units = winning_mask.shape[1]
    units = torch.randperm(num_units, generator=get_random_generator(seed))
    units = units[:num_units_to_plot].tolist()

    # Deduce winnings indices.
    winning_indices = winning_segment_indices(winning_mask, units)

    # Generate subplots.
    fig, axs = plt.subplots(1, num_units_to_plot, figsize=(6 * num_units_to_plot, 4))
    if num_units_to_plot == 1:
        axs = [axs]  # ensure this is subscriptable

    # Generate a plot for each unit.
    num_segments = winning_mask.shape[2]
    for i, unit in enumerate(units):
        indices = winning_indices[:, i].cpu().numpy()
        plot_winning_segment_distribution(indices, num_segments, unit=unit, ax=axs[i])

    fig.tight_layout()
    return fig


# ----------------
# Helper functions
# ----------------

def get_random_generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def plot_winning_segment_distribution(winning_indices, num_segments, unit=0, ax=None):
    binrange = (0, num_segments)
    if ax is None:
        _, ax = plt.subplots()
    sns.histplot(
        winning_indices,
        kde=True,
        stat="probability",
        binwidth=1,
        binrange=binrange,
        ax=ax,
    )
    ax.set_xlabel("Segment")
    ax.set_title(f"Probability of Activation of Unit {unit}")
    return ax
