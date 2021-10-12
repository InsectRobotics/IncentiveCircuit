"""
Package that contains all the plotting methods that create the figures of the manuscript.
"""

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2021, Insect Robotics Group," \
                "Institute of Perception, Action and Behaviour," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "GPLv3+"
__version__ = "v1.1-alpha"
__maintainer__ = "Evripidis Gkanias"

from .models_base import MBModel

from typing import List
from matplotlib import cm, colors
from matplotlib.colors import Normalize, ListedColormap

import matplotlib.pyplot as plt
import numpy as np

REWARD_COLOUR = np.array(colors.to_rgb("green"))
PUNISHMENT_COLOUR = np.array(colors.to_rgb("red"))
ODOUR_A_COLOUR = np.array([246, 74, 138]) / 255
ODOUR_B_COLOUR = np.array([255, 165, 0]) / 255
ODOUR_A_CMAP = ListedColormap(np.clip(np.linspace(np.ones(3), ODOUR_A_COLOUR, 100), 0, 1))
ODOUR_B_CMAP = ListedColormap(np.clip(np.linspace(np.ones(3), ODOUR_B_COLOUR, 100), 0, 1))
PRETRAINING_COLOUR = np.array(colors.to_rgb("blue"))
POSTRAINING_COLOUR = np.array(colors.to_rgb("black"))
MAX_NUMBER_OF_EXPERIMENTS = 3


def plot_weights_matrices(ms, nids=None, vmin=-2., vmax=2., only_nids=False, figsize=None):
    """
    Plots the KC-MBON synaptic weights strength as a matrix where the rows are different neurons and the columns are the
    time-steps. The colour reveals the actual strength of the synaptic weights from the KCs to MBONs. The data are split
    in two groups related to odour A and odour B, and the average synaptic strength for each group is plotted.

    Parameters
    ----------
    ms: List[MBModel]
        the models where the values are taken from
    nids: List[int]
        the indices of the neurons that we want to show their names
    vmin: float
        the lower bound for the colour map. Default is -2
    vmin: float
        the upper bound for the colour map. Default is 2
    only_nids: bool
        when True, only the specified neurons are plotted
    figsize: list
        the size of the figure
    """
    title = "weights-" + '-'.join(str(ms[0]).split("'")[1:-1:2])

    nb_models = len(ms)
    xticks = ["%d" % (i+1) for i in range(16)]

    if figsize is None:
        figsize = (7.5, 5)
    plt.figure(title, figsize=figsize)

    for i in range(nb_models):
        nb_timesteps = ms[i].nb_timesteps
        nb_trials = ms[i].nb_trials

        yticks = np.array(ms[i].names)
        if nids is None:
            if ms[i].neuron_ids is None:
                nids = np.arange(ms[i].nb_dan + ms[i].nb_mbon)[::8]
            else:
                nids = ms[i].neuron_ids
        ylim = (len(nids) if only_nids else (ms[i].nb_dan + ms[i].nb_mbon)) - 1

        w = ms[i].w_k2m

        wa_acq = w[1:6*2*nb_timesteps+nb_timesteps, 0]
        wb_acq = w[1:6*2*nb_timesteps+nb_timesteps, 5]
        wa_for = w[1+5*2*nb_timesteps+nb_timesteps:12*2*nb_timesteps, 0]
        wb_for = w[1+6*2*nb_timesteps+nb_timesteps:13*2*nb_timesteps, 5]
        if only_nids:
            wa_acq = wa_acq[:, nids]
            wb_acq = wb_acq[:, nids]
            wa_for = wa_for[:, nids]
            wb_for = wb_for[:, nids]

        if i < 1:
            ax = plt.subplot(2, nb_models + 1, 1)
            plt.imshow(wa_acq.T, vmin=vmin, vmax=vmax, cmap="coolwarm", aspect="auto")
            plt.xticks(2 * nb_timesteps * (np.arange(5) + 1), xticks[:5])
            plt.yticks(np.arange(len(nids)) if only_nids else nids, [r'$%s$' % tick for tick in yticks[nids]])
            ax.yaxis.set_ticks_position('both')
            ax.set_ylabel("Odour A")
            plt.title("acquisition", color=np.array([205, 222, 238]) / 255.)

            ax = plt.subplot(2, nb_models + 1, 2 + nb_models)
            plt.imshow(wb_acq.T, vmin=vmin, vmax=vmax, cmap="coolwarm", aspect="auto")
            plt.plot([np.array([2, 3, 4, 5, 6]) * 2 * nb_timesteps - 1] * 2, [[0] * 5, [ylim] * 5], 'r-')
            plt.xticks(2 * nb_timesteps * (np.arange(5) + 1), xticks[:5])
            plt.yticks(np.arange(len(nids)) if only_nids else nids, [r'$%s$' % tick for tick in yticks[nids]])
            ax.yaxis.set_ticks_position('both')
            ax.set_ylabel("Odour B")
            plt.title("acquisition", color=np.array([255, 197, 200]) / 255.)

        s = (i + 1) / (nb_models + 1)
        ax = plt.subplot(2, nb_models + 1, nb_models + 1 - i)
        plt.imshow(wa_for.T, vmin=vmin, vmax=vmax, cmap="coolwarm", aspect="auto")
        if "reversal" in ms[i].routine_name:
            plt.plot([np.array([2, 3, 4, 5, 6]) * 2 * nb_timesteps - 1] * 2, [[0] * 5, [ylim] * 5], 'r-')
        elif "unpaired" in ms[i].routine_name:
            plt.plot([(np.array([2, 3, 4, 5, 6]) - 1) * 2 * nb_timesteps + 1] * 2, [[0] * 5, [ylim] * 5], 'r-')
        plt.xticks(2 * nb_timesteps * (np.arange(5) + 1), xticks[:5])
        plt.yticks(np.arange(len(nids)) if only_nids else nids, ['' for tick in yticks[nids]])
        ax.yaxis.set_ticks_position('both')
        plt.title(ms[i].routine_name, color=np.array([s * 205, s * 222, 238]) / 255.)

        ax = plt.subplot(2, nb_models + 1, 2 + 2 * nb_models - i)
        plt.imshow(wb_for.T, vmin=vmin, vmax=vmax, cmap="coolwarm", aspect="auto")
        plt.xticks(2 * nb_timesteps * (np.arange(5) + 1), xticks[:5])
        plt.yticks(np.arange(len(nids)) if only_nids else nids, ['' for tick in yticks[nids]])
        ax.yaxis.set_ticks_position('both')
        plt.title(ms[i].routine_name, color=np.array([255, s * 197, s * 200]) / 255.)

    plt.tight_layout()
    plt.show()


def plot_phase_overlap_mean_responses_from_model(ms, nids=None, only_nids=True, figsize=None, show_legend=True):
    """
    Plots the average responses of the neurons per phase/trial with overlapping lines.

    Parameters
    ----------
    ms: list[list[MBModel]]
        the models where the values are taken from
    nids: List[int]
        the indices of the neurons that we want to show their names
    only_nids: bool
        when True, only the specified neurons are plotted.
    figsize: list
        the size of the figure
    """
    title = "individuals-" + '-'.join(str(ms[0][0]).split("'")[1:-1:2])

    nb_odours = 2
    nb_repeats = len(ms)
    nb_models = len(ms[0])

    if nids is None:
        if ms[0][0].neuron_ids is None:
            nids = np.arange(ms[0][0].nb_dan + ms[0][0].nb_mbon)[::8]
        else:
            nids = ms[0][0].neuron_ids
    if only_nids:
        names = np.array(ms[0][0].names)[nids]
    else:
        names = np.array(ms[0][0].names)

    nb_neurons = len(names)
    if only_nids:
        nb_rows = 4
        nb_cols = 6
        if figsize is None:
            figsize = (9, 5)
    else:
        nb_rows = 16
        nb_cols = 7
        if figsize is None:
            figsize = (5, 7)
    plt.figure(title, figsize=figsize)

    exp_names = []
    vajs, vbjs = [], []
    for k in range(nb_repeats):

        vajs.append([])
        vbjs.append([])
        for i in range(nb_models):
            nb_timesteps = ms[k][i].nb_timesteps
            v = ms[k][i]._v

            va = v[1:].reshape((-1, nb_odours, nb_timesteps, v.shape[-1]))[:, 0].reshape((-1, v.shape[-1]))
            vb = v[1:].reshape((-1, nb_odours, nb_timesteps, v.shape[-1]))[:, 1].reshape((-1, v.shape[-1]))

            if only_nids:
                va = va[:, nids]
                vb = vb[:, nids]

            vajs[k].append([])
            vbjs[k].append([])
            for j in range(nb_neurons):
                vajs[k][i].append(va[:, j].reshape((-1, nb_timesteps))[:, 1:].reshape((-1,)))
                vbjs[k][i].append(vb[:, j].reshape((-1, nb_timesteps))[:, 1:].reshape((-1,)))

    va_q25 = np.nanquantile(vajs, .25, axis=0)
    va_q50 = np.nanquantile(vajs, .50, axis=0)
    va_q75 = np.nanquantile(vajs, .75, axis=0)
    vb_q25 = np.nanquantile(vbjs, .25, axis=0)
    vb_q50 = np.nanquantile(vbjs, .50, axis=0)
    vb_q75 = np.nanquantile(vbjs, .75, axis=0)
    for i in range(nb_models):
        exp_names.append(ms[0][i].routine_name)
    _plot_phase_overlap_mean_responses([va_q25, va_q50, va_q75], [vb_q25, vb_q50, vb_q75],
                                       exp_names=exp_names, neuron_names=names,
                                       nb_timesteps=ms[0][0].nb_timesteps, nb_trials=ms[0][0].nb_trials,
                                       title=title, figsize=figsize, show_legend=show_legend and only_nids)


def plot_phase_overlap_mean_responses_from_data(sum_res, only_nids=True, figsize=None, show_legend=True):
    """
    Plots the average responses of the neurons per phase/trial for a specific experiment with overlapping phases.

    Parameters
    ----------
    sum_res: dict
        the dictionary with the responses
    only_nids: bool, optional
        whether to plot only the responses of the specified neurons. Default is True
    figsize: tuple, optional
        the size of the figure
    show_legend: bool, optional
        whether to also plot the legend
    """
    title = "individuals-from-data"
    genotypes = [key for key in sum_res.keys()]

    va, vb = [[[]], [[]], [[]]], [[[]], [[]], [[]]]
    for j, genotype in enumerate(genotypes):

        data_a_q25 = sum_res[genotype]["qa25"]
        data_a_q50 = sum_res[genotype]["qa50"]
        data_a_q75 = sum_res[genotype]["qa75"]
        data_b_q25 = sum_res[genotype]["qb25"]
        data_b_q50 = sum_res[genotype]["qb50"]
        data_b_q75 = sum_res[genotype]["qb75"]

        data_aj_q25 = np.hstack([np.full(1, np.nan, dtype=data_a_q25.dtype), data_a_q25,
                                 np.full(26 - data_a_q25.shape[0] - 1, np.nan, dtype=data_a_q25.dtype)])
        data_aj_q50 = np.hstack([np.full(1, np.nan, dtype=data_a_q50.dtype), data_a_q50,
                                 np.full(26 - data_a_q50.shape[0] - 1, np.nan, dtype=data_a_q50.dtype)])
        data_aj_q75 = np.hstack([np.full(1, np.nan, dtype=data_a_q75.dtype), data_a_q75,
                                 np.full(26 - data_a_q75.shape[0] - 1, np.nan, dtype=data_a_q75.dtype)])

        va[0][0].append(data_aj_q25)
        va[1][0].append(data_aj_q50)
        va[2][0].append(data_aj_q75)

        br = 12
        data_bj_q25 = np.hstack([data_b_q25[:br], np.full(1, data_b_q25[br], dtype=data_a_q25.dtype), data_b_q25[br:],
                                 np.full(26 - data_b_q25.shape[0] - 1, np.nan, dtype=data_b_q25.dtype)])
        data_bj_q50 = np.hstack([data_b_q50[:br], np.full(1, data_b_q50[br], dtype=data_a_q50.dtype), data_b_q50[br:],
                                 np.full(26 - data_b_q50.shape[0] - 1, np.nan, dtype=data_b_q50.dtype)])
        data_bj_q75 = np.hstack([data_b_q75[:br], np.full(1, data_b_q75[br], dtype=data_a_q75.dtype), data_b_q75[br:],
                                 np.full(26 - data_b_q75.shape[0] - 1, np.nan, dtype=data_b_q75.dtype)])
        vb[0][0].append(data_bj_q25)
        vb[1][0].append(data_bj_q50)
        vb[2][0].append(data_bj_q75)

    va = np.array(va)
    vb = np.array(vb)

    _plot_phase_overlap_mean_responses(va, vb, exp_names="reversal", neuron_names=genotypes, is_data=True,
                                       nb_timesteps=3, nb_trials=26,
                                       title=title, show_legend=show_legend and only_nids, figsize=figsize)


def _plot_phase_overlap_mean_responses(va, vb, neuron_names, exp_names=None, nb_timesteps=1, nb_trials=1,
                                       title=None, figsize=None, show_legend=True, is_data=False):

    if exp_names is None:
        exp_names = ["reversal"]
    elif not isinstance(exp_names, list):
        exp_names = [exp_names]

    if title is None:
        title = "phase-overlap-responses"

    plot_shock = ["d_{av}", "c_{av}", "f_{av}", "320a", "296a", "301a"]

    ymin, ymax = 0, 2
    ylim = [ymin - .1, ymax + .1]
    xticks = ["%d" % i for i in range(16)]

    nb_models = len(exp_names)
    nb_neurons = len(neuron_names)
    nb_plots = nb_neurons * 2

    nb_rows = 4
    nb_cols = nb_plots // nb_rows
    while nb_cols > 7:
        nb_rows += 4
        nb_cols = nb_plots // nb_rows + 1

    if figsize is None:
        figsize = (8 - 2 * int(nb_neurons != 12), nb_rows + 1)

    plt.figure(title, figsize=figsize)
    subs = []
    for i in range(len(exp_names)-1, -1, -1):
        x_ticks_ = xticks[1:(nb_trials // 2) // 2] * 2
        n = len(x_ticks_)
        _x_ticks = np.arange(n, dtype=float) + 2 - 1 / (nb_timesteps - 1)
        _x_ticks[n//2:] += 1. - 1 / (nb_timesteps - 1)

        x_ = np.arange(0, nb_trials // 2, 1 / (nb_timesteps - 1)) - 1 / (nb_timesteps - 1)

        for j in range(nb_neurons):
            label = None
            a_col = ODOUR_A_CMAP(1 - (i + 1) / (MAX_NUMBER_OF_EXPERIMENTS + 1))
            if j == nb_neurons - 1:
                label = exp_names[i]

            vaj_q25, vaj_q50, vaj_q75 = [v[i, j] for v in va]
            if len(subs) <= j:
                axa = plt.subplot(nb_rows, nb_cols, 2 * (j // nb_cols) * nb_cols + j % nb_cols + 1)
                axa.set_xticks(_x_ticks)
                axa.set_xticklabels(["" for _ in x_ticks_])
                axa.set_yticks([0, ymax/2, ymax])
                axa.set_ylim(ylim)
                axa.set_xlim([0, n + 1 + 1 / (nb_timesteps - 1)])
                axa.tick_params(labelsize=8)
                axa.set_title(neuron_names[j].replace("_", "\n") if is_data else r"$%s$" % neuron_names[j], fontsize=8)
                if j % nb_cols == 0:
                    axa.set_ylabel("Odour A", fontsize=8)
                else:
                    axa.set_yticklabels([""] * 3)
                    axa.spines['left'].set_visible(False)
                    axa.set_yticks([])
                axa.spines['top'].set_visible(False)
                axa.spines['right'].set_visible(False)

                a_acol = ODOUR_A_CMAP(.2)

                if nb_models == 1:
                    axa.fill_between(x_[3:13], vaj_q25[3:13], vaj_q75[3:13], facecolor=a_acol, alpha=.2)
                axa.plot(x_[:4], vaj_q50[:4], color=(.8, .8, .8), lw=2)
                axa.plot(x_[3:13], vaj_q50[3:13], color=a_acol, lw=2, label="acquisition")
                subs.append(axa)

            if nb_models == 1:
                subs[j].fill_between(x_[14+int(is_data):24+int(is_data)], vaj_q25[14+int(is_data):24+int(is_data)],
                                     vaj_q75[14+int(is_data):24+int(is_data)], facecolor=a_col, alpha=.2)
            subs[j].plot(x_[12:15+int(is_data)], vaj_q50[12:15+int(is_data)], color=(.8, .8, .8), lw=2)
            subs[j].plot(x_[23+int(is_data):], vaj_q50[23+int(is_data):], color=(.8, .8, .8), lw=2)
            subs[j].plot(x_[14+int(is_data):24+int(is_data)], vaj_q50[14+int(is_data):24+int(is_data)],
                         color=a_col, lw=2, label=label)
            if ("extinction" in exp_names[i] or "unpaired" in exp_names[i] or
                    np.all([ps not in neuron_names[j] for ps in plot_shock])):
                continue
            shock_i = [15+int(is_data), 17+int(is_data), 19+int(is_data), 21+int(is_data), 23+int(is_data)]
            subs[j].plot(x_[shock_i], vaj_q50[shock_i], color=PUNISHMENT_COLOUR, marker='.', linestyle=' ')

        x_b = x_ + 1 - 1 / (nb_timesteps - 1)
        for j in range(nb_neurons):
            jn = j + nb_neurons

            label = None
            b_col = ODOUR_B_CMAP(1 - (i + 1) / (MAX_NUMBER_OF_EXPERIMENTS + 1))
            if j == nb_neurons - 1:
                label = exp_names[i]

            vbj_q25, vbj_q50, vbj_q75 = [v[i, j] for v in vb]
            if len(subs) <= jn:
                axb = plt.subplot(nb_rows, nb_cols, (2 * (j // nb_cols) + 1) * nb_cols + j % nb_cols + 1)
                axb.set_xticks(_x_ticks)
                axb.set_xticklabels(x_ticks_)
                axb.set_yticks([0, ymax/2, ymax])
                axb.set_ylim(ylim)
                axb.set_xlim([0, n + 1 + 1 / (nb_timesteps - 1)])
                axb.tick_params(labelsize=8)
                if j % nb_cols == 0:
                    axb.set_ylabel("Odour B", fontsize=8)
                    if nb_rows <= 4:
                        axb.text(-3, -.8, "Trial #", fontsize=8)
                else:
                    axb.set_yticklabels([""] * 3)
                    axb.spines['left'].set_visible(False)
                    axb.set_yticks([])

                axb.spines['top'].set_visible(False)
                axb.spines['right'].set_visible(False)

                b_acol = ODOUR_B_CMAP(0.2)

                if nb_models == 1:
                    axb.fill_between(x_b[2:12], vbj_q25[2:12], vbj_q75[2:12], facecolor=b_acol, alpha=.2)
                axb.plot(x_b[:3], vbj_q50[:3], color=(.8, .8, .8))
                axb.plot(x_b[2:12], vbj_q50[2:12], color=b_acol, lw=2, label="acquisition")
                subs.append(axb)

            if nb_models == 1:
                subs[jn].fill_between(x_b[13:23], vbj_q25[13:23], vbj_q75[13:23], facecolor=b_col, alpha=.2)
            subs[jn].plot(x_b[11:14], vbj_q50[11:14], color=(.8, .8, .8), lw=2)
            subs[jn].plot(x_b[22:], vbj_q50[22:], color=(.8, .8, .8), lw=2)
            subs[jn].plot(x_b[13:23], vbj_q50[13:23], color=b_col, lw=2, label=label)

            if i > 0 or np.all([ps not in neuron_names[j] for ps in plot_shock]):
                continue
            shock_i = [3, 5, 7, 9, 11]
            subs[jn].plot(x_b[shock_i], vbj_q50[shock_i], color=PUNISHMENT_COLOUR, marker='.', linestyle=' ')

    if show_legend:
        subs[len(subs)//2 - 1].legend(fontsize=8, bbox_to_anchor=(1.05, 1.35), loc='upper left',
                                      framealpha=0., labelspacing=1.)
        subs[-1].legend(fontsize=8, bbox_to_anchor=(1.05, 1.35), loc='upper left', framealpha=0., labelspacing=1)
    plt.tight_layout()
    plt.show()


def plot_weights(ms, nids=None, only_nids=True, figsize=None):
    """
    Plots the average synaptic weights of the post-synaptic neurons per phase/trial for a specific experiment with
    overlapping phases. The average weight is calculated per odour.

    Parameters
    ----------
    ms: List[MBModel]
        the models where the values are taken from
    nids: List[int]
        the indices of the neurons that we want to show their names
    only_nids: bool
        when True, only the specified neurons are plotted.
    figsize: list
        the size of the figure
    """
    title = "weights-" + '-'.join(str(ms[0]).split("'")[1:-1:2])

    nb_odours = 2
    nb_models = len(ms)
    xticks = ["%d" % i for i in range(16)]
    ylim = [-0.1, 2.1]

    if nids is None:
        if ms[0].neuron_ids is None:
            nids = np.arange(ms[0].nb_dan + ms[0].nb_mbon)[::8]
        else:
            nids = ms[0].neuron_ids
    if only_nids:
        names = np.array(ms[0].names)[nids]
    else:
        names = np.array(ms[0].names)

    nb_neurons = len(names)
    if only_nids:
        nb_rows = 4
        nb_cols = 6
        if figsize is None:
            figsize = (9, 5)
    else:
        nb_rows = 16
        nb_cols = 7
        if figsize is None:
            figsize = (5, 7)
    plt.figure(title, figsize=figsize)

    subs = []
    for i in range(nb_models-1, -1, -1):
        nb_timesteps = ms[i].nb_timesteps
        nb_trials = ms[i].nb_trials

        w = ms[i].w_k2m

        # trial, odour, time-step, neuron
        wa = np.nanmean(w[1:, :5], axis=1).reshape((-1, nb_odours, nb_timesteps, w.shape[-1]))[:, 0].reshape((-1, w.shape[-1]))
        if only_nids:
            wa = wa[:, nids]

        x_ticks_ = xticks[1:(nb_trials // 2) // 2] * 2
        n = len(x_ticks_)
        _x_ticks = np.arange(n, dtype=float) + 2 - 1 / (nb_timesteps - 1)
        _x_ticks[n//2:] += 1. - 1 / (nb_timesteps - 1)

        x_ = np.arange(0, nb_trials // 2, 1 / (nb_timesteps - 1)) - 1 / (nb_timesteps - 1)

        for j in range(nb_neurons):

            label = None
            a_col = ODOUR_A_CMAP(1 - (i + 1) / (nb_models + 1))
            if j == nb_neurons - 1:
                label = ms[i].routine_name

            waj = wa[:, j].reshape((-1, nb_timesteps))[:, 1:].reshape((-1,))
            if len(subs) <= j:
                axa = plt.subplot(nb_rows, nb_cols, 2 * (j // nb_cols) * nb_cols + j % nb_cols + 1)

                axa.set_xticks(_x_ticks)
                axa.set_xticklabels(["" for _ in x_ticks_])
                axa.set_yticks([0, 1, 2])
                axa.set_ylim(ylim)
                axa.set_xlim([0, n + 1 + 1 / (nb_timesteps - 1)])
                axa.tick_params(labelsize=8)
                axa.set_title(r"$%s$" % names[j], fontsize=8)
                if j == 0:
                    axa.set_ylabel("Odour A", fontsize=8)
                else:
                    axa.set_yticklabels([""] * 3)
                    axa.spines['left'].set_visible(False)
                    axa.set_yticks([])

                axa.spines['top'].set_visible(False)
                axa.spines['right'].set_visible(False)

                a_acol = ODOUR_A_CMAP(1 / (nb_models + 1))
                axa.plot(x_[:4], waj[:4], color=(.8, .8, .8))
                axa.plot(x_[3:13], waj[3:13], color=a_acol, label="acquisition")
                subs.append(axa)
            subs[j].plot(x_[12:15], waj[12:15], color=(.8, .8, .8))
            subs[j].plot(x_[23:], waj[23:], color=(.8, .8, .8))
            subs[j].plot(x_[14:24], waj[14:24], color=a_col, label=label)

            if ("extinction" in ms[i].routine_name or "unpaired" in ms[i].routine_name or
                    "d" not in names[j] and "c" not in names[j] or "av" not in names[j]):
                continue
            shock_i = [15, 17, 19, 21, 23]
            subs[j].plot(x_[shock_i], waj[shock_i], color=PUNISHMENT_COLOUR, linestyle='.')

        # axb = plt.subplot(nb_models * 2, 2, 2 + i * 4)
        wb = np.nanmean(w[1:, 5:], axis=1).reshape((-1, nb_odours, nb_timesteps, w.shape[-1]))[:, 1].reshape((-1, w.shape[-1]))
        if only_nids:
            wb = wb[:, nids]
        x_b = x_ + 1 - 1 / (nb_timesteps - 1)
        for j in range(nb_neurons):
            jn = j + nb_neurons

            label = None
            b_col = ODOUR_B_CMAP(1 - (i + 1) / (nb_models + 1))
            if j == nb_neurons - 1:
                label = ms[i].routine_name

            wbj = wb[:, j].reshape((-1, nb_timesteps))[:, 1:].reshape((-1,))
            if len(subs) <= jn:
                axb = plt.subplot(nb_rows, nb_cols, (2 * (j // nb_cols) + 1) * nb_cols + j % nb_cols + 1)
                axb.set_xticks(_x_ticks)
                axb.set_xticklabels(x_ticks_)
                axb.set_yticks([0, 1, 2])
                axb.set_ylim(ylim)
                axb.set_xlim([0, n + 1 + 1 / (nb_timesteps - 1)])
                axb.tick_params(labelsize=8)
                if j % nb_cols == 0:
                    axb.set_ylabel("Odour B", fontsize=8)
                    if only_nids:
                        axb.text(-3, -.8, "Trial #", fontsize=8)
                else:
                    axb.set_yticklabels([""] * 3)
                    axb.spines['left'].set_visible(False)
                    axb.set_yticks([])
                # axb.yaxis.grid()
                axb.spines['top'].set_visible(False)
                axb.spines['right'].set_visible(False)

                b_acol = ODOUR_B_CMAP(1 / (nb_models + 1))
                axb.plot(x_b[:3], wbj[:3], color=(.8, .8, .8))
                axb.plot(x_b[2:12], wbj[2:12], color=b_acol, label="acquisition")
                subs.append(axb)

            subs[jn].plot(x_b[11:14], wbj[11:14], color=(.8, .8, .8))
            subs[jn].plot(x_b[22:], wbj[22:], color=(.8, .8, .8))
            subs[jn].plot(x_b[13:23], wbj[13:23], color=b_col, label=label)

            if i > 0 or "d" not in names[j] and "c" not in names[j] or "av" not in names[j]:
                continue
            shock_i = [3, 5, 7, 9, 11]
            subs[jn].plot(x_b[shock_i], wbj[shock_i], color=PUNISHMENT_COLOUR, linestyle='.')

    subs[len(subs)//2 - 1].legend(fontsize=8, bbox_to_anchor=(1.05, 1.35), loc='upper left',
                                  framealpha=0., labelspacing=1.)
    subs[-1].legend(fontsize=8, bbox_to_anchor=(1.05, 1.35), loc='upper left', framealpha=0., labelspacing=1)
    plt.tight_layout()
    plt.show()


def plot_model_structure(m: MBModel, nids=None, vmin=-.5, vmax=.5, only_nids=False, figsize=None):
    """
    Plots the structure of the model for each synaptic weights matrix where input comes from the left and output goes to
    the bottom. The connections are drawn as bonds between the axon of each input neuron (horizontal line) and the
    dendrite of each output neuron (vertical line), where the colour and size of the bond illustrates its strength.

    Parameters
    ----------
    m: MBModel
        the model where the values are taken from
    nids: List[int]
        the indices of the neurons that we want to show their names
    vmin: float
        the lower bound of the colour map
    vmax: float
        the upper bound of the colour map
    only_nids: bool
        when True, only the specified neurons are plotted.
    figsize: list
        the size of the figure
    """
    if figsize is None:
        figsize = (5, 4)
    fig = plt.figure("weight-tables-" + m.__class__.__name__.lower(), figsize=figsize)
    ax = fig.subplot_mosaic(
        """
        bbbb.aaa
        bbbb....
        bbbb....
        bbbb.ccc
        bbbb.ccc
        ........
        ........
        ddddeeee
        ddddeeee
        ddddeeee
        ........
        ........
        ffff.ggg
        ffff....
        ffff....
        ffff.hhh
        """
    )
    if nids is None:
        nids = np.array(m.neuron_ids)

    if only_nids:
        neurs = nids
    else:
        neurs = np.arange(m.w_m2v.shape[1])
    neur_names = np.array([r"$%s$" % name for name in m.names])

    ax['a'].set_title(r"$W_{U2D}$", fontsize=8)
    plot_synapses(w=m.w_u2d[4:5, neurs[nids < 16]].T, vmin=vmin, vmax=vmax,
                  names_in=["shock"], names_out=neur_names[nids[nids < 16]], ax=ax['a'])

    ax['b'].set_title(r"$W_{D2KM}$", fontsize=8)
    plot_synapses(w=m._w_d2k[neurs[nids < 16]][:, neurs[nids >= 16]], vmin=vmin, vmax=vmax,
                  names_in=neur_names[nids[nids < 16]], names_out=neur_names[nids[nids >= 16]], ax=ax['b'])

    ax['c'].set_title(r"$W_{P2K}$", fontsize=8)
    plot_synapses(w=m.w_p2k.T, names_in=["odour A", "odour B"], vmin=vmin, vmax=vmax,
                  names_out=[r"$k_{%d}$" % k for k in np.arange(m.w_p2k.shape[1])], ax=ax['c'])

    ax['d'].set_title(r"$W_{M2D}$", fontsize=8)
    plot_synapses(w=m._w_m2v[neurs[nids >= 16]][:, neurs[nids < 16]], vmin=vmin, vmax=vmax,
                  names_in=neur_names[nids[nids >= 16]], names_out=neur_names[nids[nids < 16]], ax=ax['d'])

    ax['e'].set_title(r"$W_{M2M}$", fontsize=8)
    plot_synapses(w=m._w_m2v[neurs[nids >= 16]][:, neurs[nids >= 16]], vmin=vmin, vmax=vmax,
                  names_in=neur_names[nids[nids >= 16]], names_out=neur_names[nids[nids >= 16]], ax=ax['e'])
    ax['e'].set_yticklabels([""] * len(nids[nids >= 16]))

    ax['f'].set_title(r"$W_{K2M}^T$", fontsize=8)
    plot_synapses(w=m.w_k2m[0, :, neurs[nids >= 16]], names_in=[r"$k_{%d}$" % k for k in np.arange(m.w_p2k.shape[1])],
                  names_out=neur_names[nids[nids >= 16]], vmin=vmin, vmax=vmax, ax=ax['f'])
    # fig.colorbar(im, ax=ax['f'], ticks=[-.5, 0, .5])
    ax['g'].set_title(r"$w_{rest}$", fontsize=8)
    plot_synapses(w=np.array([m.bias[neurs[nids < 16]]]).T, vmin=vmin, vmax=vmax,
                  names_in=[r"$1$"], names_out=neur_names[nids[nids < 16]], ax=ax['g'])

    ax['h'].set_title(r"", fontsize=8)
    plot_synapses(w=np.array([m.bias[neurs[nids >= 16]]]).T, vmin=vmin, vmax=vmax,
                  names_in=[r"$1$"], names_out=neur_names[nids[nids >= 16]], ax=ax['h'])

    plt.tight_layout()
    plt.show()


def plot_synapses(w, names_in, names_out, ax=None, cmap="coolwarm", vmin=-.5, vmax=.5):
    """
    Plots the synaptic weights matrix where input comes from the left and output goes to the bottom. The connections are
    drawn as bonds between the axon of each input neuron (horizontal line) and the dendrite of each output neuron
    (vertical line), where the colour and size of the bond illustrates its strength.

    Parameters
    ----------
    w: np.ndarray
        the weights matrix where the synaptic weights are drawn from
    names_in: List[str]
        the names of the pre-synaptic neurons
    names_out: List[str]
        the names of the post-synaptic neurons
    ax: optional
        a matplotlib axis to draw the plot in
    cmap: str
        the colour map name to use for  specifying the direction and amplitude of synaptic weight
    vmin: float
        the lower bound of the colour map
    vmax: float
        the upper bound of the colour map
    figsize: list
        the size of the figure
    """
    x, y = np.meshgrid(np.arange(w.shape[0]), np.arange(w.shape[1]))

    if ax is None:
        ax = plt.subplot(111)

    mag = np.absolute(w)
    sig = np.sign(w)

    # plot grid of axons and dendrites
    ax.plot(np.vstack([x[:1, :], x, x[:1, :]]), np.hstack([[[-1]] * y.shape[1], y.T, [[y.shape[0]]] * y.shape[1]]).T,
            'k-', lw=.5, alpha=.2)
    ax.plot(np.hstack([[[-1]] * x.shape[0], x, [[x.shape[1]]] * x.shape[0]]).T, np.hstack([y[:, :1], y, y[:, :1]]).T,
            'k-', lw=.5, alpha=.2)

    # plot direction arrows
    ax.scatter(x[0], [-.85] * y.shape[1], marker='v', s=10, c='k', alpha=.2)
    ax.scatter([x.shape[1]-.1] * x.shape[0], y[:, 0], marker='>', s=10, c='k', alpha=.2)

    # plot connection strength
    ax.scatter(x, y, marker='.', s=200 * mag, c=sig * np.log(mag + 1), cmap=cmap, vmin=vmin, vmax=vmax)

    # configure layout
    ax.set_yticks(np.arange(len(names_in)))
    ax.set_xticks(np.arange(len(names_out)))
    ax.set_yticklabels(names_in, fontsize=8)
    ax.set_xticklabels(names_out, fontsize=8)
    ax.tick_params(axis='x', rotation=30)
    ax.tick_params(axis='y', rotation=30)
    ax.set_ylim([-1., len(names_in)])
    ax.set_xlim([-1., len(names_out)])
    for p in ["top", "bottom", "right", "left"]:
        ax.spines[p].set_visible(False)
    ax.yaxis.set_ticks_position("none")
    ax.xaxis.set_ticks_position("none")


def plot_learning_rule(wrt_k=True, wrt_w=True, wrt_d=True, colour_bar=False, fill=True, figsize=None):
    """

    Plots 21 contour sub-plots in a 3-by-7 grid where the relationship among the parameters of the dopaminergic learning
    rule is visualised. In each row contours are created for 7 values of one of the parameters: KC+W(t), D(t) or W(t+1).

    Parameters
    ----------
    wrt_k: bool, optional
        create row of plots with respect to the KC activity. Default is True.
    wrt_w: bool, optional
        create row of plots with respect to the W_k2m synaptic weight. Default is True.
    wrt_d: bool, optional
        create row of plots with respect to the dopaminergic factor (D). Default is True.
    colour_bar: bool, optional
        show colour bar. Default is False.
    fill: bool, optional
        fill the contours. Default is True.
    figsize: list, optional
        the size of the figure
    """

    contour = plt.contour
    if fill:
        contour = plt.contourf
    k_, d_, w_ = np.linspace(0, 1, 101), np.linspace(-1, 1, 101), np.linspace(0, 1, 101)

    nb_rows = int(wrt_k) + int(wrt_w) + int(wrt_d)
    nb_cols = 7 + int(colour_bar)
    if figsize is None:
        figsize = (nb_cols - 1, nb_rows + 1 - .5 * float(colour_bar))
    plt.figure("dlr", figsize=figsize)
    if wrt_k:
        w, d = np.meshgrid(w_, d_)
        for i, k in enumerate(np.linspace(0, 1, 7)):
            y = d * (k + w - 1)
            plt.subplot(nb_rows, nb_cols, i + 1)
            contour(w, d, y, 30, cmap="coolwarm", vmin=-2, vmax=2)
            plt.xticks([0, .5, 1], fontsize=8)
            plt.xlabel("w", fontsize=8)
            if i < 1:
                plt.yticks([-1, 0, 1], fontsize=8)
                plt.ylabel("D", fontsize=8)
            else:
                plt.yticks([-1, 0, 1], [""] * 3, fontsize=8)

            plt.title("k=%.2f" % k, fontsize=8)

    if wrt_w:
        k, d = np.meshgrid(k_, d_)
        for i, w in enumerate(np.linspace(0, 1, 7)):
            y = d * (k + w - 1)
            plt.subplot(nb_rows, nb_cols, int(wrt_k) * nb_cols + i + 1)
            contour(k, d, y, 30, cmap="coolwarm", vmin=-2, vmax=2)
            plt.xticks([0, .5, 1], fontsize=8)
            plt.xlabel("k", fontsize=8)
            if i < 1:
                plt.yticks([-1, 0, 1], fontsize=8)
                plt.ylabel("D", fontsize=8)
            else:
                plt.yticks([-1, 0, 1], [""] * 3, fontsize=8)

            plt.title("w=%.2f" % w, fontsize=8)

    if wrt_d:
        k, w = np.meshgrid(k_, w_)
        for i, d in enumerate(np.linspace(-1, 1, 7)):
            y = d * (k + w - 1)
            plt.subplot(nb_rows, nb_cols, (int(wrt_k) + int(wrt_w)) * nb_cols + i + 1)
            contour(k, w, y, 30, cmap="coolwarm", vmin=-2, vmax=2)
            plt.xticks([0, .5, 1], fontsize=8)
            plt.xlabel("k", fontsize=8)
            if i < 1:
                plt.yticks([0, 0.5, 1], fontsize=8)
                plt.ylabel("w", fontsize=8)
            else:
                plt.yticks([0, 0.5, 1], [""] * 3, fontsize=8)

            plt.title("D=%.2f" % d, fontsize=8)

    if colour_bar:
        cax = plt.axes([0.9, 0.34, 0.01, 0.45])
        plt.colorbar(cm.ScalarMappable(norm=Normalize(vmin=-1, vmax=1), cmap="coolwarm"), cax=cax, ticks=[-1, 0, 1])

    plt.tight_layout()
    plt.show()


def plot_sm(m, nids=None, sub=None):
    """
    The responses of the susceptible memory (SM) sub-circuit.

    Parameters
    ----------
    m: MBModel
        the model to get the responses from
    nids: List[int], optional
        the 2 neuron indices which we want to plot their responses. Default is 1 and 6.
    sub: int, tuple
        the subfigure code
    """
    if nids is None:
        nids = [1, 6]
    _plot_subcircuit(m, nids, nnames=[r"$d_{av}$", r"$s_{at}$"], title="SM",
                     ncolours=["#db006aff", "#6adbb8ff"], uss=["r", None], sub=sub)


def plot_rm(m, nids=None, sub=None):
    """
    The responses of the restrained memory (RM) sub-circuit.

    Parameters
    ----------
    m: MBModel
        the model to get the responses from
    nids: List[int]
        the 2 neuron indices which we want to plot their responses. Default is 6 and 9.
    sub: int, tuple
        the subfigure code
    """
    if nids is None:
        nids = [6, 9]
    _plot_subcircuit(m, nids, nnames=[r"$s_{at}$", r"$r_{av}$"], title="RM",
                     ncolours=["#6adbb8ff", "#db6a6aff"], sub=sub)


def plot_ltm(m, nids=None, sub=None):
    """
    The responses of the long-term memory (LTM) sub-circuit.

    Parameters
    ----------
    m: MBModel
        the model to get the responses from
    nids: List[int]
        the 2 neuron indices which we want to plot their responses. Default is 2 and 10.
    sub: int, tuple
        the subfigure code
    """
    if nids is None:
        nids = [2, 10]
    _plot_subcircuit(m, nids, nnames=[r"$c_{at}$", r"$m_{at}$"], title="LTM",
                     ncolours=["#6adb00ff", "#6adbb8ff"], uss=["g", None], sub=sub)


def plot_rrm(m, nids=None, sub=None):
    """
    The responses of the reciprocal restrained memories (RRM) sub-circuit.

    Parameters
    ----------
    m: MBModel
        the model to get the responses from
    nids: List[int]
        the 4 neuron indices which we want to plot their responses. Default is 3, 8, 2 and 9.
    sub: int, tuple
        the subfigure code
    """
    if nids is None:
        nids = [3, 8, 2, 9]
    _plot_subcircuit(m, nids, title="RRM", uss=["r", None, None, None],
                     nnames=[r"$c_{av}$", r"$r_{at}$", r"$c_{at}$", r"$r_{av}$"],
                     ncolours=["#db006aff", "#6adbb8ff", "#6adb00ff", "#db6a6aff"], sub=sub)


def plot_rfm(m, nids=None, sub=None):
    """
    The responses of the reciprocal forgetting memories (RFM) sub-circuit.

    Parameters
    ----------
    m: MBModel
        the model to get the responses from
    nids: List[int]
        the 4 neuron indices which we want to plot their responses. Default is 5, 10, 4 and 11.
    sub: int, tuple
        the subfigure code
    """
    if nids is None:
        nids = [5, 10, 4, 11]
    _plot_subcircuit(m, nids, title="RFM",
                     uss=["r", None, None, None],
                     nnames=[r"$f_{av}$", r"$m_{at}$", r"$f_{at}$", r"$m_{av}$"],
                     ncolours=["#db006aff", "#6adbb8ff", "#6adb00ff", "#db6a6aff"], sub=sub)


def plot_mam(m, nids=None, sub=None):
    """
    The responses of the memory assimilation mechanism (MAM).

    Parameters
    ----------
    m: MBModel
        the model to get the responses from
    nids: List[int]
        the 4 neuron indices which we want to plot their responses. Default is 4, 8, 2 and 10.
    sub: int, tuple
        the subfigure code
    """
    if nids is None:
        nids = [4, 8, 2, 10]
    _plot_subcircuit(m, nids, title="MAM", uss=[None, None, "g", None],
                     nnames=[r"$f_{at}$", r"$r_{at}$", r"$c_{at}$", r"$m_{at}$"],
                     ncolours=["#6adb00ff", "#6adbb8ff", "#6adb00ff", "#6adbb8ff"], sub=sub)


def _plot_subcircuit(m, nids, nnames, ncolours, uss=None, title="sub-circuit", sub=None):
    """
    Plots pairs of the responses of the specified neurons for the mushroom body model. The IDs of the neurons to plot
    are specified in a list of IDs (nids). Neurons are drawn in pairs from the list and are plotted the one on the top
    of the other in separate figures (if the specified neurons are more than 2).

    Parameters
    ----------
    m: MBModel
        the model where the values are taken from
    nids: List[int]
        the indices of the neurons that we want to plot
    nnames: List[str]
        a list with the names of the neurons
    ncolours: List[str]
        colour for the line representing the neuron with the same index
    uss: List[str], optional
        a list of colour-names for the US of the same size as the nids;'r' for red (punishment), or 'g' for green
        (reward), None for no reinforcement. Default is None.
    title: str, optional
        the title of the figure. If more than 1 plots are to be generated, a number is added at the end of the title.
        Default is 'sub-circuit'
    sub: int, tuple
        the subfigure code
    """

    if uss is None:
        uss = [None] * len(nids)

    nb_cols = len(nids) // 2
    show_figure = sub is None

    for i in range(nb_cols):
        if show_figure and nb_cols > 1:
            plt.figure("%s_%d" % (title, i+1), figsize=(1, 1))
        elif show_figure:
            plt.figure(title, figsize=(1, 1))
            plt.clf()
            sub = 111
        else:
            plt.figure("sub-circuits", figsize=(10, 1.5))

        if not isinstance(sub, list):
            sub = [sub]

        ax = plt.subplot(sub[i])
        for nid, nname, colour, us in zip(
                nids[i*2:(i+1)*2], nnames[i*2:(i+1)*2], ncolours[i*2:(i+1)*2], uss[i*2:(i+1)*2]):
            v = m._v[:, nid]
            plt.plot(v, c=colour, lw=2, label=nname)
            if us is not None:
                plt.plot(np.arange(len(v))[v > 1.5], v[v > 1.5], us + ".")

        ax.plot([10, 10, 80, 80], [-.1, .5, .5, -.1], color=[.7]*3, linestyle=':')
        if np.any(uss):
            c = [1., .7, .7] if 'r' in uss else [.7, 1., .7]
            ax.plot([30, 30, 60, 60], [-.1, 2, 2, -.1], color=c, linestyle=':')
        plt.ylim([-0.1, 2.1])
        plt.yticks([0, 1, 2])
        plt.xlim([1, 100])
        plt.xticks([10, 30, 60, 80],
                   [r"$t_{1}$", r"$t_{2}$", r"$t_{3}$", r"$t_{4}$"])
        plt.tick_params(labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # plt.legend(loc="upper right", fontsize=8)
        if not show_figure:
            ax.set_title(title)

        plt.tight_layout()

    if show_figure:
        plt.show()


def plot_arena_paths(data, cases, names, repeats, code, name="arena-paths", lw=1., alpha=.2, figsize=None):
    """
    Plots the paths in the arena for all the given cases andme names.

    Parameters
    ----------
    data: list[np.ndarray]
        list of the paths for every case
    cases: list[list]
        list of the required cases
    names: list[str]
        list of names of the cases
    repeats: list[int]
        list of repeats that specify the repeat of each case
    code: str
        the combination of MBONs to show
    name: str, optional
        the title of the figure. Default is 'arena-paths'
    lw: float, optional
        the line width of the paths. Default is 0.1
    alpha: float, optional
        the transparency of the path-lines. Default is 0.2
    figsize: tuple, optional
        the figure size. Default is (5, 4)
    """
    if figsize is None:
        figsize = (5, 4)
    plt.figure(name, figsize=figsize)
    for d, c, n, r in zip(data, cases, names, repeats):
        if c[0] == code:
            ax = plt.subplot(np.max(repeats), 6, (r - 1) * 6 + cases.index(c) % 6 + 1, polar=True)
            _plot_arena_paths(d, name=n, lw=lw, alpha=alpha, save=False, show=False, ax=ax)
    plt.tight_layout()
    plt.show()


def _plot_arena_paths(data, name="arena", lw=1., alpha=.2, ax=None, save=False, show=True, figsize=None):
    """
    Plots the paths in the arena on the top of the gradients of the odours.

    Parameters
    ----------
    data: np.ndarray[complex]
         N x T matrix of 2D position (complex number) where each row represents a different fly and each column
        represents a different timentstep
    name: str, optional
        used as the title of the figure. Default is 'arena'
    lw: float, optional
        line width of the path. Default is 1
    alpha: float, optional
        the transparency parameter for the path line. Default is 0.2
    ax: optional
        the axis to draw the paths on
    save: bool, optional
        whether to save the figure. Default is False
    show: bool, optional
        whether to show the figure. Default is True
    figsize: tuple
        the size of the figure. Default is (2, 2)
    """
    if ax is None:
        if figsize is None:
            figsize = (2, 2)
        plt.figure(name, figsize=figsize)
        ax = plt.subplot(111, polar=True)

    # ax.set_theta_offset(np.pi)
    ax.set_theta_zero_location("W")

    draw_gradients(ax, radius=1.)

    if len(data) > 1:
        nb_flies, nb_steps = data.shape
        e_pre = int(.2 * nb_steps)
        s_post = int(.5 * nb_steps)
        for i in range(nb_flies):
            ax.plot(np.angle(data[i, :e_pre]), np.absolute(data[i, :e_pre]),
                    color=PRETRAINING_COLOUR, alpha=alpha, lw=lw)
            ax.plot(np.angle(data[i, e_pre:s_post]), np.absolute(data[i, e_pre:s_post]),
                    color=PUNISHMENT_COLOUR if 'quinine' in name else REWARD_COLOUR, alpha=alpha, lw=lw)
            ax.plot(np.angle(data[i, s_post:]), np.absolute(data[i, s_post:]),
                    color=POSTRAINING_COLOUR, alpha=alpha, lw=lw)

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylim([0, 1])

    if save:
        plt.savefig(name + ".svg", dpi=600)
    if show:
        plt.show()


def plot_arena_box(df, max_repeat=None, name="arena-box", show=True):
    """
    Plots box-plots based on the arena statistics.

    Parameters
    ----------
    df: pd.DataFrame
        contains the stats extracted from the flies running in the arena
    max_repeat : int, optional
        the maximum repeat to visualise. Default is None
    name: str, optional
        it will be used as the title of the figure. Default is 'arena-box'
    show: bool, optional
        if True, it shows the plot. Default is True
    """
    from .arena import FruitFly, gaussian_p

    repeats = np.unique(df["repeat"])
    if max_repeat is not None:
        repeats = repeats[repeats <= max_repeat]
    repeats = list(repeats)
    # mechanisms = [["susceptible"], ["restrained"], ["long-term memory"],
    #               ["susceptible", "restrained", "long-term memory"]]
    reinforcements = ["punishment", "reward"]
    odours = ["A", "B", "AB"]
    ms, rs, os = np.meshgrid(repeats, reinforcements, odours)

    labels, data = [""] * (6 * len(repeats)), [[]] * (6 * len(repeats))
    for repeat, reinforcement, odour in zip(ms.flatten(), rs.flatten(), os.flatten()):

        # mechanisms_not = list({"susceptible", "restrained", "long-term memory"} - set(mechanism))
        # dff = df[np.all([df[m] for m in mechanism] + [~df[m] for m in mechanisms_not], axis=0)]
        dff = df[np.any([df["repeat"] == repeat], axis=0)]
        dff = dff[np.any([dff["reinforcement"] == reinforcement], axis=0)]
        dff = dff[np.any([dff["paired odour"] == odour], axis=0)]

        d_pre = np.array(dff[dff["phase"] == "pre"]["PI"])
        d_learn = np.array(dff[dff["phase"] == "learn"]["PI"])
        d_post = np.array(dff[dff["phase"] == "post"]["PI"])

        # i_mecha = mechanisms.index(mechanism)
        i_repea = repeats.index(repeat)
        i_reinf = reinforcements.index(reinforcement)
        i_odour = odours.index(odour)
        i = i_repea * 6 + i_reinf * 3 + i_odour
        label = "%s-%s-%02d" % (odour.lower(), reinforcement[0], repeat)

        labels[i] = label
        data[i] = [d_pre, d_learn, d_post]

    m_a = FruitFly.a_source.real
    m_b = FruitFly.b_source.real
    s_a = FruitFly.a_sigma
    s_b = FruitFly.b_sigma

    # a = 1 / (2 * std1 ** 2) - 1 / (2 * std2 ** 2)
    # b = m2 / (std2 ** 2) - m1 / (std1 ** 2)
    # c = m1 ** 2 / (2 * std1 ** 2) - m2 ** 2 / (2 * std2 ** 2) - np.log(std2 / std1)
    # return np.roots([a, b, c])

    a = 1 / (2 * s_a ** 2) - 1 / (2 * s_b ** 2)
    b = m_b / (s_b ** 2) - m_a / (s_a ** 2)
    c = m_a ** 2 / (2 * s_a ** 2) - m_b ** 2 / (2 * s_b ** 2) - np.log(s_b / s_a)
    mid_point = np.roots([a, b, c]) * (m_b - m_a) / 2

    # print("A:", m_a, s_a)
    # print("B:", m_b, s_b)
    # print("Mid-points", mid_point)
    #
    # x = np.linspace(-1, 1, 1000)
    # y_a = gaussian_p(x + 0j, m_a + 0j, s_a)
    # y_b = gaussian_p(x + 0j, m_b + 0j, s_b)
    # plt.plot(x, y_a)
    # plt.plot(x, y_b)
    # plt.xticks([-1., -.6, -.2, .2, .6, 1])
    # plt.show()
    #
    # return

    plt.figure(name, figsize=(4, len(repeats)))
    ticks = []
    for i in range(len(labels)):
        plt.subplot(len(repeats), 1, i // 6 + 1)
        plt.plot([0, 7], [0, 0], 'grey', lw=.5, linestyle=":")
        plt.plot([0, 7], [mid_point, mid_point], 'grey', lw=2)
        data_0 = np.array(data[i][0])
        plt.boxplot(data_0[~np.isnan(data_0)], positions=[i % 6 + 0.8], notch=True, patch_artist=True,
                    boxprops=dict(color=PRETRAINING_COLOUR, facecolor=PRETRAINING_COLOUR),
                    whiskerprops=dict(color=PRETRAINING_COLOUR),
                    flierprops=dict(color=PRETRAINING_COLOUR, markeredgecolor=PRETRAINING_COLOUR, marker="."),
                    capprops=dict(color=PRETRAINING_COLOUR),
                    medianprops=dict(color=PRETRAINING_COLOUR))
        data_1 = np.array(data[i][1])
        r_colour = PUNISHMENT_COLOUR if ("p" in labels[i]) else REWARD_COLOUR
        plt.boxplot(data_1[~np.isnan(data_1)], positions=[i % 6 + 1.0], notch=True, patch_artist=True,
                    boxprops=dict(color=r_colour, facecolor=r_colour),
                    whiskerprops=dict(color=r_colour),
                    flierprops=dict(color=r_colour, markeredgecolor=r_colour, marker="."),
                    capprops=dict(color=r_colour),
                    medianprops=dict(color=r_colour))
        data_2 = np.array(data[i][2])
        plt.boxplot(data_2[~np.isnan(data_2)], positions=[i % 6 + 1.2], notch=True, patch_artist=True,
                    boxprops=dict(color=POSTRAINING_COLOUR, facecolor=POSTRAINING_COLOUR),
                    whiskerprops=dict(color=POSTRAINING_COLOUR),
                    flierprops=dict(color=POSTRAINING_COLOUR, markeredgecolor=POSTRAINING_COLOUR, marker="."),
                    capprops=dict(color=POSTRAINING_COLOUR),
                    medianprops=dict(color=POSTRAINING_COLOUR))
        if i < 6:
            title = labels[i].split('-')[0].upper() + "+" + labels[i].split('-')[1]
            title = title.replace("p", "shock")
            title = title.replace("r", "sugar")
            title = title.replace("AB", "A/B")
            ticks.append(title)
        if i // 6 >= len(repeats) - 1:
            plt.xticks([1, 2, 3, 4, 5, 6], ticks)
        else:
            plt.xticks([1, 2, 3, 4, 5, 6], [""] * 6)

        if i % 6 == 0:
            # plt.ylabel(["s (PI)", "r (PI)", "m (PI)", "s/r/m (PI)"][i // 6])
            plt.ylabel("r%02d (PI)" % repeats[i // 6])
        plt.yticks([-1, 0, 1], ["A", "0", "B"])
        plt.ylim([-1.05, 1.05])
        plt.xlim([.5, 6.5])

    plt.tight_layout()
    if show:
        plt.show()


def draw_gradients(ax, radius=1., draw_sources=False, levels=5):
    """
    Draws the odour distribution (gradients) on the given axis as contours.

    Parameters
    ----------
    ax
        the axis to draw the gradient to
    radius: float, optional
        the radius of the arena. Default is 1
    draw_sources: bool, optional
        whether to draw the sources or not. Default is True
    levels: int, optional
        the levels of the contours. Default is 20

    Returns
    -------
    ax
        the axis where the gradients are drawn
    """
    from .arena import FruitFly, gaussian_p

    a_mean, a_sigma = FruitFly.a_source, FruitFly.a_sigma
    b_mean, b_sigma = FruitFly.b_source, FruitFly.b_sigma
    x, y = np.meshgrid(np.linspace(-radius, radius, int(radius * 100) + 1),
                       np.linspace(-radius, radius, int(radius * 100) + 1))
    p = (x + y * 1j).T

    p_a = gaussian_p(p, a_mean, a_sigma)
    p_b = gaussian_p(p, b_mean, b_sigma)

    levels = np.linspace(FruitFly.i_threshold, max(np.max(p_a), np.max(p_b)), levels)
    rho, dist = np.angle(p), np.absolute(p)
    ax.contourf(rho-np.pi/2, dist, p_a, levels=levels, cmap=ODOUR_A_CMAP)
    ax.contourf(rho-np.pi/2, dist, p_b, levels=levels, cmap=ODOUR_B_CMAP)

    if draw_sources:
        ax.scatter(np.angle(a_mean), np.absolute(a_mean), s=20, color=ODOUR_A_COLOUR, label="odour A")
        ax.scatter(np.angle(b_mean), np.absolute(b_mean), s=20, color=ODOUR_B_COLOUR, label="odour B")

    return ax


def plot_arena_traces(data, d_names, cases, names, name="arena-paths", lw=1., alpha=.2, figsize=None):
    """
    Plots the neural traces in the arena for all the given cases and names.

    Parameters
    ----------
    data: list[np.ndarray]
        list of the paths for every case
    cases: list[list]
        list of the required cases
    names: list[str]
        list of names of the cases
    name: str, optional
        the title of the figure. Default is 'arena-paths'
    lw: float, optional
        the line width of the paths. Default is 0.1
    alpha: float, optional
        the transparency of the path-lines. Default is 0.2
    figsize: tuple, optional
        the figure size. Default is (5, 4)
    """
    if figsize is None:
        figsize = (15, 10)
    plt.figure(name, figsize=figsize)
    for d, dn, c, n in zip(data, d_names, cases, names):
        ax = plt.subplot(1, 6, cases.index(c) + 1)
        _plot_arena_trace(d, dn, name=n, lw=lw, alpha=alpha, save=False, show=False, ax=ax)
    plt.tight_layout()
    plt.show()


def _plot_arena_trace(data, d_names, name="arena", lw=1., alpha=.2, ax=None, save=False, show=True, figsize=None):
    """
    Plots the paths in the arena on the top of the gradients of the odours.

    Parameters
    ----------
    data: np.ndarray[complex]
         N x T matrix of 2D position (complex number) where each row represents a different fly and each column
        represents a different timentstep
    name: str, optional
        used as the title of the figure. Default is 'arena'
    lw: float, optional
        line width of the path. Default is 1
    alpha: float, optional
        the transparency parameter for the path line. Default is 0.2
    ax: optional
        the axis to draw the paths on
    save: bool, optional
        whether to save the figure. Default is False
    show: bool, optional
        whether to show the figure. Default is True
    figsize: tuple
        the size of the figure. Default is (2, 2)
    """
    if ax is None:
        if figsize is None:
            figsize = (2, 2)
        plt.figure(name, figsize=figsize)
        ax = plt.subplot(111)

    if len(data) > 1:
        if data.ndim > 3:  # synaptic weights
            nb_flies, nb_steps, _, nb_en = data.shape[:4]
        else:  # responses
            nb_flies, nb_steps, nb_en = data.shape[:3]
            nb_kc = None
        e_pre = int(.2 * nb_steps)
        s_post = int(.5 * nb_steps)
        x_pre = np.linspace(-30, 0, e_pre, endpoint=False)
        x_train = np.linspace(0, 50, s_post - e_pre, endpoint=False)
        x_post = np.linspace(50, 80, nb_steps - s_post, endpoint=False)
        for i in range(nb_en):
            ax.fill_between(x_pre,
                            3 * i + np.nanquantile(data[:, :e_pre, ..., i], .25, axis=(0, 2) if data.ndim > 3 else 0),
                            3 * i + np.nanquantile(data[:, :e_pre, ..., i], .75, axis=(0, 2) if data.ndim > 3 else 0),
                            facecolor=PRETRAINING_COLOUR, alpha=.2)
            ax.fill_between(x_train,
                            3 * i + np.nanquantile(data[:, e_pre:s_post, ..., i], .25, axis=(0, 2) if data.ndim > 3 else 0),
                            3 * i + np.nanquantile(data[:, e_pre:s_post, ..., i], .75, axis=(0, 2) if data.ndim > 3 else 0),
                            facecolor=PUNISHMENT_COLOUR if 'quinine' in name else REWARD_COLOUR, alpha=.2)
            ax.fill_between(x_post,
                            3 * i + np.nanquantile(data[:, s_post:, ..., i], .25, axis=(0, 2) if data.ndim > 3 else 0),
                            3 * i + np.nanquantile(data[:, s_post:, ..., i], .75, axis=(0, 2) if data.ndim > 3 else 0),
                            facecolor=POSTRAINING_COLOUR, alpha=.2)
            # ax.plot([x_pre] * 100, 3 * i + np.median(data[:, :e_pre, ..., i], axis=2), color='b', lw=.05)
            ax.plot(x_pre, 3 * i + np.nanmedian(data[:, :e_pre, ..., i], axis=(0, 2) if data.ndim > 3 else 0),
                    color=PRETRAINING_COLOUR, lw=lw)
            ax.plot(x_train, 3 * i + np.nanmedian(data[:, e_pre:s_post, ..., i], axis=(0, 2) if data.ndim > 3 else 0),
                    color=PUNISHMENT_COLOUR if 'quinine' in name else REWARD_COLOUR, lw=lw)
            ax.plot(x_post, 3 * i + np.nanmedian(data[:, s_post:, ..., i], axis=(0, 2) if data.ndim > 3 else 0),
                    color=POSTRAINING_COLOUR, lw=lw)

        ax.set_yticks(np.arange(0, nb_en * 3, 3))
        ax.set_yticklabels([r'$%s$' % nm for nm in d_names], fontsize=16)
        ax.set_xticks([-30, 0, 50, 79])
        ax.set_xticklabels([-30, 0, 50, 79], fontsize=16)
        ax.set_ylim([-1., nb_en * 3])

    if save:
        plt.savefig(name + ".svg", dpi=600)
    if show:
        plt.show()


def plot_arena_weights(data, d_names, cases, names, name="arena-paths", figsize=None):
    """
    Plots the KC-MBON synaptic weights during the arena experiment for all the given cases and names.

    Parameters
    ----------
    data: list[np.ndarray]
        list of the paths for every case
    cases: list[list]
        list of the required cases
    names: list[str]
        list of names of the cases
    name: str, optional
        the title of the figure. Default is 'arena-paths'
    figsize: tuple, optional
        the figure size. Default is (5, 4)
    """
    if figsize is None:
        figsize = (15, 10)
    plt.figure(name, figsize=figsize)
    for d, dn, c, n in zip(data, d_names, cases, names):
        ax = plt.subplot(1, 6, cases.index(c) + 1)
        _plot_arena_weights(d, dn, name=n, save=False, show=False, ax=ax)
    plt.tight_layout()
    plt.show()


def _plot_arena_weights(data, d_names, name="arena", ax=None, save=False, show=True, figsize=None):
    """
    Plots the KC-MBON synaptic weights during the arena experiment.

    Parameters
    ----------
    data: np.ndarray[complex]
         N x T matrix of 2D position (complex number) where each row represents a different fly and each column
        represents a different timentstep
    name: str, optional
        used as the title of the figure. Default is 'arena'
    ax: optional
        the axis to draw the paths on
    save: bool, optional
        whether to save the figure. Default is False
    show: bool, optional
        whether to show the figure. Default is True
    figsize: tuple
        the size of the figure. Default is (2, 2)
    """
    if ax is None:
        if figsize is None:
            figsize = (2, 2)
        plt.figure(name, figsize=figsize)
        ax = plt.subplot(111)

    if len(data) > 1:
        if data.ndim > 3:  # synaptic weights
            nb_flies, nb_steps, nb_kc, nb_en = data.shape[:4]
        else:  # responses
            nb_flies, nb_steps, nb_en = data.shape[:3]
            nb_kc = 1
        e_pre = int(.2 * nb_steps)
        s_post = int(.5 * nb_steps)

        r_colour = PUNISHMENT_COLOUR if 'quinine' in name else REWARD_COLOUR
        data_i = np.ones((data.shape[1], (nb_en // 2) * (nb_kc + 1) - 1, 3), dtype=float)
        for i in range(nb_en // 2):
            data_t = np.nanmedian(data[..., i + nb_en // 2], axis=0)
            i_start = i * (nb_kc + 1)
            i_end = (i + 1) * (nb_kc + 1) - 1

            data_i[:e_pre, i_start:i_end] = data_t[:e_pre][..., np.newaxis] * PRETRAINING_COLOUR
            data_i[e_pre:s_post, i_start:i_end] = data_t[e_pre:s_post][..., np.newaxis] * r_colour
            data_i[s_post:, i_start:i_end] = data_t[s_post:][..., np.newaxis] * (1 - POSTRAINING_COLOUR)

        ax.imshow(np.clip(np.transpose(data_i, (1, 0, 2)) / 2., 0, 1), aspect="auto")

        ax.set_yticks(np.arange((nb_kc + 1) / 2, (nb_en // 2) * (nb_kc + 1), nb_kc + 1) - 1)
        ax.set_yticklabels([r'$%s$' % nm for nm in d_names[nb_en // 2:]], fontsize=16)
        ax.set_xticks([0, .2 * nb_steps, .5 * nb_steps, nb_steps-1])
        ax.set_xticklabels([-30, 0, 50, 80], fontsize=16)

    if save:
        plt.savefig(name + ".svg", dpi=600)
    if show:
        plt.show()
