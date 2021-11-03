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

import re

REWARD_COLOUR = np.array(colors.to_rgb("green"))
PUNISHMENT_COLOUR = np.array(colors.to_rgb("red"))
ODOUR_A_COLOUR = np.array([246, 74, 138]) / 255
ODOUR_B_COLOUR = np.array([255, 165, 0]) / 255
ODOUR_A_CMAP = ListedColormap(np.clip(np.linspace(np.ones(3), ODOUR_A_COLOUR, 100), 0, 1))
ODOUR_B_CMAP = ListedColormap(np.clip(np.linspace(np.ones(3), ODOUR_B_COLOUR, 100), 0, 1))
PRETRAINING_COLOUR = np.array(colors.to_rgb("blue"))
POSTRAINING_COLOUR = np.array(colors.to_rgb("black"))
MAX_NUMBER_OF_EXPERIMENTS = 3


def plot_responses_from_model(ms, nids=None, only_nids=True, figsize=None, show_legend=True):
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
    _plot_mean_responses([va_q25, va_q50, va_q75], [vb_q25, vb_q50, vb_q75],
                         exp_names=exp_names, neuron_names=names,
                         nb_timesteps=ms[0][0].nb_timesteps, nb_trials=ms[0][0].nb_trials,
                         title=title, figsize=figsize, show_legend=show_legend and only_nids)


def plot_responses_from_data(sum_res, only_nids=True, figsize=None, show_legend=True):
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

    _plot_mean_responses(va, vb, exp_names="reversal", neuron_names=genotypes, is_data=True,
                         nb_timesteps=3, nb_trials=26,
                         title=title, show_legend=show_legend and only_nids, figsize=figsize)


def _plot_mean_responses(va, vb, neuron_names, exp_names=None, nb_timesteps=1, nb_trials=1,
                         title=None, figsize=None, show_legend=True, is_data=False, vc=None):

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
    nb_plots = nb_neurons * 2 + nb_neurons * int(vc is not None)

    nb_rows = 4 + 2 * int(vc is not None)
    nb_cols = nb_plots // nb_rows
    while nb_cols > 7:
        nb_rows += 4
        nb_cols = nb_plots // nb_rows + 1

    if figsize is None:
        figsize = (8 - 2 * int(nb_neurons != 12), nb_rows + 1)

    nb_groups = 2 + int(vc is not None)

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
                axa = plt.subplot(nb_rows, nb_cols, nb_groups * (j // nb_cols) * nb_cols + j % nb_cols + 1)
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
                axb = plt.subplot(nb_rows, nb_cols, (nb_groups * (j // nb_cols) + 1) * nb_cols + j % nb_cols + 1)
                axb.set_xticks(_x_ticks)
                if vc is None:
                    axb.set_xticklabels(x_ticks_)
                else:
                    axb.set_xticklabels(["" for _ in x_ticks_])
                axb.set_yticks([0, ymax/2, ymax])
                axb.set_ylim(ylim)
                axb.set_xlim([0, n + 1 + 1 / (nb_timesteps - 1)])
                axb.tick_params(labelsize=8)
                if j % nb_cols == 0:
                    axb.set_ylabel("Odour B", fontsize=8)
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

        if vc is not None:
            x_c = x_ + 1 - 1 / (nb_timesteps - 1)
            for j in range(nb_neurons):
                jn = j + 2 * nb_neurons

                label = None
                c = 1 - (i + 1) / (MAX_NUMBER_OF_EXPERIMENTS + 1)
                c_col = tuple((np.array(ODOUR_A_CMAP(c)) + np.array(ODOUR_B_CMAP(c))) / 2.)
                if j == nb_neurons - 1:
                    label = exp_names[i]

                vcj_q25, vcj_q50, vcj_q75 = [v[i, j] for v in vc]
                if len(subs) <= jn:
                    axc = plt.subplot(nb_rows, nb_cols, (nb_groups * (j // nb_cols) + 2) * nb_cols + j % nb_cols + 1)
                    axc.set_xticks(_x_ticks)
                    axc.set_xticklabels(x_ticks_)
                    axc.set_yticks([0, ymax/2, ymax])
                    axc.set_ylim(ylim)
                    axc.set_xlim([0, n + 1 + 1 / (nb_timesteps - 1)])
                    axc.tick_params(labelsize=8)
                    if j % nb_cols == 0:
                        axc.set_ylabel("Odour A / B", fontsize=8)
                        if nb_rows == 6:
                            axc.text(-3, -.8, "Trial #", fontsize=8)
                    else:
                        axc.set_yticklabels([""] * 3)
                        axc.spines['left'].set_visible(False)
                        axc.set_yticks([])

                    axc.spines['top'].set_visible(False)
                    axc.spines['right'].set_visible(False)

                    c_acol = tuple((np.array(ODOUR_A_CMAP(.2)) + np.array(ODOUR_B_CMAP(0.2))) / 2)

                    if nb_models == 1:
                        axc.fill_between(x_c[2:12], vcj_q25[2:12], vcj_q75[2:12], facecolor=c_acol, alpha=.2)
                    axc.plot(x_c[:3], vcj_q50[:3], color=(.8, .8, .8))
                    axc.plot(x_c[2:12], vcj_q50[2:12], color=c_acol, lw=2, label="acquisition")
                    subs.append(axc)

                if nb_models == 1:
                    subs[jn].fill_between(x_c[13:23], vcj_q25[13:23], vcj_q75[13:23], facecolor=c_col, alpha=.2)

                subs[jn].plot(x_c[11:14], vcj_q50[11:14], color=(.8, .8, .8), lw=2)
                subs[jn].plot(x_c[22:], vcj_q50[22:], color=(.8, .8, .8), lw=2)
                subs[jn].plot(x_c[13:23], vcj_q50[13:23], color=c_col, lw=2, label=label)

                if i > 0 or np.all([ps not in neuron_names[j] for ps in plot_shock]):
                    continue
                shock_i = [3, 5, 7, 9, 11]
                subs[jn].plot(x_c[shock_i], vcj_q50[shock_i], color=PUNISHMENT_COLOUR, marker='.', linestyle=' ')

    if show_legend:
        for i in range(3):
            subs[i * len(subs)//(2 + int(vc is not None)) - 1].legend(fontsize=8, bbox_to_anchor=(1.05, 1.35),
                                                                      loc='upper left', framealpha=0., labelspacing=1.)
            # subs[-1].legend(fontsize=8, bbox_to_anchor=(1.05, 1.35), loc='upper left', framealpha=0., labelspacing=1)
    plt.tight_layout()
    plt.show()


def plot_weights(ms, nids=None, only_nids=True, figsize=None, show_legend=True):
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
    if figsize is None:
        if only_nids:
            figsize = (9, 7)
        else:
            figsize = (5, 7)
    plt.figure(title, figsize=figsize)

    exp_names = []
    vajs, vabjs, vbjs = [], [], []
    for k in range(nb_repeats):

        vajs.append([])
        vbjs.append([])
        vabjs.append([])
        for i in range(nb_models):
            nb_timesteps = ms[k][i].nb_timesteps
            w = ms[k][i].w_k2m

            # trial, odour, time-step, neuron
            va = np.nanmean(w[1:, :4], axis=1).reshape((-1, nb_odours, nb_timesteps, w.shape[-1]))[:, 0].reshape(
                (-1, w.shape[-1]))
            vab = np.nanmean(w[1:, 4:7], axis=1).reshape((-1, nb_odours, nb_timesteps, w.shape[-1]))[:, 0].reshape(
                (-1, w.shape[-1]))
            vb = np.nanmean(w[1:, 7:], axis=1).reshape((-1, nb_odours, nb_timesteps, w.shape[-1]))[:, 1].reshape(
                (-1, w.shape[-1]))

            if only_nids:
                va = va[:, nids]
                vab = vab[:, nids]
                vb = vb[:, nids]

            vajs[k].append([])
            vabjs[k].append([])
            vbjs[k].append([])
            for j in range(nb_neurons):
                vajs[k][i].append(va[:, j].reshape((-1, nb_timesteps))[:, 1:].reshape((-1,)))
                vabjs[k][i].append(vab[:, j].reshape((-1, nb_timesteps))[:, 1:].reshape((-1,)))
                vbjs[k][i].append(vb[:, j].reshape((-1, nb_timesteps))[:, 1:].reshape((-1,)))

    va_q25 = np.nanquantile(vajs, .25, axis=0)
    va_q50 = np.nanquantile(vajs, .50, axis=0)
    va_q75 = np.nanquantile(vajs, .75, axis=0)
    vab_q25 = np.nanquantile(vabjs, .25, axis=0)
    vab_q50 = np.nanquantile(vabjs, .50, axis=0)
    vab_q75 = np.nanquantile(vabjs, .75, axis=0)
    vb_q25 = np.nanquantile(vbjs, .25, axis=0)
    vb_q50 = np.nanquantile(vbjs, .50, axis=0)
    vb_q75 = np.nanquantile(vbjs, .75, axis=0)
    for i in range(nb_models):
        exp_names.append(ms[0][i].routine_name)
    _plot_mean_responses([va_q25, va_q50, va_q75], [vb_q25, vb_q50, vb_q75],
                         vc=[vab_q25, vab_q50, vab_q75], exp_names=exp_names, neuron_names=names,
                         nb_timesteps=ms[0][0].nb_timesteps, nb_trials=ms[0][0].nb_trials,
                         title=title, figsize=figsize, show_legend=show_legend and only_nids)


def plot_arena_fishbone(df, odours_visited="A and B", code="srm", rpe=False, nb_repeats=10,
                        lw=.2, alpha=.5, nb_samples=3, verbose=False):

    nb_flies, _ = df[np.all([df["phase"] == "pre", df["paired odour"] == "A", df["reinforcement"] == "reward",
                             df["susceptible"], df["restrained"], df["long-term memory"],
                             df["repeat"] == 1], axis=0)].shape

    plt.figure(("rpe-" if rpe else "") + "arena-pi-lines-" + odours_visited.lower().replace(" ", "-"), figsize=(8, 3))

    data = {
        "A - punishment": np.zeros((nb_repeats, 6, nb_flies), dtype=float),
        "B - punishment": np.zeros((nb_repeats, 6, nb_flies), dtype=float),
        "AB - punishment": np.zeros((nb_repeats, 6, nb_flies), dtype=float),
        "A - reward": np.zeros((nb_repeats, 6, nb_flies), dtype=float),
        "B - reward": np.zeros((nb_repeats, 6, nb_flies), dtype=float),
        "AB - reward": np.zeros((nb_repeats, 6, nb_flies), dtype=float)
    }

    for i, title in enumerate(data):
        details = re.match(r"([\w]{1,2}) - ([\w]+)", title)
        odour = details.group(1)
        reinforcement = details.group(2)
        col = i + 1

        for r in range(nb_repeats):
            for t, time in enumerate(["time_A", "time_B"]):
                for p, phase in enumerate(["pre", "learn", "post"]):
                    data[title][r, t * 3 + p, :] = df[np.all([df["phase"] == phase,
                                                              df["reinforcement"] == reinforcement,
                                                              df["paired odour"] == odour,
                                                              df["susceptible"] == ("s" in code),
                                                              df["restrained"] == ("r" in code),
                                                              df["long-term memory"] == ("m" in code),
                                                              df["repeat"] == r + 1], axis=0)][time]
        data[title][:, [0, 3]] /= 20
        data[title][:, [1, 4]] /= 30
        data[title][:, [2, 5]] /= 50

        data[title] = np.cumsum(data[title], axis=0)
        data[title] /= (np.array(nb_repeats) + 1)  # normalise with the repeat time

        a_pr = data[title][:, 0]  # - data[title][:, 1]
        a_tr = data[title][:, 1]  # - data[title][:, 1]
        a_ps = data[title][:, 2]  # - data[title][:, 1]
        b_pr = data[title][:, 3]  # - data[title][:, 4]
        b_tr = data[title][:, 4]  # - data[title][:, 4]
        b_ps = data[title][:, 5]  # - data[title][:, 4]

        ia = np.any(a_tr > 0, axis=0)
        ib = np.any(b_tr > 0, axis=0)

        i = np.zeros_like(ia)
        operation = None
        is_not = False
        iis = {"a": ia, "b": ib}
        for s in odours_visited.lower().split(" "):
            if s in ["not"]:
                is_not = not is_not
            elif s in ["and", "or"]:
                operation = s
            elif s in ["a", "b"]:
                ii = ~iis[s] if is_not else iis[s]
                is_not = False
                if operation is None:
                    i = ii
                else:
                    if operation in ["and"]:
                        i = i & ii
                        operation = None
                    elif operation in ["or"]:
                        i = i | ii
                        operation = None

        if verbose:
            print(f"{np.sum(i)} / {i.shape[0]}")

        ca = np.zeros((a_pr[:, i].shape[0] * 3, a_pr[:, i].shape[1]), dtype=a_pr.dtype)
        ca[0::3, :] = a_pr[:, i]
        ca[1::3, :] = a_tr[:, i]
        ca[2::3, :] = a_ps[:, i]
        ca_q25 = np.nanquantile(ca, .25, axis=1)
        ca_q50 = np.nanquantile(ca, .50, axis=1)
        ca_q75 = np.nanquantile(ca, .75, axis=1)

        cb = np.zeros((b_pr[:, i].shape[0] * 3, b_pr[:, i].shape[1]), dtype=b_pr.dtype)
        cb[0::3, :] = b_pr[:, i]
        cb[1::3, :] = b_tr[:, i]
        cb[2::3, :] = b_ps[:, i]
        cb_q25 = np.nanquantile(cb, .25, axis=1)
        cb_q50 = np.nanquantile(cb, .50, axis=1)
        cb_q75 = np.nanquantile(cb, .75, axis=1)

        c = (ca - cb) / (ca + cb + np.finfo(float).eps)
        c[np.isnan(c)] = 0.
        c_q25 = np.nanquantile(c, .25, axis=1)
        c_q50 = np.nanquantile(c, .50, axis=1)
        c_q75 = np.nanquantile(c, .75, axis=1)

        color_r = 'red' if reinforcement == "punishment" else "green"

        t_in = np.argmax(c > 0, axis=0) // 3
        s, p = [], []
        for t in range(nb_repeats):
            ss = np.arange(c.shape[1])[t_in == t]
            if len(ss) > 0:
                s.append(ss[0])
                p.append(len(ss))
        s = np.array(s)[np.argsort(p)[::-1]][:nb_samples]

        y_max = np.maximum(np.sqrt(ca.max()), np.sqrt(cb.max())) * 1.1

        x = np.arange(30) / 3 + 1

        ax = plt.subplot(2, 6, col)
        plt.plot(x, np.sqrt(ca[:, s]), linestyle=(0, (5, 10)), color=ODOUR_A_CMAP(0.99), lw=lw, alpha=alpha)
        plt.plot(x, -np.sqrt(cb[:, s]), linestyle=(0, (5, 10)), color=ODOUR_B_CMAP(0.99), lw=lw, alpha=alpha)
        plt.plot(np.array([x] * len(s)).T.reshape((-1, 3, len(s))).transpose((0, 2, 1)).reshape((-1, 3)).T,
                 np.sqrt(ca[:, s]).reshape((-1, 3, len(s))).transpose((0, 2, 1)).reshape((-1, 3)).T, '-',
                 color=ODOUR_A_CMAP(0.99), lw=lw, alpha=alpha)
        plt.plot(np.array([x] * len(s)).T.reshape((-1, 3, len(s))).transpose((0, 2, 1)).reshape((-1, 3)).T,
                 -np.sqrt(cb[:, s]).reshape((-1, 3, len(s))).transpose((0, 2, 1)).reshape((-1, 3)).T, '-',
                 color=ODOUR_B_CMAP(0.99), lw=lw, alpha=alpha)

        plt.plot(x.reshape((-1, 3)).T, np.sqrt(ca_q50).reshape((-1, 3)).T, '-', color=ODOUR_A_CMAP(0.99), lw=2)
        plt.plot(x.reshape((-1, 3)).T, -np.sqrt(cb_q50).reshape((-1, 3)).T, '-', color=ODOUR_B_CMAP(0.99), lw=2)

        if "A" in odour:
            plt.plot(x[1::3], np.sqrt(ca_q50[1::3]), '.', color=color_r)
        if "B" in odour:
            plt.plot(x[1::3], -np.sqrt(cb_q50[1::3]), '.', color=color_r)

        plt.ylim(-y_max, y_max)
        plt.xlim(0.5, 11.4)
        plt.xticks([1, nb_repeats / 2, nb_repeats])
        plt.yticks([-y_max, 0, y_max], ["B", "0", "A"])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        ax = plt.subplot(2, 6, col + 6)

        plt.plot(x, c[:, s], linestyle=(0, (5, 10)), color='black', lw=lw, alpha=alpha)
        plt.plot(np.array([x] * len(s)).T.reshape((-1, 3, len(s))).transpose((0, 2, 1)).reshape((-1, 3)).T,
                 c[:, s].reshape((-1, 3, len(s))).transpose((0, 2, 1)).reshape((-1, 3)).T, '-',
                 color='black', lw=lw, alpha=alpha)

        plt.plot(x.reshape((-1, 3)).T, c_q50.reshape((-1, 3)).T, 'k-', lw=2)

        plt.plot(x[1::3], c_q50[1::3], '.', color=color_r)

        plt.ylim(-1.1, 1.1)
        plt.xlim(0.5, 11.4)
        plt.xticks([1, nb_repeats / 2, nb_repeats])
        plt.yticks([-1, 0, 1], ["B", "0", "A"])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_arena_fishbone_weights(df, odours_visited="A and B", code="srm", rpe=False, nb_kc=10, nb_kc_a=7, nb_kc_b=6,
                                nb_repeats=10, lw=.2, alpha=.5, nb_samples=3, verbose=False):

    nb_flies, _ = df[np.all([df["phase"] == "pre", df["paired odour"] == "A", df["reinforcement"] == "reward",
                             df["susceptible"], df["restrained"], df["long-term memory"],
                             df["repeat"] == 1], axis=0)].shape

    plt.figure(("rpe-" if rpe else "") + "arena-pi-weights-" + odours_visited.lower().replace(" ", "-"), figsize=(8, 8))

    data = {
        "A - punishment": np.zeros((nb_repeats, 6 * 3, nb_kc, nb_flies), dtype=float),
        "B - punishment": np.zeros((nb_repeats, 6 * 3, nb_kc, nb_flies), dtype=float),
        "AB - punishment": np.zeros((nb_repeats, 6 * 3, nb_kc, nb_flies), dtype=float),
        "A - reward": np.zeros((nb_repeats, 6 * 3, nb_kc, nb_flies), dtype=float),
        "B - reward": np.zeros((nb_repeats, 6 * 3, nb_kc, nb_flies), dtype=float),
        "AB - reward": np.zeros((nb_repeats, 6 * 3, nb_kc, nb_flies), dtype=float)
    }

    d_time = {
        "A - punishment": np.zeros((nb_repeats, 6, nb_flies), dtype=float),
        "B - punishment": np.zeros((nb_repeats, 6, nb_flies), dtype=float),
        "AB - punishment": np.zeros((nb_repeats, 6, nb_flies), dtype=float),
        "A - reward": np.zeros((nb_repeats, 6, nb_flies), dtype=float),
        "B - reward": np.zeros((nb_repeats, 6, nb_flies), dtype=float),
        "AB - reward": np.zeros((nb_repeats, 6, nb_flies), dtype=float)
    }

    x = np.arange(3 * nb_repeats) / 3 + 1

    colour_a = ODOUR_A_CMAP(0.99)
    colour_b = ODOUR_B_CMAP(0.99)
    colour_ab = tuple((np.array(colour_a) + np.array(colour_b)) / 2)

    for i, title in enumerate(data):
        details = re.match(r"([\w]{1,2}) - ([\w]+)", title)
        odour = details.group(1)
        reinforcement = details.group(2)
        col = i + 1

        for r in range(nb_repeats):
            for p, phase in enumerate(["pre", "learn", "post"]):
                for k in range(nb_kc):
                    for n, neuron in enumerate(["s+", "s-", "r+", "r-", "m+", "m-"]):
                        prop = "k%d2%s" % (k, neuron)
                        data[title][r, n * 3 + p, k, :] = df[np.all([df["phase"] == phase,
                                                                     df["reinforcement"] == reinforcement,
                                                                     df["paired odour"] == odour,
                                                                     df["susceptible"] == ("s" in code),
                                                                     df["restrained"] == ("r" in code),
                                                                     df["long-term memory"] == ("m" in code),
                                                                     df["repeat"] == r + 1], axis=0)][prop]

                for t, time in enumerate(["time_A", "time_B"]):
                    d_time[title][r, t * 3 + p, :] = df[np.all([df["phase"] == phase,
                                                                df["reinforcement"] == reinforcement,
                                                                df["paired odour"] == odour,
                                                                df["susceptible"] == ("s" in code),
                                                                df["restrained"] == ("r" in code),
                                                                df["long-term memory"] == ("m" in code),
                                                                df["repeat"] == r + 1], axis=0)][time]
        d_time[title][:, [0, 3]] /= 20
        d_time[title][:, [1, 4]] /= 30
        d_time[title][:, [2, 5]] /= 50

        d_time[title] = np.cumsum(d_time[title], axis=0)
        d_time[title] /= (np.array(nb_repeats) + 1)  # normalise with the repeat time

        a_pr = d_time[title][:, 0]
        a_tr = d_time[title][:, 1]
        a_ps = d_time[title][:, 2]
        b_pr = d_time[title][:, 3]
        b_tr = d_time[title][:, 4]
        b_ps = d_time[title][:, 5]

        ia = np.any(a_tr > 0, axis=0)
        ib = np.any(b_tr > 0, axis=0)

        i = np.zeros_like(ia)
        operation = None
        is_not = False
        iis = {"a": ia, "b": ib}
        for s in odours_visited.lower().split(" "):
            if s in ["not"]:
                is_not = not is_not
            elif s in ["and", "or"]:
                operation = s
            elif s in ["a", "b"]:
                ii = ~iis[s] if is_not else iis[s]
                is_not = False
                if operation is None:
                    i = ii
                else:
                    if operation in ["and"]:
                        i = i & ii
                        operation = None
                    elif operation in ["or"]:
                        i = i | ii
                        operation = None

        if verbose:
            print(f"{np.sum(i)} / {i.shape[0]}")

        ca = np.zeros((a_pr[:, i].shape[0] * 3, a_pr[:, i].shape[1]), dtype=a_pr.dtype)
        ca[0::3, :] = a_pr[:, i]
        ca[1::3, :] = a_tr[:, i]
        ca[2::3, :] = a_ps[:, i]

        cb = np.zeros((b_pr[:, i].shape[0] * 3, b_pr[:, i].shape[1]), dtype=b_pr.dtype)
        cb[0::3, :] = b_pr[:, i]
        cb[1::3, :] = b_tr[:, i]
        cb[2::3, :] = b_ps[:, i]

        c = (ca - cb) / (ca + cb + np.finfo(float).eps)
        c[np.isnan(c)] = 0.

        color_r = 'red' if reinforcement == "punishment" else "green"

        t_in = np.argmax(c > 0, axis=0) // 3
        s, p = [], []
        for t in range(nb_repeats):
            ss = np.arange(c.shape[1])[t_in == t]
            if len(ss) > 0:
                s.append(ss[0])
                p.append(len(ss))
        s = np.array(s)[np.argsort(p)[::-1]][:nb_samples]

        for n, neuron in enumerate(["s_{at}", "s_{av}", "r_{at}", "r_{av}", "m_{at}", "m_{av}"]):
            k_pr = np.transpose(data[title][:, n*3+0, :, i], axes=(1, 2, 0))
            k_tr = np.transpose(data[title][:, n*3+1, :, i], axes=(1, 2, 0))
            k_po = np.transpose(data[title][:, n*3+2, :, i], axes=(1, 2, 0))

            ck = np.zeros((k_pr.shape[0] * 3, k_pr.shape[1], k_pr.shape[2]), dtype=k_pr.dtype)
            ck[0::3, :] = k_pr
            ck[1::3, :] = k_tr
            ck[2::3, :] = k_po
            ck[np.isnan(ck)] = 0.

            nb_a_only = np.min([nb_kc_a, nb_kc - nb_kc_b])
            nb_b_only = np.min([nb_kc - nb_kc_a, nb_kc_b])

            ca = np.nanmedian(ck[:, :nb_a_only], axis=1)
            cb = np.nanmedian(ck[:, -nb_b_only:], axis=1)
            co = np.nanmedian(ck[:, nb_a_only:-nb_b_only], axis=1)

            ca_q50 = np.nanmedian(ca, axis=1)
            cb_q50 = np.nanmedian(cb, axis=1)
            co_q50 = np.nanmedian(co, axis=1)

            ax = plt.subplot(6, 6, 6 * n + col)

            if "A" in odour:
                plt.plot(x, -co[:, s], linestyle=(0, (5, 10)), color=colour_ab, lw=lw, alpha=alpha)
                plt.plot(np.array([x] * len(s)).T.reshape((-1, 3, len(s))).transpose((0, 2, 1)).reshape((-1, 3)).T,
                         -co[:, s].reshape((-1, 3, len(s))).transpose((0, 2, 1)).reshape((-1, 3)).T, '-',
                         color=colour_ab, lw=lw, alpha=alpha)
            if "B" in odour:
                plt.plot(x, co[:, s], linestyle=(0, (5, 10)), color=colour_ab, lw=lw, alpha=alpha)
                plt.plot(np.array([x] * len(s)).T.reshape((-1, 3, len(s))).transpose((0, 2, 1)).reshape((-1, 3)).T,
                         co[:, s].reshape((-1, 3, len(s))).transpose((0, 2, 1)).reshape((-1, 3)).T, '-',
                         color=colour_ab, lw=lw, alpha=alpha)

            plt.plot(x, ca[:, s], linestyle=(0, (5, 10)), color=colour_a, lw=lw, alpha=alpha)
            plt.plot(x, -cb[:, s], linestyle=(0, (5, 10)), color=colour_b, lw=lw, alpha=alpha)

            plt.plot(np.array([x] * len(s)).T.reshape((-1, 3, len(s))).transpose((0, 2, 1)).reshape((-1, 3)).T,
                     ca[:, s].reshape((-1, 3, len(s))).transpose((0, 2, 1)).reshape((-1, 3)).T, '-',
                     color=colour_a, lw=lw, alpha=alpha)
            plt.plot(np.array([x] * len(s)).T.reshape((-1, 3, len(s))).transpose((0, 2, 1)).reshape((-1, 3)).T,
                     -cb[:, s].reshape((-1, 3, len(s))).transpose((0, 2, 1)).reshape((-1, 3)).T, '-',
                     color=colour_b, lw=lw, alpha=alpha)

            if "A" in odour:
                plt.plot(x.reshape((-1, 3)).T, -co_q50.reshape((-1, 3)).T, '-', color=colour_ab, lw=1.5)
            if "B" in odour:
                plt.plot(x.reshape((-1, 3)).T, co_q50.reshape((-1, 3)).T, '-', color=colour_ab, lw=1.5)

            plt.plot(x.reshape((-1, 3)).T, ca_q50.reshape((-1, 3)).T, '-', color=colour_a, lw=2)
            plt.plot(x.reshape((-1, 3)).T, -cb_q50.reshape((-1, 3)).T, '-', color=colour_b, lw=2)

            if "A" in odour:
                plt.plot(x[1::3], ca_q50[1::3], '.', color=color_r)
            if "B" in odour:
                plt.plot(x[1::3], -cb_q50[1::3], '.', color=color_r)

            plt.ylim(-1.1, 1.1)
            plt.xlim(0.5, 11.4)
            plt.xticks([1,nb_repeats / 2, nb_repeats])
            plt.yticks([-1, 0, 1])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)

    plt.tight_layout()
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
