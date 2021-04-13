from incentive.models_base import MBModel

from typing import List
from scipy.stats import circmean, circstd
from matplotlib import cm
from matplotlib.colors import Normalize

import matplotlib.pyplot as plt
import numpy as np


def plot_population(ms, nids=None, vmin=-2., vmax=2., only_nids=False, figsize=None):
    """
    Plots the responses as a matrix where the rows are different neurons and the columns are the time-steps. The colour
    reveals the actual responses of the neurons.

    :param ms: the models where the values are taken from
    :type ms: List[MBModel]
    :param nids: the indices of the neurons that we want to show their names
    :type nids: List[int]
    :param vmin: the lower bound for the colour map. Default is -2.
    :type vmin: float
    :param vmax: the upper bound for the colour map. Default is 2.
    :type vmin: float
    :param only_nids: when True, only the specified neurons are plotted.
    :type only_nids: bool
    """
    title = "motivation-" + '-'.join(str(ms[0]).split("'")[1:-1:2])

    nb_models = len(ms)
    xticks = ["%d" % (i+1) for i in range(16)]

    if figsize is None:
        figsize = (7.5, 10)
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

        v = ms[i]._v
        w = ms[i].w_k2m

        ax = plt.subplot(nb_models * 2, 2, 1 + i * 4)
        # trial, odour, time-step, neuron
        va = v[1:].reshape((-1, 2, nb_timesteps, v.shape[-1]))[:, ::2].reshape((-1, v.shape[-1]))
        if only_nids:
            va = va[:, nids]
        plt.imshow(va.T, vmin=vmin, vmax=vmax, cmap="coolwarm", aspect="auto")
        if "reversal" in ms[i].routine_name:
            plt.plot([np.array([8, 9, 10, 11, 12, 13]) * nb_timesteps - 1] * 2, [[0] * 6, [ylim] * 6], 'r-')
        elif "unpaired" in ms[i].routine_name:
            plt.plot([(np.array([8, 9, 10, 11, 12, 13]) - 1) * nb_timesteps] * 2, [[0] * 6, [ylim] * 6], 'r-')
        plt.xticks(nb_timesteps * np.arange(nb_trials // 2) + nb_timesteps / 4, xticks[:nb_trials // 2])
        plt.yticks(np.arange(len(nids)) if only_nids else nids, [r'$%s$' % tick for tick in yticks[nids]])
        ax.yaxis.set_ticks_position('both')
        plt.title("%s - odour A - value" % ms[i].routine_name, color="C%d" % (2 * i + 0))

        ax = plt.subplot(nb_models * 2, 2, 2 + i * 4)
        vb = v[1:].reshape((-1, 2, nb_timesteps, v.shape[-1]))[:, 1::2].reshape((-1, v.shape[-1]))
        if only_nids:
            vb = vb[:, nids]
        plt.imshow(vb.T, vmin=vmin, vmax=vmax, cmap="coolwarm", aspect="auto")
        plt.plot([np.array([2, 3, 4, 5, 6]) * nb_timesteps - 1] * 2, [[0] * 5, [ylim] * 5], 'r-')
        plt.xticks(nb_timesteps * np.arange(nb_trials // 2) + nb_timesteps / 4, xticks[:nb_trials // 2])
        plt.yticks(np.arange(len(nids)) if only_nids else nids, [r'$%s$' % tick for tick in yticks[nids]])
        ax.yaxis.set_ticks_position('both')
        ax.tick_params(labelleft=False, labelright=True)
        plt.title("%s - odour B - value" % ms[i].routine_name, color="C%d" % (2 * i + 1))

        ax = plt.subplot(nb_models * 2, 2, 3 + i * 4)
        wa = w[1:, 0]
        if only_nids:
            wa = wa[:, nids]
        plt.imshow(wa.T, vmin=vmin, vmax=vmax, cmap="coolwarm", aspect="auto")
        if "reversal" in ms[i].routine_name:
            plt.plot([np.array([8, 9, 10, 11, 12, 13]) * 2 * nb_timesteps - 1] * 2, [[0] * 6, [ylim] * 6], 'r-')
        elif "unpaired" in ms[i].routine_name:
            plt.plot([(np.array([8, 9, 10, 11, 12, 13]) - 1) * 2 * nb_timesteps] * 2, [[0] * 6, [ylim] * 6], 'r-')
        plt.xticks(2 * nb_timesteps * np.arange(nb_trials // 2) + nb_timesteps, xticks[:nb_trials // 2])
        plt.yticks(np.arange(len(nids)) if only_nids else nids, [r'$%s$' % tick for tick in yticks[nids]])
        ax.yaxis.set_ticks_position('both')
        plt.title("%s - odour A - weights" % ms[i].routine_name, color="C%d" % (2 * i + 0))

        ax = plt.subplot(nb_models * 2, 2, 4 + i * 4)
        wb = w[1:, 5]
        if only_nids:
            wb = wb[:, nids]
        plt.imshow(wb.T, vmin=vmin, vmax=vmax, cmap="coolwarm", aspect="auto")
        plt.plot([np.array([2, 3, 4, 5, 6]) * 2 * nb_timesteps - 1] * 2, [[0] * 5, [ylim] * 5], 'r-')
        plt.xticks(2 * nb_timesteps * np.arange(nb_trials // 2) + nb_timesteps, xticks[:nb_trials // 2])
        plt.yticks(np.arange(len(nids)) if only_nids else nids, [r'$%s$' % tick for tick in yticks[nids]])
        ax.yaxis.set_ticks_position('both')
        ax.tick_params(labelleft=False, labelright=True)
        plt.title("%s - odour B - weights" % ms[i].routine_name, color="C%d" % (2 * i + 1))

    # plt.colorbar()
    plt.tight_layout()
    plt.show()


def plot_weights_matrices(ms, nids=None, vmin=-2., vmax=2., only_nids=False, figsize=None):
    """
    Plots the KC-MBON synaptic weights strength as a matrix where the rows are different neurons and the columns are the
    time-steps. The colour reveals the actual strength of the synaptic weights from the KCs to MBONs. The data are split
    in two groups related to odour A and odour B, and the average synaptic strength for each group is plotted.

    :param ms: the models where the values are taken from
    :type ms: List[MBModel]
    :param nids: the indices of the neurons that we want to show their names
    :type nids: List[int]
    :param vmin: the lower bound for the colour map. Default is -2
    :type vmin: float
    :param vmax: the upper bound for the colour map. Default is 2
    :type vmin: float
    :param only_nids: when True, only the specified neurons are plotted.
    :type only_nids: bool
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

    # plt.colorbar()
    plt.tight_layout()
    plt.show()


def plot_phase_overlap_mean_responses(ms, nids=None, only_nids=True, figsize=None):
    """
    Plots the average responses of the neurons per phase/trial with overlapping lines.


    :param ms:the models where the values are taken from
    :type ms: List[MBModel]
    :param nids: the indices of the neurons that we want to show their names
    :type nids: List[int]
    :param only_nids: when True, only the specified neurons are plotted.
    :type only_nids: bool
    """
    title = "individuals-" + '-'.join(str(ms[0]).split("'")[1:-1:2])

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

        v = ms[i]._v

        # trial, odour, time-step, neuron
        va = v[1:].reshape((-1, nb_odours, nb_timesteps, v.shape[-1]))[:, 0].reshape((-1, v.shape[-1]))
        if only_nids:
            va = va[:, nids]

        x_ticks_ = xticks[1:(nb_trials // 2) // 2] * 2
        n = len(x_ticks_)
        _x_ticks = np.arange(n, dtype=float) + 2 - 1 / (nb_timesteps - 1)
        _x_ticks[n//2:] += 1. - 1 / (nb_timesteps - 1)
        # _x_ticks = _x_ticks[1:]

        x_ = np.arange(0, nb_trials // 2, 1 / (nb_timesteps - 1)) - 1 / (nb_timesteps - 1)

        for j in range(nb_neurons):

            label = None
            s = (i + 1) / (nb_models + 1)
            a_col = np.array([s * 205, s * 222, 238]) / 255.
            if j == nb_neurons - 1:
                label = ms[i].routine_name

            vaj = va[:, j].reshape((-1, nb_timesteps))[:, 1:].reshape((-1,))
            if len(subs) <= j:
                axa = plt.subplot(nb_rows, nb_cols, 2 * (j // nb_cols) * nb_cols + j % nb_cols + 1)
                # axa.plot([[15, 17, 19, 21, 23, 25]] * 2, [[0] * 6, [ylim] * 6], 'r-')
                # axa.set_xticks((nb_timesteps - 1) * np.arange(2 * (nb_trials // 4)) + (nb_timesteps - 1) / 4)
                axa.set_xticks(_x_ticks)
                axa.set_xticklabels(["" for _ in x_ticks_])
                axa.set_yticks([0, 1, 2])
                axa.set_ylim(ylim)
                axa.set_xlim([0, n + 1 + 1 / (nb_timesteps - 1)])
                axa.tick_params(labelsize=8)
                axa.set_title(r"$%s$" % names[j], fontsize=8)
                if j % nb_cols == 0:
                    axa.set_ylabel("Odour A", fontsize=8)
                else:
                    axa.set_yticklabels([""] * 3)
                    axa.spines['left'].set_visible(False)
                    axa.set_yticks([])

                # axa.yaxis.grid()
                axa.spines['top'].set_visible(False)
                axa.spines['right'].set_visible(False)

                s = (i + 2) / (nb_models + 1)
                a_acol = np.array([s * 205, s * 222, 238]) / 255.
                axa.plot(x_[:4], vaj[:4], color=(.8, .8, .8))
                axa.plot(x_[3:13], vaj[3:13], color=a_acol, label="acquisition")
                subs.append(axa)
            # y_for = va[6*nb_timesteps:, j].reshape((-1, nb_timesteps))[:, 1:].reshape((-1,))[2:-2]
            subs[j].plot(x_[12:15], vaj[12:15], color=(.8, .8, .8))
            subs[j].plot(x_[23:], vaj[23:], color=(.8, .8, .8))
            subs[j].plot(x_[14:24], vaj[14:24], color=a_col, label=label)
            if ("no shock" in ms[i].routine_name or "unpaired" in ms[i].routine_name or
                    "d" not in names[j] and "c" not in names[j] or "av" not in names[j]):
                continue
            shock_i = [15, 17, 19, 21, 23]
            subs[j].plot(x_[shock_i], vaj[shock_i], 'r.')

        # axb = plt.subplot(nb_models * 2, 2, 2 + i * 4)
        vb = v[1:].reshape((-1, nb_odours, nb_timesteps, v.shape[-1]))[:, 1].reshape((-1, v.shape[-1]))
        if only_nids:
            vb = vb[:, nids]
        x_b = x_ + 1 - 1 / (nb_timesteps - 1)
        for j in range(nb_neurons):
            jn = j + nb_neurons

            label = None
            s = (i + 1) / (nb_models + 1)
            b_col = np.array([255, s * 197, s * 200]) / 255.
            if j == nb_neurons - 1:
                label = ms[i].routine_name

            vbj = vb[:, j].reshape((-1, nb_timesteps))[:, 1:].reshape((-1,))
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
                    # else:
                    #     axb.text(-7, -.5, "Trial #", fontsize=8)
                else:
                    axb.set_yticklabels([""] * 3)
                    axb.spines['left'].set_visible(False)
                    axb.set_yticks([])

                # axb.yaxis.grid()
                axb.spines['top'].set_visible(False)
                axb.spines['right'].set_visible(False)

                s = (i + 2) / (nb_models + 1)
                b_acol = np.array([255, s * 197, s * 200]) / 255.
                axb.plot(x_b[:3], vbj[:3], color=(.8, .8, .8))
                axb.plot(x_b[2:12], vbj[2:12], color=b_acol, label="acquisition")
                subs.append(axb)

            subs[jn].plot(x_b[11:14], vbj[11:14], color=(.8, .8, .8))
            subs[jn].plot(x_b[22:], vbj[22:], color=(.8, .8, .8))
            subs[jn].plot(x_b[13:23], vbj[13:23], color=b_col, label=label)

            if i > 0 or "d" not in names[j] and "c" not in names[j] or "av" not in names[j]:
                continue
            shock_i = [3, 5, 7, 9, 11]
            subs[jn].plot(x_b[shock_i], vbj[shock_i], 'r.')

    if only_nids:
        subs[len(subs)//2 - 1].legend(fontsize=8, bbox_to_anchor=(1.05, 1.35), loc='upper left',
                                      framealpha=0., labelspacing=1.)
        subs[-1].legend(fontsize=8, bbox_to_anchor=(1.05, 1.35), loc='upper left', framealpha=0., labelspacing=1)
    plt.tight_layout()
    plt.show()


def plot_weights(ms, nids=None, only_nids=True, figsize=None):
    """
    Plots the average synaptic weights of the post-synaptic neurons per phase/trial for a specific experiment with
    overlapping phases. The average weight is calculated per odour.

    :param ms:the models where the values are taken from
    :type ms: List[MBModel]
    :param nids: the indices of the neurons that we want to show their names
    :type nids: List[int]
    :param only_nids: when True, only the specified neurons are plotted.
    :type only_nids: bool
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
            s = (i + 1) / (nb_models + 1)
            a_col = np.array([s * 205, s * 222, 238]) / 255.
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

                s = (i + 2) / (nb_models + 1)
                a_acol = np.array([s * 205, s * 222, 238]) / 255.
                axa.plot(x_[:4], waj[:4], color=(.8, .8, .8))
                axa.plot(x_[3:13], waj[3:13], color=a_acol, label="acquisition")
                subs.append(axa)
            subs[j].plot(x_[12:15], waj[12:15], color=(.8, .8, .8))
            subs[j].plot(x_[23:], waj[23:], color=(.8, .8, .8))
            subs[j].plot(x_[14:24], waj[14:24], color=a_col, label=label)

            if ("no shock" in ms[i].routine_name or "unpaired" in ms[i].routine_name or
                    "d" not in names[j] and "c" not in names[j] or "av" not in names[j]):
                continue
            shock_i = [15, 17, 19, 21, 23]
            subs[j].plot(x_[shock_i], waj[shock_i], 'r.')

        # axb = plt.subplot(nb_models * 2, 2, 2 + i * 4)
        wb = np.nanmean(w[1:, 5:], axis=1).reshape((-1, nb_odours, nb_timesteps, w.shape[-1]))[:, 1].reshape((-1, w.shape[-1]))
        if only_nids:
            wb = wb[:, nids]
        x_b = x_ + 1 - 1 / (nb_timesteps - 1)
        for j in range(nb_neurons):
            jn = j + nb_neurons

            label = None
            s = (i + 1) / (nb_models + 1)
            b_col = np.array([255, s * 197, s * 200]) / 255.
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

                s = (i + 2) / (nb_models + 1)
                b_acol = np.array([255, s * 197, s * 200]) / 255.
                axb.plot(x_b[:3], wbj[:3], color=(.8, .8, .8))
                axb.plot(x_b[2:12], wbj[2:12], color=b_acol, label="acquisition")
                subs.append(axb)

            subs[jn].plot(x_b[11:14], wbj[11:14], color=(.8, .8, .8))
            subs[jn].plot(x_b[22:], wbj[22:], color=(.8, .8, .8))
            subs[jn].plot(x_b[13:23], wbj[13:23], color=b_col, label=label)

            if i > 0 or "d" not in names[j] and "c" not in names[j] or "av" not in names[j]:
                continue
            shock_i = [3, 5, 7, 9, 11]
            subs[jn].plot(x_b[shock_i], wbj[shock_i], 'r.')

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

    :param m: the model where the values are taken from
    :type m: MBModel
    :param nids: the indices of the neurons that we want to show their names
    :type nids: List[int]
    :param vmin: the lower bound of the colour map
    :type vmin: float
    :param vmax: the upper bound of the colour map
    :type vmax: float
    :param only_nids: when True, only the specified neurons are plotted.
    :type only_nids: bool
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

    :param w: the weights matrix where the synaptic weights are drawn from
    :type w: np.ndarray
    :param names_in: the names of the pre-synaptic neurons
    :type names_in: List[str]
    :param names_out: the names of the post-synaptic neurons
    :type names_out: List[str]
    :param ax: (optional) a matplotlib axis to draw the plot in
    :param cmap: the colour map name to use for  specifying the direction and amplitude of synaptic weight
    :type cmap: str
    :param vmin: the lower bound of the colour map
    :type vmin: float
    :param vmax: the upper bound of the colour map
    :type vmax: float
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

    :param m: the model to get the responses from
    :type m: MBModel
    :param nids: (optional) the 2 neuron indices which we want to plot their responses. Default is 1 and 6.
    :type nids: List[int]
    """
    if nids is None:
        nids = [1, 6]
    _plot_subcircuit(m, nids, nnames=[r"$d_{av}$", r"$s_{at}$"], title="SM",
                     ncolours=["#db006aff", "#6adbb8ff"], uss=["r", None], sub=sub)


def plot_rm(m, nids=None, sub=None):
    """
    The responses of the restrained memory (RM) sub-circuit.

    :param m: the model to get the responses from
    :type m: MBModel
    :param nids: (optional) the 2 neuron indices which we want to plot their responses. Default is 6 and 9.
    :type nids: List[int]
    """
    if nids is None:
        nids = [6, 9]
    _plot_subcircuit(m, nids, nnames=[r"$s_{at}$", r"$r_{av}$"], title="RM",
                     ncolours=["#6adbb8ff", "#db6a6aff"], sub=sub)


def plot_ltm(m, nids=None, sub=None):
    """
    The responses of the long-term memory (LTM) sub-circuit.

    :param m: the model to get the responses from
    :type m: MBModel
    :param nids: (optional) the 2 neuron indices which we want to plot their responses. Default is 2 and 10.
    :type nids: List[int]
    """
    if nids is None:
        nids = [2, 10]
    _plot_subcircuit(m, nids, nnames=[r"$c_{at}$", r"$m_{at}$"], title="LTM",
                     ncolours=["#6adb00ff", "#6adbb8ff"], uss=["g", None], sub=sub)


def plot_rrm(m, nids=None, sub=None):
    """
    The responses of the reciprocal restrained memories (RRM) sub-circuit.

    :param m: the model to get the responses from
    :type m: MBModel
    :param nids: (optional) the 4 neuron indices which we want to plot their responses. Default is 3, 8, 2 and 9.
    :type nids: List[int]
    """
    if nids is None:
        nids = [3, 8, 2, 9]
    _plot_subcircuit(m, nids, title="RRM", uss=["r", None, None, None],
                     nnames=[r"$c_{av}$", r"$r_{at}$", r"$c_{at}$", r"$r_{av}$"],
                     ncolours=["#db006aff", "#6adbb8ff", "#6adb00ff", "#db6a6aff"], sub=sub)


def plot_rfm(m, nids=None, sub=None):
    """
    The responses of the reciprocal forgetting memories (RFM) sub-circuit.

    :param m: the model to get the responses from
    :type m: MBModel
    :param nids: (optional) the 4 neuron indices which we want to plot their responses. Default is 5, 10, 4 and 11.
    :type nids: List[int]
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

    :param m: the model to get the responses from
    :type m: MBModel
    :param nids: (optional) the 4 neuron indices which we want to plot their responses. Default is 4, 8, 2 and 10.
    :type nids: List[int]
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

    :param m: the model where the values are taken from
    :type m: MBModel
    :param nids: the indices of the neurons that we want to plot
    :type nids: List[int]
    :param nnames: a list with the names of the neurons
    :type nnames: List[str]
    :param ncolours: colour for the line representing the neuron with the same index
    :type ncolours: List[str]
    :param uss: (optional) a list of colour-names for the US of the same size as the nids;'r' for red (punishment), or
    'g' for green (reward), None for no reinforcement. Default is None.
    :type uss: List[str]
    :param title: (optional) the title of the figure. If more than 1 plots are to be generated, a number is added at the
    end of the title. Default is 'sub-circuit'
    :type title: str
    :return:
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


def plot_arena_paths(data, name="arena", lw=1., alpha=.2, ax=None, save=False, show=True, figsize=None):
    if ax is None:
        if figsize is None:
            figsize = (2, 2)
        plt.figure(name, figsize=figsize)
        ax = plt.subplot(111, polar=True)

    # ax.set_theta_offset(np.pi)
    ax.set_theta_zero_location("W")

    draw_gradients(ax, radius=1.)

    nb_flies, nb_steps = data.shape
    e_pre = int(.2 * nb_steps)
    s_post = int(.5 * nb_steps)
    for i in range(nb_flies):
        ax.plot(np.angle(data[i, :e_pre]), np.absolute(data[i, :e_pre]), color='b', alpha=alpha, lw=lw)
        ax.plot(np.angle(data[i, e_pre:s_post]), np.absolute(data[i, e_pre:s_post]),
                color='r' if 'quinine' in name else 'g', alpha=alpha, lw=lw)
        ax.plot(np.angle(data[i, s_post:]), np.absolute(data[i, s_post:]), color='k', alpha=alpha, lw=lw)

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylim([0, 1])

    if save:
        plt.savefig(name + ".svg", dpi=600)
    if show:
        plt.show()


def plot_arena_stats(df, name="arena-stats", figsize=None):

    mechanisms = ["susceptible", "reciprocal", "long-term memory", ["susceptible", "reciprocal", "long-term memory"]]
    reinforcements = ["punishment", "reward"]
    odours = ["A", "B", "AB"]
    ms, rs, os = np.meshgrid(mechanisms, reinforcements, odours)

    if figsize is None:
        figsize = (5, 4)
    plt.figure(name, figsize=figsize)
    for mechanism, reinforcement, odour in zip(ms.flatten(), rs.flatten(), os.flatten()):
        m = mechanisms.index(mechanism)
        r = reinforcements.index(reinforcement)
        o = odours.index(odour)
        ax = plt.subplot(4, 6, m * 6 + r * 3 + o + 1, polar=True)
        ax.set_theta_zero_location("W")
        _plot_arena_stats(df, mechanisms=mechanism, reinforcements=reinforcement, odours=odour, ax=ax,
                          name="%s-%s-%s" % ("".join([m[0] for m in mechanism]) if isinstance(mechanism, list)
                                             else mechanism[0], reinforcement[0], odour.lower()))
    plt.tight_layout()
    plt.show()


def _plot_arena_stats(data, mechanisms=None, reinforcements=None, odours=None, name="arena-stats",
                      bimodal_tol=np.pi/2, print_stats=False, ax=None):

    if ax is None:
        plt.figure(name, figsize=(2, 2))
        ax = plt.subplot(111, polar=True)
    else:
        ax.set_title(name, fontsize=8)
    if mechanisms is None:
        mechanisms = ["susceptible", "reciprocal", "long-term memory"]
    if not isinstance(mechanisms, list):
        mechanisms = [mechanisms]
    if reinforcements is None:
        reinforcements = ["reward"]
    if not isinstance(reinforcements, list):
        reinforcements = [reinforcements]
    if odours is None:
        odours = ["B"]
    if not isinstance(odours, list):
        odours = [odours]

    mechanisms_not = list({"susceptible", "reciprocal", "long-term memory"} - set(mechanisms))

    df = data[np.all([data[m] for m in mechanisms] + [~data[m] for m in mechanisms_not], axis=0)]
    df = df[np.any([df["reinforcement"] == r for r in reinforcements], axis=0)]
    df = df[np.any([df["paired odour"] == o for o in odours], axis=0)]

    d_pre = np.deg2rad(df[df["phase"] == "pre"]["angle"])
    d_learn = np.deg2rad(df[df["phase"] == "learn"]["angle"])
    d_post = np.deg2rad(df[df["phase"] == "post"]["angle"])

    if print_stats:
        from scipy.stats import circmean, circstd

        n_pre = len(d_pre)
        d_pre_mean = circmean(d_pre)
        d_pre_std = circstd(d_pre)
        print("pre:", np.rad2deg(d_pre_mean), np.rad2deg(d_pre_std), n_pre)

        if d_pre_std > np.pi/2:
            d_pre_000 = d_pre[np.all([-bimodal_tol/2 <= ((d_pre + np.pi) % (2 * np.pi) - np.pi),
                                      ((d_pre + np.pi) % (2 * np.pi) - np.pi) < bimodal_tol/2], axis=0)]
            d_pre_090 = d_pre[np.all([-bimodal_tol/2 <= ((d_pre - np.pi/2 + np.pi) % (2 * np.pi) - np.pi),
                                     ((d_pre - np.pi/2 + np.pi) % (2 * np.pi) - np.pi) < bimodal_tol/2], axis=0)]
            d_pre_180 = d_pre[np.all([-bimodal_tol/2 <= ((d_pre - np.pi + np.pi) % (2 * np.pi) - np.pi),
                                     ((d_pre - np.pi + np.pi) % (2 * np.pi) - np.pi) < bimodal_tol/2], axis=0)]
            d_pre_270 = d_pre[np.all([-bimodal_tol/2 <= ((d_pre + np.pi/2 + np.pi) % (2 * np.pi) - np.pi),
                                     ((d_pre + np.pi/2 + np.pi) % (2 * np.pi) - np.pi) < bimodal_tol/2], axis=0)]

            n_pre_000 = len(d_pre_000)
            d_pre_000_mean = circmean(d_pre_000)
            d_pre_000_std = circstd(d_pre_000)
            print("pre   0:", np.rad2deg(d_pre_000_mean), np.rad2deg(d_pre_000_std), n_pre_000)

            n_pre_090 = len(d_pre_090)
            d_pre_090_mean = circmean(d_pre_090)
            d_pre_090_std = circstd(d_pre_090)
            print("pre  90:", np.rad2deg(d_pre_090_mean), np.rad2deg(d_pre_090_std), n_pre_090)

            n_pre_180 = len(d_pre_180)
            d_pre_180_mean = circmean(d_pre_180)
            d_pre_180_std = circstd(d_pre_180)
            print("pre 180:", np.rad2deg(d_pre_180_mean), np.rad2deg(d_pre_180_std), n_pre_180)

            n_pre_270 = len(d_pre_270)
            d_pre_270_mean = circmean(d_pre_270)
            d_pre_270_std = circstd(d_pre_270)
            print("pre 270:", np.rad2deg(d_pre_270_mean), np.rad2deg(d_pre_270_std), n_pre_270)

        n_learn = len(d_learn)
        d_learn_mean = circmean(d_learn, high=np.pi, low=-np.pi)
        d_learn_std = circstd(d_learn)
        print("learn:", np.rad2deg(d_learn_mean), np.rad2deg(d_learn_std), n_learn)

        if d_learn_std > np.pi/2:
            d_learn_000 = d_learn[np.all([-bimodal_tol/2 <= ((d_learn + np.pi) % (2 * np.pi) - np.pi),
                                         ((d_learn + np.pi) % (2 * np.pi) - np.pi) < bimodal_tol/2], axis=0)]
            d_learn_090 = d_learn[np.all([-bimodal_tol/2 <= ((d_learn - np.pi/2 + np.pi) % (2 * np.pi) - np.pi),
                                         ((d_learn - np.pi/2 + np.pi) % (2 * np.pi) - np.pi) < bimodal_tol/2], axis=0)]
            d_learn_180 = d_learn[np.all([-bimodal_tol/2 <= ((d_learn - np.pi + np.pi) % (2 * np.pi) - np.pi),
                                         ((d_learn - np.pi + np.pi) % (2 * np.pi) - np.pi) < bimodal_tol/2], axis=0)]
            d_learn_270 = d_learn[np.all([-bimodal_tol/2 <= ((d_learn + np.pi/2 + np.pi) % (2 * np.pi) - np.pi),
                                         ((d_learn + np.pi/2 + np.pi) % (2 * np.pi) - np.pi) < bimodal_tol/2], axis=0)]

            n_learn_000 = len(d_learn_000)
            d_learn_000_mean = circmean(d_learn_000)
            d_learn_000_std = circstd(d_learn_000)
            print("learn   0:", np.rad2deg(d_learn_000_mean), np.rad2deg(d_learn_000_std), n_learn_000)

            n_learn_090 = len(d_learn_090)
            d_learn_090_mean = circmean(d_learn_090)
            d_learn_090_std = circstd(d_learn_090)
            print("learn  90:", np.rad2deg(d_learn_090_mean), np.rad2deg(d_learn_090_std), n_learn_090)

            n_learn_180 = len(d_learn_180)
            d_learn_180_mean = circmean(d_learn_180)
            d_learn_180_std = circstd(d_learn_180)
            print("learn 180:", np.rad2deg(d_learn_180_mean), np.rad2deg(d_learn_180_std), n_learn_180)

            n_learn_270 = len(d_learn_270)
            d_learn_270_mean = circmean(d_learn_270)
            d_learn_270_std = circstd(d_learn_270)
            print("learn 270:", np.rad2deg(d_learn_270_mean), np.rad2deg(d_learn_270_std), n_learn_270)

        n_post = len(d_post)
        d_post_mean = circmean(d_post, high=np.pi, low=-np.pi)
        d_post_std = circstd(d_post)
        print("post:", np.rad2deg(d_post_mean), np.rad2deg(d_post_std), n_post)

        if d_post_std > np.pi/2:
            d_post_000 = d_post[np.all([-bimodal_tol/2 <= ((d_post + np.pi) % (2 * np.pi) - np.pi),
                                       ((d_post + np.pi) % (2 * np.pi) - np.pi) < bimodal_tol/2], axis=0)]
            d_post_090 = d_post[np.all([-bimodal_tol/2 <= ((d_post - np.pi/2 + np.pi) % (2 * np.pi) - np.pi),
                                       ((d_post - np.pi/2 + np.pi) % (2 * np.pi) - np.pi) < bimodal_tol/2], axis=0)]
            d_post_180 = d_post[np.all([-bimodal_tol/2 <= ((d_post - np.pi + np.pi) % (2 * np.pi) - np.pi),
                                       ((d_post - np.pi + np.pi) % (2 * np.pi) - np.pi) < bimodal_tol/2], axis=0)]
            d_post_270 = d_post[np.all([-bimodal_tol/2 <= ((d_post + np.pi/2 + np.pi) % (2 * np.pi) - np.pi),
                                       ((d_post + np.pi/2 + np.pi) % (2 * np.pi) - np.pi) < bimodal_tol/2], axis=0)]

            n_post_000 = len(d_post_000)
            d_post_000_mean = circmean(d_post_000)
            d_post_000_std = circstd(d_post_000)
            print("post   0:", np.rad2deg(d_post_000_mean), np.rad2deg(d_post_000_std), n_post_000)

            n_post_090 = len(d_post_090)
            d_post_090_mean = circmean(d_post_090)
            d_post_090_std = circstd(d_post_090)
            print("post  90:", np.rad2deg(d_post_090_mean), np.rad2deg(d_post_090_std), n_post_090)

            n_post_180 = len(d_post_180)
            d_post_180_mean = circmean(d_post_180)
            d_post_180_std = circstd(d_post_180)
            print("post 180:", np.rad2deg(d_post_180_mean), np.rad2deg(d_post_180_std), n_post_180)

            n_post_270 = len(d_post_270)
            d_post_270_mean = circmean(d_post_270)
            d_post_270_std = circstd(d_post_270)
            print("post 270:", np.rad2deg(d_post_270_mean), np.rad2deg(d_post_270_std), n_post_270)

    draw_gradients(ax, radius=1.)

    kappa = 10
    # density_pre = gaussian_kde(d_pre, bw_method=partial(my_kde_bandwidth, fac=.5))
    x, density_pre = vonmises_fft_kde(d_pre, kappa=kappa, n_bins=36)
    x, density_learn = vonmises_fft_kde(d_learn, kappa=kappa, n_bins=36)
    x, density_post = vonmises_fft_kde(d_post, kappa=kappa, n_bins=36)

    rein_c = 'r' if "punishment" in reinforcements else "g"
    ax.fill_between(x, density_pre * 0., density_pre, facecolor='b', alpha=.2)
    ax.plot(x, density_pre, 'b', lw=.5)
    ax.fill_between(x, density_learn * 0., density_learn, facecolor=rein_c, alpha=.2)
    ax.plot(x, density_learn, rein_c, lw=.5)
    ax.fill_between(x, density_post * 0., density_post, facecolor='k', alpha=.2)
    ax.plot(x, density_post, 'k', lw=.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim([0, 1])


def plot_arena_box(df, name="arena-box"):

    mechanisms = [["susceptible"], ["reciprocal"], ["long-term memory"],
                  ["susceptible", "reciprocal", "long-term memory"]]
    reinforcements = ["punishment", "reward"]
    odours = ["A", "B", "AB"]
    ms, rs, os = np.meshgrid(mechanisms, reinforcements, odours)

    labels, data = [""] * 24, [[]] * 24
    for mechanism, reinforcement, odour in zip(ms.flatten(), rs.flatten(), os.flatten()):

        mechanisms_not = list({"susceptible", "reciprocal", "long-term memory"} - set(mechanism))
        dff = df[np.all([df[m] for m in mechanism] + [~df[m] for m in mechanisms_not], axis=0)]
        dff = dff[np.any([dff["reinforcement"] == reinforcement], axis=0)]
        dff = dff[np.any([dff["paired odour"] == odour], axis=0)]

        if reinforcement == "punishment":
            if odour == "A":
                d_pre = np.array(dff[dff["phase"] == "pre"]["avoid A"])
                d_learn = np.array(dff[dff["phase"] == "learn"]["avoid A"])
                d_post = np.array(dff[dff["phase"] == "post"]["avoid A"])
            elif odour == "B":
                d_pre = np.array(dff[dff["phase"] == "pre"]["avoid B"])
                d_learn = np.array(dff[dff["phase"] == "learn"]["avoid B"])
                d_post = np.array(dff[dff["phase"] == "post"]["avoid B"])
            else:
                d_pre = np.array(dff[dff["phase"] == "pre"]["avoid A/B"])
                d_learn = np.array(dff[dff["phase"] == "learn"]["avoid A/B"])
                d_post = np.array(dff[dff["phase"] == "post"]["avoid A/B"])
        else:
            if odour == "A":
                d_pre = np.array(dff[dff["phase"] == "pre"]["attract A"])
                d_learn = np.array(dff[dff["phase"] == "learn"]["attract A"])
                d_post = np.array(dff[dff["phase"] == "post"]["attract A"])
            elif odour == "B":
                d_pre = np.array(dff[dff["phase"] == "pre"]["attract B"])
                d_learn = np.array(dff[dff["phase"] == "learn"]["attract B"])
                d_post = np.array(dff[dff["phase"] == "post"]["attract B"])
            else:
                d_pre = np.array(dff[dff["phase"] == "pre"]["attract A/B"])
                d_learn = np.array(dff[dff["phase"] == "learn"]["attract A/B"])
                d_post = np.array(dff[dff["phase"] == "post"]["attract A/B"])

        # d_pre = np.array(dff[dff["phase"] == "pre"]["pref A"] - dff[dff["phase"] == "pre"]["pref B"])
        # d_learn = np.array(dff[dff["phase"] == "learn"]["pref A"] - dff[dff["phase"] == "learn"]["pref B"])
        # d_post = np.array(dff[dff["phase"] == "post"]["pref A"] - dff[dff["phase"] == "post"]["pref B"])

        i_mecha = mechanisms.index(mechanism)
        i_reinf = reinforcements.index(reinforcement)
        i_odour = odours.index(odour)
        i = i_mecha * 6 + i_reinf * 3 + i_odour
        label = "%s%s-%s" % (reinforcement[0],
                             "".join([m[0] for m in mechanism]) if isinstance(mechanism, list) else mechanism[0],
                             odour.lower())

        labels[i] = label
        data[i] = [d_pre, d_learn, d_post]

    plt.figure(name, figsize=(6, 4))
    for i in range(len(labels)):
        plt.subplot(4, 6, i + 1)
        plt.boxplot(np.array(data[i][0]), positions=[1], notch=True, patch_artist=True,
                    boxprops=dict(color="b", facecolor="b"),
                    whiskerprops=dict(color="b"),
                    flierprops=dict(color="b", markeredgecolor="b", marker="."),
                    capprops=dict(color="b"),
                    medianprops=dict(color="b"))
        plt.boxplot(np.array(data[i][1]), positions=[2], notch=True, patch_artist=True,
                    boxprops=dict(color="r", facecolor="r"),
                    whiskerprops=dict(color="r"),
                    flierprops=dict(color="r", markeredgecolor="r", marker="."),
                    capprops=dict(color="r"),
                    medianprops=dict(color="r"))
        plt.boxplot(np.array(data[i][2]), positions=[3], notch=True, patch_artist=True,
                    boxprops=dict(color="k", facecolor="k"),
                    whiskerprops=dict(color="k"),
                    flierprops=dict(color="k", markeredgecolor="k", marker="."),
                    capprops=dict(color="k"),
                    medianprops=dict(color="k"))
        if i < 6:
            title = labels[i][3:].upper() + "+" + labels[i][0]
            title = title.replace("p", "shock")
            title = title.replace("r", "sugar")
            title = title.replace("AB", "A/B")
            plt.title(title)
        if i >= 18:
            plt.xticks([1, 2, 3], ["pre", "train", "post"])
        else:
            plt.xticks([1, 2, 3], [""] * 3)

        if i % 6 == 0:
            label = labels[i][1:-2].replace("l", "m").upper()
            if len(label) > 1:
                label = label[0] + "+" + label[1] + "+" + label[2]
            plt.ylabel(label)
            plt.yticks([-1, 0, 1], ["av", "0", "at"])
        else:
            plt.yticks([-1, 0, 1], [""] * 3)
        plt.ylim([-1.05, 1.05])
    plt.tight_layout()
    plt.show()


def _get_bimodal_mean(data, bimodal_tol=np.pi, verbose=False):
    n = [len(data)]
    d_mean = [circmean(data)]
    d_std = [circstd(data)]

    if d_std[0] > np.pi / 3:
        d_000 = data[np.all([-bimodal_tol / 2 <= ((data + np.pi) % (2 * np.pi) - np.pi),
                             ((data + np.pi) % (2 * np.pi) - np.pi) < bimodal_tol / 2], axis=0)]
        d_090 = data[np.all([-bimodal_tol / 2 <= ((data - np.pi / 2 + np.pi) % (2 * np.pi) - np.pi),
                             ((data - np.pi / 2 + np.pi) % (2 * np.pi) - np.pi) < bimodal_tol / 2], axis=0)]
        d_180 = data[np.all([-bimodal_tol / 2 <= ((data - np.pi + np.pi) % (2 * np.pi) - np.pi),
                             ((data - np.pi + np.pi) % (2 * np.pi) - np.pi) < bimodal_tol / 2], axis=0)]
        d_270 = data[np.all([-bimodal_tol / 2 <= ((data + np.pi / 2 + np.pi) % (2 * np.pi) - np.pi),
                             ((data + np.pi / 2 + np.pi) % (2 * np.pi) - np.pi) < bimodal_tol / 2], axis=0)]

        n_000 = len(d_000)
        d_000_mean = circmean(d_000)
        d_000_std = circstd(d_000)
        if verbose:
            print("data   0:", np.rad2deg(d_000_mean), np.rad2deg(d_000_std), n_000)

        n_090 = len(d_090)
        d_090_mean = circmean(d_090)
        d_090_std = circstd(d_090)
        if verbose:
            print("data  90:", np.rad2deg(d_090_mean), np.rad2deg(d_090_std), n_090)

        n_180 = len(d_180)
        d_180_mean = circmean(d_180)
        d_180_std = circstd(d_180)
        if verbose:
            print("data 180:", np.rad2deg(d_180_mean), np.rad2deg(d_180_std), n_180)

        n_270 = len(d_270)
        d_270_mean = circmean(d_270)
        d_270_std = circstd(d_270)
        if verbose:
            print("data 270:", np.rad2deg(d_270_mean), np.rad2deg(d_270_std), n_270)

        if d_000_std < np.pi / 6 and d_180_std < np.pi / 6:
            # bimodal on horizontal axis
            d_mean = [d_000_mean, d_180_mean]
            d_std = [d_000_std, d_180_std]
            n = [n_000, n_180]

        elif d_090_std < np.pi / 6 and d_270_std < np.pi / 6:
            # bimodal on vertical axis
            d_mean = [d_090_mean, d_270_mean]
            d_std = [d_090_std, d_270_std]
            n = [n_090, n_270]

    return d_mean, d_std, n


def draw_gradients(ax, radius=1., draw_sources=True, cmap="coolwarm", levels=20, vminmax=3):
    from arena import FruitFly, gaussian_p

    a_mean, a_sigma = FruitFly.a_source, FruitFly.a_sigma
    b_mean, b_sigma = FruitFly.b_source, FruitFly.b_sigma
    x, y = np.meshgrid(np.linspace(-radius, radius, int(radius * 100) + 1),
                       np.linspace(-radius, radius, int(radius * 100) + 1))
    p = (x + y * 1j).T

    p_a = gaussian_p(p, a_mean, a_sigma)
    p_b = gaussian_p(p, b_mean, b_sigma)

    rho, dist = np.angle(p), np.absolute(p)
    ax.contourf(rho-np.pi/2, dist, p_b - p_a, cmap=cmap, levels=levels, vmin=-vminmax, vmax=vminmax)
    ax.contour(rho-np.pi/2, dist, p_b - p_a, levels=[-.0001, .0001], colors='lightsteelblue', linestyles='--')

    if draw_sources:
        ax.scatter(np.angle(a_mean), np.absolute(a_mean), s=20, color="C0", label="odour A")
        ax.scatter(np.angle(b_mean), np.absolute(b_mean), s=20, color="C1", label="odour B")

    return ax


def vonmises_pdf(x, mu, kappa):
    from scipy.special import i0

    return np.exp(kappa * np.cos(x - mu)) / (2. * np.pi * i0(kappa))


def vonmises_fft_kde(data, kappa, n_bins):
    bins = np.linspace(-np.pi, np.pi, n_bins + 1, endpoint=True)
    hist_n, bin_edges = np.histogram(data, bins=bins)
    bin_centers = np.mean([bin_edges[1:], bin_edges[:-1]], axis=0)
    kernel = vonmises_pdf(
        x=bin_centers,
        mu=0,
        kappa=kappa
    )
    kde = np.fft.fftshift(np.fft.irfft(np.fft.rfft(kernel) * np.fft.rfft(hist_n)))
    kde /= np.trapz(kde, x=bin_centers)
    bin_centers = np.r_[bin_centers, bin_centers[0]]
    kde = np.r_[kde, kde[0]]
    return bin_centers, kde