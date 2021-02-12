from models_base import MBModel

from typing import List
from matplotlib import cm
from matplotlib.colors import Normalize

import matplotlib.pyplot as plt
import numpy as np


def plot_population(ms, nids=None, vmin=-2., vmax=2., only_nids=False):
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

    plt.figure(title, figsize=(7.5, 10))

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


def plot_weights_matrices(ms, nids=None, vmin=-2., vmax=2., only_nids=False):
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

    plt.figure(title, figsize=(7.5, 5))

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


def plot_phase_overlap_mean_responses(ms, nids=None, only_nids=True):
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
        nb_rows = 2
        nb_cols = 12
        figsize = (8, 2)
    else:
        nb_rows = 8
        nb_cols = 8
        figsize = (5, 7)
        # figsize = (5, 5)
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

        for j in range(nb_neurons):

            label = None
            s = (i + 1) / (nb_models + 1)
            colour = np.array([s * 205, s * 222, 238]) / 255.
            if j == nb_neurons - 1:
                label = ms[i].routine_name

            if len(subs) <= j:
                axa = plt.subplot(nb_rows, nb_cols, j+1)
                # axa.plot([[15, 17, 19, 21, 23, 25]] * 2, [[0] * 6, [ylim] * 6], 'r-')
                axa.set_xticks((nb_timesteps - 1) * np.arange(nb_trials // 4) + (nb_timesteps - 1) / 4)
                axa.set_xticklabels(xticks[:(nb_trials // 4)])
                axa.set_yticks([0, 1, 2])
                axa.set_ylim(ylim)
                axa.set_xlim([1, 6 * (nb_timesteps - 1)])
                axa.tick_params(labelsize=8)
                axa.set_title(r"$%s$" % names[j], fontsize=8)
                if j % nb_cols == 0:
                    axa.set_ylabel("Odour A", fontsize=8)
                    if only_nids:
                        axa.text(-7, -.8, "Trial #", fontsize=8)
                    # else:
                    #     axa.text(-7, -.5, "Trial #", fontsize=8)
                else:
                    axa.set_yticklabels([""] * 3)
                # axa.yaxis.grid()
                axa.spines['top'].set_visible(False)
                axa.spines['right'].set_visible(False)

                s = (i + 2) / (nb_models + 1)
                acolour = np.array([s * 205, s * 222, 238]) / 255.
                y_acq = va[:6*nb_timesteps, j].reshape((-1, nb_timesteps))[:, 1:].reshape((-1,))
                axa.plot(y_acq, color=acolour, label="acquisition")
                subs.append(axa)
            y_for = va[6*nb_timesteps:, j].reshape((-1, nb_timesteps))[:, 1:].reshape((-1,))
            subs[j].plot(y_for, color=colour, label=label)
            if "no shock" not in ms[i].routine_name and "unpaired" not in ms[i].routine_name:
                y_shock = va[np.array([8, 9, 10, 11, 12]) * nb_timesteps - 1, j]
                x_shock = np.array([2, 3, 4, 5, 6]) * (nb_timesteps - 1) - 1
                subs[j].plot(x_shock, y_shock, 'r.')

        # axb = plt.subplot(nb_models * 2, 2, 2 + i * 4)
        vb = v[1:].reshape((-1, nb_odours, nb_timesteps, v.shape[-1]))[:, 1].reshape((-1, v.shape[-1]))
        if only_nids:
            vb = vb[:, nids]

        for j in range(nb_neurons):
            jn = j + nb_neurons

            label = None
            s = (i + 1) / (nb_models + 1)
            colour = np.array([255, s * 197, s * 200]) / 255.
            if j == nb_neurons - 1:
                label = ms[i].routine_name

            if len(subs) <= jn:
                axb = plt.subplot(nb_rows, nb_cols, j + (nb_rows * nb_cols) // 2 + 1)
                axb.set_xticks((nb_timesteps - 1) * np.arange(nb_trials // 4) + (nb_timesteps - 1) / 4)
                axb.set_xticklabels(xticks[:(nb_trials // 4)])
                axb.set_yticks([0, 1, 2])
                axb.set_ylim(ylim)
                axb.set_xlim([1, 6 * (nb_timesteps - 1)])
                axb.tick_params(labelsize=8)
                if j % nb_cols == 0:
                    axb.set_ylabel("Odour B", fontsize=8)
                    if only_nids:
                        axb.text(-7, -.8, "Trial #", fontsize=8)
                    # else:
                    #     axb.text(-7, -.5, "Trial #", fontsize=8)
                else:
                    axb.set_yticklabels([""] * 3)
                # axb.yaxis.grid()
                axb.spines['top'].set_visible(False)
                axb.spines['right'].set_visible(False)

                s = (i + 2) / (nb_models + 1)
                acolour = np.array([255, s * 197, s * 200]) / 255.
                y_acq = vb[:6*nb_timesteps, j].reshape((-1, nb_timesteps))[:, 1:].reshape((-1,))
                axb.plot(y_acq, color=acolour, label="acquisition")
                y_shock = vb[np.array([2, 3, 4, 5, 6]) * nb_timesteps - 1, j]
                x_shock = np.array([2, 3, 4, 5, 6]) * (nb_timesteps - 1) - 1
                axb.plot(x_shock, y_shock, 'r.')
                subs.append(axb)
            y_for = vb[6*nb_timesteps:, j].reshape((-1, nb_timesteps))[:, 1:].reshape((-1,))
            subs[jn].plot(y_for, color=colour, label=label)

    if only_nids:
        subs[len(subs)//2 - 1].legend(fontsize=8, bbox_to_anchor=(1.05, 1.35), loc='upper left',
                                      framealpha=0., labelspacing=1.)
        subs[-1].legend(fontsize=8, bbox_to_anchor=(1.05, 1.35), loc='upper left', framealpha=0., labelspacing=1)
    plt.tight_layout()
    plt.show()


def plot_weights(ms, nids=None, only_nids=True):
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

    plt.figure(title, figsize=(8, 2))

    subs = []
    for i in range(nb_models-1, -1, -1):
        nb_timesteps = ms[i].nb_timesteps
        nb_trials = ms[i].nb_trials

        if nids is None:
            if ms[i].neuron_ids is None:
                nids = np.arange(ms[i].nb_dan + ms[i].nb_mbon)[::8]
            else:
                nids = ms[i].neuron_ids
        ylim = [-0.1, 2.1]

        w = ms[i].w_k2m

        # trial, odour, time-step, neuron
        wa = np.nanmean(w[1:, :5], axis=1).reshape((-1, nb_odours, nb_timesteps, w.shape[-1]))[:, 0].reshape((-1, w.shape[-1]))
        if only_nids:
            wa = wa[:, nids]

        names = np.array(ms[i].names)[nids]
        nb_neurons = wa.shape[1]
        nb_plots = 2 * nb_neurons
        for j in range(nb_neurons):

            label = None
            s = (i + 1) / (nb_models + 1)
            colour = np.array([s * 205, s * 222, 238]) / 255.
            if j == nb_neurons - 1:
                label = ms[i].routine_name

            if len(subs) <= j:
                axa = plt.subplot(2, nb_plots // 2, j+1)
                # axa.plot([[15, 17, 19, 21, 23, 25]] * 2, [[0] * 6, [ylim] * 6], 'r-')
                axa.set_xticks((nb_timesteps - 1) * np.arange(nb_trials // 4) + (nb_timesteps - 1) / 4)
                axa.set_xticklabels(xticks[:(nb_trials // 4)])
                axa.set_yticks([0, 1, 2])
                axa.set_ylim(ylim)
                axa.set_xlim([1, 6 * (nb_timesteps - 1)])
                axa.tick_params(labelsize=8)
                axa.set_title(r"$%s$" % names[j], fontsize=8)
                if j == 0:
                    axa.set_ylabel("Odour A", fontsize=8)
                    axa.text(-7, -.8, "Trial #", fontsize=8)
                else:
                    axa.set_yticklabels([""] * 3)
                # axa.yaxis.grid()
                axa.spines['top'].set_visible(False)
                axa.spines['right'].set_visible(False)

                s = (i + 2) / (nb_models + 1)
                acolour = np.array([s * 205, s * 222, 238]) / 255.
                y_acq = wa[:6*nb_timesteps, j].reshape((-1, nb_timesteps))[:, 1:].reshape((-1,))
                axa.plot(y_acq, color=acolour, label="acquisition")
                subs.append(axa)
            y_for = wa[6*nb_timesteps:, j].reshape((-1, nb_timesteps))[:, 1:].reshape((-1,))
            subs[j].plot(y_for, color=colour, label=label)
            if "no shock" not in ms[i].routine_name and "unpaired" not in ms[i].routine_name:
                y_shock = wa[np.array([8, 9, 10, 11, 12]) * nb_timesteps - 1, j]
                x_shock = np.array([2, 3, 4, 5, 6]) * (nb_timesteps - 1) - 1
                subs[j].plot(x_shock, y_shock, 'r.')

        # axb = plt.subplot(nb_models * 2, 2, 2 + i * 4)
        wb = np.nanmean(w[1:, 5:], axis=1).reshape((-1, nb_odours, nb_timesteps, w.shape[-1]))[:, 1].reshape((-1, w.shape[-1]))
        if only_nids:
            wb = wb[:, nids]

        for j in range(nb_neurons):
            jn = j + nb_neurons

            label = None
            s = (i + 1) / (nb_models + 1)
            colour = np.array([255, s * 197, s * 200]) / 255.
            if j == nb_neurons - 1:
                label = ms[i].routine_name

            if len(subs) <= jn:
                axb = plt.subplot(2, nb_plots // 2, jn+1)
                axb.set_xticks((nb_timesteps - 1) * np.arange(nb_trials // 4) + (nb_timesteps - 1) / 4)
                axb.set_xticklabels(xticks[:(nb_trials // 4)])
                axb.set_yticks([0, 1, 2])
                axb.set_ylim(ylim)
                axb.set_xlim([1, 6 * (nb_timesteps - 1)])
                axb.tick_params(labelsize=8)
                if j == 0:
                    axb.set_ylabel("Odour B", fontsize=8)
                    axb.text(-7, -.8, "Trial #", fontsize=8)
                else:
                    axb.set_yticklabels([""] * 3)
                # axb.yaxis.grid()
                axb.spines['top'].set_visible(False)
                axb.spines['right'].set_visible(False)

                s = (i + 2) / (nb_models + 1)
                acolour = np.array([255, s * 197, s * 200]) / 255.
                y_acq = wb[:6*nb_timesteps, j].reshape((-1, nb_timesteps))[:, 1:].reshape((-1,))
                axb.plot(y_acq, color=acolour, label="acquisition")
                y_shock = wb[np.array([2, 3, 4, 5, 6]) * nb_timesteps - 1, j]
                x_shock = np.array([2, 3, 4, 5, 6]) * (nb_timesteps - 1) - 1
                axb.plot(x_shock, y_shock, 'r.')
                subs.append(axb)
            y_for = wb[6*nb_timesteps:, j].reshape((-1, nb_timesteps))[:, 1:].reshape((-1,))
            subs[jn].plot(y_for, color=colour, label=label)

    subs[len(subs)//2 - 1].legend(fontsize=8, bbox_to_anchor=(1.05, 1.35), loc='upper left',
                                  framealpha=0., labelspacing=1.)
    subs[-1].legend(fontsize=8, bbox_to_anchor=(1.05, 1.35), loc='upper left', framealpha=0., labelspacing=1)
    plt.tight_layout()
    plt.show()


def plot_model_structure(m: MBModel, nids=None, vmin=-.5, vmax=.5, only_nids=False):
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
    fig = plt.figure("weight-tables-" + m.__class__.__name__.lower(), figsize=(5, 4))
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


def plot_learning_rule(wrt_k=True, wrt_w=True, wrt_d=True, colour_bar=False):
    """
    Plots 21 contour sub-plots in a 3-by-7 grid where the relationship among the parameters of the dopaminergic learning
    rule is visualised. In each row contours are created for 7 values of one of the parameters: KC+W(t), D(t) or W(t+1).
    """

    k_, d_, w_ = np.linspace(0, 1, 101), np.linspace(-1, 1, 101), np.linspace(0, 1, 101)

    nb_rows = int(wrt_k) + int(wrt_w) + int(wrt_d)
    nb_cols = 7 + int(colour_bar)
    plt.figure("dlr", figsize=(nb_cols - 1, nb_rows + 1 - .5 * float(colour_bar)))
    if wrt_k:
        w, d = np.meshgrid(w_, d_)
        for i, k in enumerate(np.linspace(0, 1, 7)):
            y = d * (k + w - 1)
            plt.subplot(nb_rows, nb_cols, i + 1)
            plt.contour(w, d, y, 30, cmap="coolwarm", vmin=-2, vmax=2)
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
            plt.contour(k, d, y, 30, cmap="coolwarm", vmin=-2, vmax=2)
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
            plt.contour(k, w, y, 30, cmap="coolwarm", vmin=-2, vmax=2)
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


def plot_learning_rule_3d():
    """
    Three-dimensional plot of the relationship between the parameters of the dopaminergic learning rule.
    """
    from mpl_toolkits.mplot3d import Axes3D

    plt.figure("dlr-3d", figsize=(3, 3))
    ax = plt.gca(projection='3d')

    k_w_, d_, = np.linspace(0, 3, 16), np.linspace(-1, 1, 11)
    k_w, d = np.meshgrid(k_w_, d_)
    w = d * (k_w - 1)

    ax.contour(k_w, d, w, 60, cmap="coolwarm")
    ax.set_xlabel(r"$k^i(t) + W^{ij}(t)$", fontsize=8)
    ax.set_ylabel(r"$d^j(t)$", fontsize=8)
    ax.set_zlabel(r"$dW^{ij}/dt$", fontsize=8)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_yticks([-1, 0, 1])
    ax.set_zticks([-2, 0, 2])
    ax.set_xlim([0, 3])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-2, 2])
    ax.tick_params(labelsize=8)
    # ax.set_axis_off()

    ax.view_init(5, -97)

    plt.tight_layout()
    plt.show()


def plot_fom(m, nids=None):
    """
    The responses of the first order memory (FOM) sub-circuit.

    :param m: the model to get the responses from
    :type m: MBModel
    :param nids: (optional) the 2 neuron indices which we want to plot their responses. Default is 1 and 6.
    :type nids: List[int]
    """
    if nids is None:
        nids = [1, 6]
    _plot_subcircuit(m, nids, nnames=[r"$d_{av}$", r"$a_{at}$"], title="fom",
                     ncolours=["#db006aff", "#6adbb8ff"], uss=["r", None])


def plot_ltm(m, nids=None):
    """
    The responses of the long-term memory (LTM) sub-circuit.

    :param m: the model to get the responses from
    :type m: MBModel
    :param nids: (optional) the 2 neuron indices which we want to plot their responses. Default is 2 and 10.
    :type nids: List[int]
    """
    if nids is None:
        nids = [2, 10]
    _plot_subcircuit(m, nids, nnames=[r"$r_{at}$", r"$m_{at}$"], title="ltm",
                     ncolours=["#6adb00ff", "#6adbb8ff"], uss=["g", None])


def plot_bm(m, nids=None):
    """
    The responses of the blocking memory (BM) sub-circuit.

    :param m: the model to get the responses from
    :type m: MBModel
    :param nids: (optional) the 2 neuron indices which we want to plot their responses. Default is 6 and 9.
    :type nids: List[int]
    """
    if nids is None:
        nids = [6, 9]
    _plot_subcircuit(m, nids, nnames=[r"$a_{at}$", r"$h_{av}$"], title="bm",
                     ncolours=["#6adbb8ff", "#db6a6aff"])


def plot_rsom(m, nids=None):
    """
    The responses of the reciprocal second order memories (RSOM) sub-circuit.

    :param m: the model to get the responses from
    :type m: MBModel
    :param nids: (optional) the 4 neuron indices which we want to plot their responses. Default is 3, 8, 2 and 9.
    :type nids: List[int]
    """
    if nids is None:
        nids = [3, 8, 2, 9]
    _plot_subcircuit(m, nids, title="rsom", uss=["r", None, None, None],
                     nnames=[r"$r_{av}$", r"$h_{at}$", r"$r_{at}$", r"$h_{av}$"],
                     ncolours=["#db006aff", "#6adbb8ff", "#6adb00ff", "#db6a6aff"])


def plot_rfm(m, nids=None):
    """
    The responses of the reciprocal forgetting memories (RFM) sub-circuit.

    :param m: the model to get the responses from
    :type m: MBModel
    :param nids: (optional) the 4 neuron indices which we want to plot their responses. Default is 5, 10, 4 and 11.
    :type nids: List[int]
    """
    if nids is None:
        nids = [5, 10, 4, 11]
    _plot_subcircuit(m, nids, title="rfm",
                     nnames=[r"$f_{av}$", r"$m_{at}$", r"$f_{at}$", r"$m_{av}$"],
                     ncolours=["#db006aff", "#6adbb8ff", "#6adb00ff", "#db6a6aff"])


def plot_mdm(m, nids=None):
    """
    The responses of the memory digestion mechanism (MDM).

    :param m: the model to get the responses from
    :type m: MBModel
    :param nids: (optional) the 4 neuron indices which we want to plot their responses. Default is 4, 8, 2 and 10.
    :type nids: List[int]
    """
    if nids is None:
        nids = [4, 8, 2, 10]
    _plot_subcircuit(m, nids, title="mdm", uss=[None, None, "g", None],
                     nnames=[r"$f_{at}$", r"$h_{at}$", r"$r_{at}$", r"$m_{at}$"],
                     ncolours=["#6adb00ff", "#6adbb8ff", "#6adb00ff", "#6adbb8ff"])


def _plot_subcircuit(m, nids, nnames, ncolours, uss=None, title="sub-circuit"):
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

    for i in range(nb_cols):
        if nb_cols > 1:
            fig_title = "%s_%d" % (title, i+1)
        else:
            fig_title = title
        plt.figure(fig_title, figsize=(1, 1))
        plt.clf()

        ax = plt.subplot(111)
        for nid, nname, colour, us in zip(
                nids[i*2:(i+1)*2], nnames[i*2:(i+1)*2], ncolours[i*2:(i+1)*2], uss[i*2:(i+1)*2]):
            v = m._v[:, nid]
            plt.plot(v, c=colour, lw=2, label=nname)
            if us is not None:
                plt.plot(np.arange(len(v))[v > 1], v[v > 1], us + "*")
        plt.ylim([-0.1, 2.1])
        plt.yticks([0, 1, 2])
        plt.xlim([1, 100])
        plt.xticks([10, 30, 60, 80],
                   [r"$t_{1}$", r"$t_{2}$", r"$t_{3}$", r"$t_{4}$"])
        plt.tick_params(labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # plt.legend(loc="upper right", fontsize=8)

        plt.tight_layout()
    plt.show()
