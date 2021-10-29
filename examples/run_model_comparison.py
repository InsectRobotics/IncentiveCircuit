
__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright 2021, School of Informatics, the University of Edinburgh"
__licence__ = "MIT"
__version__ = "1.1-alpha"
__maintainer__ = "Evripidis Gkanias"
__email__ = "ev.gkanias@ed.ac.uk"
__status__ = "Production"


from incentive.imaging import load_data, get_individual_responses
from incentive.circuit import IncentiveCircuit
from incentive.results import run_main_experiments

from scipy.stats.stats import pearsonr

import matplotlib.pyplot as plt
import numpy as np


def main(*args):

    nb_kcs = 10
    nb_kc_odour = 5
    show_max = False

    # read the parameters
    experiment = "B+"

    # load the data
    df = load_data(experiment)
    data_res = get_individual_responses(df, experiment=experiment)

    # create the Incentive Complex
    model = IncentiveCircuit(
        learning_rule="dlr", nb_timesteps=3, nb_trials=26, nb_kc=nb_kcs, has_real_names=False,
        nb_kc_odour=nb_kc_odour, nb_active_kcs=5, nb_kc_odour_1=7, nb_kc_odour_2=6,
        has_sm=True, has_rm=True, has_rrm=True, has_ltm=True, has_rfm=True, has_mam=True)

    # run the reversal experiment and get a copy of the model with the history of their responses and parameters
    models = run_main_experiments(model, reversal=True, unpaired=False, extinction=False)
    model_res = get_summarised_responses_from_model(models[0])

    data_names, model_names = [], []
    data_feats, model_feats = [], []
    for neuron in data_res:
        data_names.append(neuron)
        data_feats.append(np.hstack([
                                # data_res[neuron]["ma"][:, 2:12] - data_res[neuron]["ma"][:, 0:10],
                                data_res[neuron]["ma"][:, 8:12] - data_res[neuron]["ma"][:, 0:4],
                                data_res[neuron]["ma"][:, 14:18] - data_res[neuron]["ma"][:, 12:16],
                                # data_res[neuron]["mb"][:, 2:12] - data_res[neuron]["mb"][:, 0:10],
                                data_res[neuron]["mb"][:, 8:12] - data_res[neuron]["mb"][:, 0:4],
                                data_res[neuron]["mb"][:, 12:16] - data_res[neuron]["mb"][:, 10:14]]).T)

    for neuron in model_res:
        model_names.append(neuron)
        model_feats.append(np.r_[
                                 # model_res[neuron]["va"][3:13] - model_res[neuron]["va"][1:11],
                                 model_res[neuron]["va"][9:13] - model_res[neuron]["va"][1:5],
                                 model_res[neuron]["va"][14:18] - model_res[neuron]["va"][12:16],
                                 # model_res[neuron]["vb"][2:12] - model_res[neuron]["vb"][0:10],
                                 model_res[neuron]["vb"][8:12] - model_res[neuron]["vb"][0:4],
                                 model_res[neuron]["vb"][13:17] - model_res[neuron]["vb"][11:15]])

    c = np.zeros((len(model_feats), len(data_feats)), dtype=float)
    p = np.zeros((len(model_feats), len(data_feats)), dtype=float)
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            if show_max:
                c_k, p_k = [], []
                for k in range(data_feats[j].shape[1]):
                    r_, p_ = pearsonr(data_feats[j][:, k], model_feats[i])
                    c_k.append(r_)
                    p_k.append(p_)
                k = np.argmax(c_k)
                c[i, j] = c_k[k]
                p[i, j] = p_k[k]
            else:
                c[i, j], p[i, j] = pearsonr(np.nanmedian(data_feats[j], axis=1), model_feats[i])

    plt.figure("cross-correlation", figsize=(8, 8))

    nb_data = len(data_names)
    nb_mbons = 18
    nb_dans = nb_data - nb_mbons
    dan_ids = np.array([33, 39, 21, 41, 42, 30])
    mbon_ids = np.array([13, 16, 14, 17, 12, 2])
    dan_rand = model.rng.permutation(nb_dans)[:6]
    mbon_rand = model.rng.permutation(nb_mbons)[:6]

    c_dan = c[:6, nb_mbons:]
    p_dan = p[:6, nb_mbons:]
    c_mbon = c[6:, :nb_mbons]
    p_mbon = p[6:, :nb_mbons]
    c_model = np.nanmean(np.r_[c_dan[np.arange(6), dan_ids - nb_mbons], c_mbon[np.arange(6), mbon_ids]])
    p_model = np.nanmean(np.r_[p_dan[np.arange(6), dan_ids - nb_mbons], p_mbon[np.arange(6), mbon_ids]])
    c_random = np.nanmean(np.r_[c_dan[np.arange(6), dan_rand], c_mbon[np.arange(6), mbon_rand]])
    p_random = np.nanmean(np.r_[p_dan[np.arange(6), dan_rand], p_mbon[np.arange(6), mbon_rand]])
    print(f"Cross-correlation of our model and the data: R={c_model:.2f}, p={p_model:.4f}")
    print(f"Cross-correlation of a random model and the data: R={c_random:.2f}, p={p_random:.4f}")

    ax = plt.subplot(211)  # DANs
    plt.imshow(c_dan, vmin=-1, vmax=1, cmap="coolwarm")
    plt.scatter(np.array(dan_rand), np.arange(6), c='grey')
    plt.scatter(np.array(dan_ids) - nb_mbons, np.arange(6), c='k')
    plt.xticks(np.arange(nb_dans), data_names[nb_mbons:], rotation=90)
    plt.yticks(np.arange(6), [r"$%s$" % n for n in model_names[:6]])
    ax.xaxis.tick_top()

    plt.subplot(212)  # MBONs
    plt.imshow(c_mbon, vmin=-1, vmax=1, cmap="coolwarm")
    plt.scatter(mbon_rand, np.arange(6), c='grey')
    plt.scatter(mbon_ids, np.arange(6), c='k')
    plt.xticks(np.arange(nb_mbons), data_names[:nb_mbons], rotation=90)
    plt.yticks(np.arange(6), [r"$%s$" % n for n in model_names[6:]])

    for i, i_r, j, dname, mname in zip(dan_ids - nb_mbons, dan_rand, np.arange(6), np.array(data_names)[dan_ids], model_names[:6]):
        print(f"{dname} (${mname}$): R={c_dan[j, i]:.2f}, p={p_dan[j, i]:.4f}")

    for i, i_r, j, dname, mname in zip(mbon_ids, mbon_rand, np.arange(6), np.array(data_names)[mbon_ids], model_names[6:]):
        print(f"{dname} (${mname}$): R={c_mbon[j, i]:.2f}, p={p_mbon[j, i]:.4f}")

    plt.tight_layout()
    plt.show()


def get_summarised_responses_from_model(model):
    """
    Plots the average responses of the neurons per phase/trial with overlapping lines.

    Parameters
    ----------
    model: MBModel
        the model where the values are taken from
    """

    nb_odours = 2
    xticks = ["%d" % i for i in range(16)]

    if model.neuron_ids is None:
        nids = np.arange(model.nb_dan + model.nb_mbon)[::8]
    else:
        nids = model.neuron_ids

    nb_neurons = len(nids)
    nb_timesteps = model.nb_timesteps
    nb_trials = model.nb_trials

    v = model._v

    # trial, odour, time-step, neuron
    va = v[1:].reshape((-1, nb_odours, nb_timesteps, v.shape[-1]))[:, 0].reshape((-1, v.shape[-1]))
    vb = v[1:].reshape((-1, nb_odours, nb_timesteps, v.shape[-1]))[:, 1].reshape((-1, v.shape[-1]))

    x_ticks_ = xticks[1:(nb_trials // 2) // 2] * 2
    n = len(x_ticks_)
    _x_ticks = np.arange(n, dtype=float) + 2 - 1 / (nb_timesteps - 1)
    _x_ticks[n//2:] += 1. - 1 / (nb_timesteps - 1)

    x_ = np.arange(0, nb_trials // 2, 1 / (nb_timesteps - 1)) - 1 / (nb_timesteps - 1)

    responses = {}

    for j in range(nb_neurons):

        vaj = va[:, j].reshape((-1, nb_timesteps))[:, 1:].reshape((-1,))
        vbj = vb[:, j].reshape((-1, nb_timesteps))[:, 1:].reshape((-1,))

        responses[model.names[j]] = {
            "xa": x_,
            "xb": x_ + 1 - 1 / (nb_timesteps - 1),
            "va": vaj,
            "vb": vbj
        }

    return responses


if __name__ == '__main__':
    import sys

    main(*sys.argv)
