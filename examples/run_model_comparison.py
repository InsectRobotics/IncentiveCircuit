
__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright 2021, School of Informatics, the University of Edinburgh"
__licence__ = "MIT"
__version__ = "1.1-alpha"
__maintainer__ = "Evripidis Gkanias"
__email__ = "ev.gkanias@ed.ac.uk"
__status__ = "Production"


from incentive.imaging import load_data, get_summarised_responses
from incentive.circuit import IncentiveCircuit
from incentive.results import run_main_experiments

import matplotlib.pyplot as plt
import numpy as np


def main(*args):

    nb_kcs = 10
    nb_kc_odour = 5

    # read the parameters
    experiment = "B+"

    # load the data
    df = load_data(experiment)
    data_res = get_summarised_responses(df, experiment=experiment)

    # create the Incentive Complex
    model = IncentiveCircuit(
        learning_rule="dlr", nb_apl=0, nb_timesteps=3, nb_trials=26, nb_kc=nb_kcs, has_real_names=False,
        nb_kc_odour=nb_kc_odour, nb_active_kcs=3,
        # nb_kc_odour_1=nb_kc_odour + 1, nb_kc_odour_2=nb_kc_odour + 2,
        has_sm=True, has_rm=True, has_rrm=True, has_ltm=True, has_rfm=True, has_mam=True)

    # run the reversal experiment and get a copy of the model with the history of their responses and parameters
    models = run_main_experiments(model, reversal=True, unpaired=False, extinction=False)
    model_res = get_summarised_responses_from_model(models[0])

    data_names, model_names = [], []
    data_feats, model_feats = [], []
    dq25_feats, dq75_feats = [], []
    for neuron in data_res:
        data_names.append(neuron)
        data_feats.append(np.r_[
                                # data_res[neuron]["qa50"][2:12] - data_res[neuron]["qa50"][0:10],
                                data_res[neuron]["qa50"][8:12] - data_res[neuron]["qa50"][0:4],
                                data_res[neuron]["qa50"][14:18] - data_res[neuron]["qa50"][12:16],
                                # data_res[neuron]["qb50"][2:12] - data_res[neuron]["qb50"][0:10],
                                data_res[neuron]["qb50"][8:12] - data_res[neuron]["qb50"][0:4],
                                data_res[neuron]["qb50"][12:16] - data_res[neuron]["qb50"][10:14]])
        dq25_feats.append(np.r_[
                                # data_res[neuron]["qa25"][2:12] - data_res[neuron]["qa25"][0:10],
                                data_res[neuron]["qa25"][8:12] - data_res[neuron]["qa25"][0:4],
                                data_res[neuron]["qa25"][14:18] - data_res[neuron]["qa25"][12:16],
                                # data_res[neuron]["qb25"][2:12] - data_res[neuron]["qb25"][0:10],
                                data_res[neuron]["qb25"][8:12] - data_res[neuron]["qb25"][0:4],
                                data_res[neuron]["qb25"][12:16] - data_res[neuron]["qb25"][10:14]])
        dq75_feats.append(np.r_[
                                # data_res[neuron]["qa75"][2:12] - data_res[neuron]["qa75"][0:10],
                                data_res[neuron]["qa75"][8:12] - data_res[neuron]["qa75"][0:4],
                                data_res[neuron]["qa75"][14:18] - data_res[neuron]["qa75"][12:16],
                                # data_res[neuron]["qb75"][2:12] - data_res[neuron]["qb75"][0:10],
                                data_res[neuron]["qb75"][8:12] - data_res[neuron]["qb75"][0:4],
                                data_res[neuron]["qb75"][12:16] - data_res[neuron]["qb75"][10:14]])

    for neuron in model_res:
        model_names.append(neuron)
        model_feats.append(np.r_[
                                 # model_res[neuron]["va"][3:13] - model_res[neuron]["va"][1:11],
                                 model_res[neuron]["va"][9:13] - model_res[neuron]["va"][1:5],
                                 model_res[neuron]["va"][14:18] - model_res[neuron]["va"][12:16],
                                 # model_res[neuron]["vb"][2:12] - model_res[neuron]["vb"][0:10],
                                 model_res[neuron]["vb"][8:12] - model_res[neuron]["vb"][0:4],
                                 model_res[neuron]["vb"][13:17] - model_res[neuron]["vb"][11:15]])

    data_feats = np.array(data_feats)[:, ::2] / 2
    dq25_feats = np.array(dq25_feats)[:, ::2] / 2
    dq75_feats = np.array(dq75_feats)[:, ::2] / 2
    model_feats = np.array(model_feats)[:, ::2] / 2

    c = np.zeros((model_feats.shape[0], data_feats.shape[0]), dtype=float)
    d = np.zeros((model_feats.shape[0], data_feats.shape[0]), dtype=float)
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            c[i, j] = np.correlate(data_feats[j], model_feats[i])
            d[i, j] = np.correlate(-data_feats[j], model_feats[i])
            # d_high = np.maximum(dq75_feats[j] - model_feats[i], 0)
            # d_low = np.minimum(model_feats[i] - dq25_feats[j], 0)
            # c[i, j] = 1
            # d[i, j] = np.sqrt(np.sum(np.square(d_high - d_low)))
            # print(np.sum(d_high))

    c = c - d
    # c = np.maximum(c, 0) - np.maximum(d, 0)
    # c = np.sqrt(np.maximum(c, 0)) - np.sqrt(np.maximum(d, 0))

    plt.figure("cross-correlation", figsize=(8, 8))

    nb_data = len(data_names)
    nb_mbons = 18
    nb_dans = nb_data - nb_mbons
    dan_ids = np.array([33, 39, 21, 41, 42, 30])
    mbon_ids = np.array([13, 16, 14, 17, 12, 2])
    dan_rand = model.rng.permutation(nb_dans)[:6]
    mbon_rand = model.rng.permutation(nb_mbons)[:6]

    c_dan = c[:6, nb_mbons:]
    c_mbon = c[6:, :nb_mbons]
    c_model = np.r_[c_dan[np.arange(6), dan_ids - nb_mbons], c_mbon[np.arange(6), mbon_ids]].mean()
    c_random = np.r_[c_dan[np.arange(6), dan_rand], c_mbon[np.arange(6), mbon_rand]].mean()
    print(f"Cross-correlation of our model and the data: {c_model:.2f}")
    print(f"Cross-correlation of a random model and the data: {c_random:.2f}")

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
