
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

from incentive.plot import plot_responses_from_model
from run_model_comparison import get_summarised_responses_from_model

from scipy.stats.stats import pearsonr

import matplotlib.pyplot as plt
import numpy as np


def main(*args):

    nb_kcs = 10
    nb_kc_odour = 5
    show_max = False

    # # read the parameters
    # experiment = "B+"
    #
    # # load the data
    # df = load_data(experiment)
    # data_res = get_individual_responses(df, experiment=experiment)
    # dan_ids = [33, 39, 21, 41, 42, 30]
    # mbon_ids = [13, 16, 14, 17, 12, 2]
    # data_ids = dan_ids + mbon_ids

    bs = [-2.0, -1.5, -1.0, -0.5, 0.0]
    ws = [.1, .25, .5, .75, 1]

    parameters = {
        "b_d": -0.5,
        "b_c": -0.15,
        "b_f": -0.15,
        "b_s": -2.0,
        "b_r": -0.5,
        "b_m": -0.5,
        "w_s2d": 0.3,
        "w_d2s": 1.0,
        "w_s2r": 1.0,
        "w_r2c": 0.5,
        "w_c2r": 1.0,
        "w_m2c": 0.3,
        "w_c2m": 0.3,
        "w_m2f": 0.5,
        "w_f2m": 1.0,
        "w_f2r": 0.3
    }
    targets = {
        "b_d": [0, 1],
        "b_c": [2, 3],
        "b_f": [4, 5],
        "b_s": [6, 7],
        "b_r": [8, 9],
        "b_m": [10, 11],
        "w_s2d": [0, 1],
        "w_d2s": [6, 7],
        "w_s2r": [8, 9],
        "w_r2c": [2, 3],
        "w_c2r": [8, 9],
        "w_m2c": [2, 3],
        "w_c2m": [10, 11],
        "w_m2f": [4, 5],
        "w_f2m": [10, 11],
        "w_f2r": [8, 9]
    }

    # data_names = []
    # data_feats = []
    # for neuron in enumerate(data_res):
    #     data_names.append(neuron)
    #     data_feats.append(np.hstack([
    #                             # data_res[neuron]["ma"][:, 2:12] - data_res[neuron]["ma"][:, 0:10],
    #                             data_res[neuron]["ma"][:, 8:12] - data_res[neuron]["ma"][:, 0:4],
    #                             data_res[neuron]["ma"][:, 14:18] - data_res[neuron]["ma"][:, 12:16],
    #                             # data_res[neuron]["mb"][:, 2:12] - data_res[neuron]["mb"][:, 0:10],
    #                             data_res[neuron]["mb"][:, 8:12] - data_res[neuron]["mb"][:, 0:4],
    #                             data_res[neuron]["mb"][:, 12:16] - data_res[neuron]["mb"][:, 10:14]]).T)

    for param, value in parameters.items():

        if "b" in param:
            vals = bs
        else:
            vals = ws

        nb_repeats = 10
        models = []
        for val in vals + [value]:
            kwargs = {param: val}
            # create the Incentive Complex
            model = IncentiveCircuit(
                learning_rule="dpr", nb_timesteps=3, nb_trials=26, nb_kc=nb_kcs, has_real_names=False,
                nb_kc_odour=nb_kc_odour, nb_active_kcs=5, nb_kc_odour_1=7, nb_kc_odour_2=6,
                has_sm=True, has_rm=True, has_rrm=True, has_ltm=True, has_rfm=True, has_mam=True, **kwargs)

            # run the reversal experiment and get a copy of the model with the history of their responses and parameters
            for repeat in range(nb_repeats):
                model.rng = np.random.RandomState(2021 + repeat)
                m = run_main_experiments(model, reversal=True, unpaired=False, extinction=False)[0]
                m.routine_name = f"{param}={val:.2f}"
                if len(models) <= repeat:
                    models.append([])
                models[repeat].append(m)

        plot_responses_from_model(models, only_nids=True, title=param, cma="cool", cmb="Wistia",
                                  show_legend=False, figsize=(9, 6), last_high=True)

            # model_res = get_summarised_responses_from_model(models[0])

            # model_names = []
            # model_feats = []
            # for neuron in model_res:
            #     model_names.append(neuron)
            #     model_feats.append(np.r_[
            #                              # model_res[neuron]["va"][3:13] - model_res[neuron]["va"][1:11],
            #                              model_res[neuron]["va"][9:13] - model_res[neuron]["va"][1:5],
            #                              model_res[neuron]["va"][14:18] - model_res[neuron]["va"][12:16],
            #                              # model_res[neuron]["vb"][2:12] - model_res[neuron]["vb"][0:10],
            #                              model_res[neuron]["vb"][8:12] - model_res[neuron]["vb"][0:4],
            #                              model_res[neuron]["vb"][13:17] - model_res[neuron]["vb"][11:15]])
            #
            # c = np.zeros(len(model_feats[targets[param]]), dtype=float)
            # p = np.zeros(len(model_feats[targets[param]]), dtype=float)
            # for i in range(c.shape[0]):
            #     if show_max:
            #         c_k, p_k = [], []
            #         for k in range(c.shape[0]):
            #             r_, p_ = pearsonr(data_feats[data_ids[targets[param][i]]][:, k], model_feats[targets[param][i]])
            #             c_k.append(r_)
            #             p_k.append(p_)
            #         k = np.argmax(c_k)
            #         c[i] = c_k[k]
            #         p[i] = p_k[k]
            #     else:
            #         c[i], p[i] = pearsonr(np.nanmedian(data_feats[data_ids[targets[param][i]]], axis=1),
            #                               model_feats[targets[param][i]])
    #
    # plt.figure("cross-correlation", figsize=(8, 8))
    #
    # nb_data = len(data_names)
    # nb_mbons = 18
    # nb_dans = nb_data - nb_mbons
    # dan_rand = model.rng.permutation(nb_dans)[:6]
    # mbon_rand = model.rng.permutation(nb_mbons)[:6]
    #
    # c_dan = c[:6, nb_mbons:]
    # p_dan = p[:6, nb_mbons:]
    # c_mbon = c[6:, :nb_mbons]
    # p_mbon = p[6:, :nb_mbons]
    # c_model = np.nanmean(np.r_[c_dan[np.arange(6), dan_ids - nb_mbons], c_mbon[np.arange(6), mbon_ids]])
    # p_model = np.nanmean(np.r_[p_dan[np.arange(6), dan_ids - nb_mbons], p_mbon[np.arange(6), mbon_ids]])
    # c_random = np.nanmean(np.r_[c_dan[np.arange(6), dan_rand], c_mbon[np.arange(6), mbon_rand]])
    # p_random = np.nanmean(np.r_[p_dan[np.arange(6), dan_rand], p_mbon[np.arange(6), mbon_rand]])
    # print(f"Cross-correlation of our model and the data: R={c_model:.2f}, p={p_model:.4f}")
    # print(f"Cross-correlation of a random model and the data: R={c_random:.2f}, p={p_random:.4f}")
    #
    # ax = plt.subplot(211)  # DANs
    # plt.imshow(c_dan, vmin=-1, vmax=1, cmap="coolwarm")
    # plt.scatter(np.array(dan_rand), np.arange(6), c='grey')
    # plt.scatter(np.array(dan_ids) - nb_mbons, np.arange(6), c='k')
    # plt.xticks(np.arange(nb_dans), data_names[nb_mbons:], rotation=90)
    # plt.yticks(np.arange(6), [r"$%s$" % n for n in model_names[:6]])
    # ax.xaxis.tick_top()
    #
    # plt.subplot(212)  # MBONs
    # plt.imshow(c_mbon, vmin=-1, vmax=1, cmap="coolwarm")
    # plt.scatter(mbon_rand, np.arange(6), c='grey')
    # plt.scatter(mbon_ids, np.arange(6), c='k')
    # plt.xticks(np.arange(nb_mbons), data_names[:nb_mbons], rotation=90)
    # plt.yticks(np.arange(6), [r"$%s$" % n for n in model_names[6:]])
    #
    # for i, i_r, j, dname, mname in zip(dan_ids - nb_mbons, dan_rand, np.arange(6), np.array(data_names)[dan_ids], model_names[:6]):
    #     print(f"{dname} (${mname}$): R={c_dan[j, i]:.2f}, p={p_dan[j, i]:.4f}")
    #
    # for i, i_r, j, dname, mname in zip(mbon_ids, mbon_rand, np.arange(6), np.array(data_names)[mbon_ids], model_names[6:]):
    #     print(f"{dname} (${mname}$): R={c_mbon[j, i]:.2f}, p={p_mbon[j, i]:.4f}")
    #
    # plt.tight_layout()
    # plt.show()


if __name__ == '__main__':
    import warnings
    import sys

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        main(*sys.argv)
