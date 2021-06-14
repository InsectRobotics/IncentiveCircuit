import numpy as np

from incentive.circuit import IncentiveCircuit
from incentive.routines import unpaired_a_routine
from incentive.results import run_custom_routine
from incentive.tools import read_arg, run_arg

import matplotlib.pyplot as plt


if __name__ == '__main__':

    # read the parameters
    only_nids = read_arg(["--only-nids"])
    nb_kcs = read_arg(["-k", "--nb-kc", "--nb-kcs"], vtype=int, default=10)
    kc1 = read_arg(["-k1", "--nb-kc1", "--odour1"], vtype=int, default=nb_kcs // 2)
    kc2 = read_arg(["-k2", "--nb-kc2", "--odour2"], vtype=int, default=nb_kcs // 2)

    # create the Incentive Complex
    model = IncentiveCircuit(
        learning_rule="dlr", nb_apl=0, pn2kc_init="default", nb_timesteps=3, nb_trials=26,
        nb_kc=nb_kcs, nb_kc_odour_1=kc1, nb_kc_odour_2=kc2, has_real_names=False,
        has_sm=True, has_rm=True, has_ltm=True, has_rrm=True, has_rfm=True, has_mam=True)

    # run all the experiments and get a copy of the model with the history of their responses and parameters for each
    # one of them
    models = [run_custom_routine(model, routine=unpaired_a_routine)]
    # run_arg(model, models, only_nids)

    v = models[0]._v
    va = v[1:].reshape((-1, 2, 3, v.shape[-1]))[:, 0].reshape((-1, v.shape[-1]))
    vb = v[1:].reshape((-1, 2, 3, v.shape[-1]))[:, 1].reshape((-1, v.shape[-1]))

    pis = {}
    for j in range(12):
        vaj = va[:, j].reshape((-1, 3))[:, 1:].reshape((-1,))
        vbj = vb[:, j].reshape((-1, 3))[:, 1:].reshape((-1,))

        pi = (vbj - vaj) / 2.
        pis[model.names[j]] = pi

    pi = pis["s_{at}"] + pis["r_{at}"] + pis["m_{at}"] - pis["s_{av}"] - pis["r_{av}"] - pis["m_{av}"]
    plt.figure('simple-unpaired-pi', figsize=(2, 2))
    plt.plot([0, 26], [0, 0], 'grey', lw=3)
    plt.plot(pi, 'k-')
    plt.plot([3, 5, 7, 9, 11], pi[[3, 5, 7, 9, 11]], 'r.')
    plt.yticks([-.8, 0, .8], ["A", "0", "B"])
    plt.xlim(2, 25)
    plt.xticks([2, 8, 13, 15, 20, 25], ["", "train", "", "", "test", ""])
    plt.ylim(-1, 1)
    plt.show()

    # plot the results based on the input flags
    # run_arg(model, models, only_nids)
