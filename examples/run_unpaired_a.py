__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright 2021, School of Informatics, the University of Edinburgh"
__licence__ = "MIT"
__version__ = "1.1-alpha"
__maintainer__ = "Evripidis Gkanias"
__email__ = "ev.gkanias@ed.ac.uk"
__status__ = "Production"

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
    kc1 = read_arg(["-k1", "--nb-kc1", "--odour1"], vtype=int, default=7)
    kc2 = read_arg(["-k2", "--nb-kc2", "--odour2"], vtype=int, default=6)

    pi = []
    for i in range(10):
        # create the Incentive Complex
        model = IncentiveCircuit(
            learning_rule="dpr", nb_timesteps=3, nb_trials=26, rng=np.random.RandomState(2021 + i),
            nb_kc=nb_kcs, nb_active_kcs=5, nb_kc_odour_1=kc1, nb_kc_odour_2=kc2, has_real_names=False,
            has_sm=True, has_rm=True, has_ltm=True, has_rrm=True, has_rfm=True, has_mam=True)

        # run all the experiments and get a copy of the model with the history of their responses and parameters for each
        # one of them
        models = [run_custom_routine(model, routine=unpaired_a_routine)]
        # run_arg(model, models, only_nids)

        v = models[0]._v
        va = v[1:].reshape((-1, 2, 3, v.shape[-1]))[:, 0].reshape((-1, v.shape[-1]))
        vb = v[1:].reshape((-1, 2, 3, v.shape[-1]))[:, 1].reshape((-1, v.shape[-1]))

        vai, vbi = 0, 0
        signs = {"s_{at}": +1, "s_{av}": -1, "r_{at}": +1, "r_{av}": -1, "m_{at}": + 1, "m_{av}": -1}
        for j in range(12):
            if model.names[j] not in signs:
                continue

            vaj = va[:, j].reshape((-1, 3))[:, 1:].reshape((-1,))
            vbj = vb[:, j].reshape((-1, 3))[:, 1:].reshape((-1,))

            vai += signs[model.names[j]] * vaj
            vbi += signs[model.names[j]] * vbj
        pi.append((vai - vbi) / 4)

    pi = np.array(pi).T
    pi_q50 = np.nanmedian(pi, axis=1)

    plt.figure('simple-unpaired-pi', figsize=(2, 2))
    plt.plot([0, 26], [0, 0], 'grey', lw=3)
    plt.plot(pi, 'k-', lw=.5, alpha=.5)
    plt.plot(pi_q50, 'k-', lw=2)
    plt.plot([3, 5, 7, 9, 11], pi_q50[[3, 5, 7, 9, 11]], 'r.')
    plt.yticks([-.8, 0, .8], ["B", "0", "A"])
    plt.xlim(2, 25)
    plt.xticks([2, 8, 13, 15, 20, 25], ["", "train", "", "", "test", ""])
    plt.ylim(-1, 1)
    plt.show()

    # plot the results based on the input flags
    # run_arg(model, models, only_nids)
