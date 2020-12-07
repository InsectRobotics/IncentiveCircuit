from modelling import MBModel

import matplotlib.pyplot as plt
import numpy as np


class MotivationModel(MBModel):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("nb_dan", 3)
        kwargs.setdefault("nb_mbon", 3)
        kwargs.setdefault("leak", .0)
        super().__init__(*args, **kwargs)

        p_dan_prim = 0
        p_dan_seco = 1
        p_mbon_pri = 3
        p_mbon_sec = 4
        self._v[:, p_dan_prim:p_dan_seco] = +2.  # P-DANs
        self._v[:, p_dan_seco:p_mbon_pri] = 1.  # S-DANs
        self._v[:, p_mbon_pri:p_mbon_sec] = 1.  # P-MBONs
        self._v[:, p_mbon_sec:] = 0.  # S-MBONs
        # self._v[:, 0] = +2.  # PPL1-γ1ped
        # self._v[:, 2] = 1.   # PPL1-γ2α'1a
        # self._v[:, 13] = 1.  # PAM-β'2a

        self.v_init[p_dan_prim:p_dan_seco] = 2.
        self.v_init[p_dan_seco:p_mbon_pri] = -7.
        self.v_init[p_mbon_pri:p_mbon_sec] = -3.
        self.v_init[p_mbon_sec:] = 0.

        nb_neurons = self.nb_dan + self.nb_mbon
        print(nb_neurons, self.nb_dan, self.nb_mbon)

        #     DANs                          MBONs
        # --------------------------------------------------
        # 00: sadness (γ1ped)           03: joy (γ1ped)
        # 01: disapproval (PPL1-γ2α'1)  04: optimism (γ2α'1)
        # 02: optimism (PAM-β'2a)       05: disapproval (γ5β'2a)
        self._w_m2v = np.zeros((nb_neurons, nb_neurons), dtype=float)
        print(self._w_m2v)
        self._w_m2v[p_mbon_pri:p_mbon_sec, p_dan_prim:p_dan_seco] = -np.eye(1)  # primary emotions depress their DANs
        self._w_m2v[p_mbon_pri:p_mbon_sec, p_mbon_sec:] = np.array([  # P-MBONs to S-MBONs
            [-.0, -.5]
        ]) * 1.
        self._w_m2v[p_mbon_sec:, p_dan_seco:p_mbon_pri] = np.array([  # S-MBONs to S-DANs
            [+.0, +1.],
            [+1., +.0]
        ]) * 2.

        self.names[0], self.names[1], self.names[2] = ("PPL1-γ1ped", "PPL1-γ2α'1", "PAM-β'2a")
        self.names[3], self.names[4], self.names[5] = ("MBON-γ1ped", "MBON-γ2α'1", "MBON-γ5β'2a")

        # plt.figure("Synaptic weights")
        # plt.imshow(self._w_m2v, vmin=-1, vmax=1, cmap="coolwarm")
        # plt.xticks([0, 9, 13, 16, 25, 29], np.array(self.names)[[0, 9, 13, 16, 25, 29]])
        # plt.yticks([0, 9, 13, 16, 25, 29], np.array(self.names)[[0, 9, 13, 16, 25, 29]])
        # plt.tight_layout()
        # plt.show()


def plot_population(ms, nids=None):
    title = "liyan-motivation-" + '-'.join(str(ms[0]).split("'")[1:-1:2])

    nb_models = len(ms)
    xticks = ["%d" % (i+1) for i in range(16)]

    plt.figure(title, figsize=(7.5, 10))

    for i in range(nb_models):
        nb_neurons = len(ms[i].names)
        yticks = np.array(ms[i].names)
        if nids == None:
            nids = np.arange(ms[i].nb_dan + ms[i].nb_mbon)[::8]
        v = ms[i]._v
        w = ms[i].w_k2m

        plt.subplot(nb_models * 2, 2, 1 + i * 4)
        va = v[1:].reshape((-1, 2, 2, nb_neurons))[:, ::2].reshape((-1, nb_neurons))
        plt.imshow(va.T, vmin=-2., vmax=2., cmap="coolwarm", aspect="auto")
        plt.plot([[15, 17, 19, 21, 23, 25]] * 2, [[0] * 6, [nb_neurons-1] * 6], 'r-')
        plt.xticks(2 * np.arange(va.shape[0] // 2), xticks[:va.shape[0] // 2])
        plt.yticks(nids, yticks[nids])
        plt.title("%s - odour A - value" % ms[i].routine_name, color="C%d" % (2 * i + 0))

        plt.subplot(nb_models * 2, 2, 2 + i * 4)
        vb = v[1:].reshape((-1, 2, 2, nb_neurons))[:, 1::2].reshape((-1, nb_neurons))
        plt.imshow(vb.T, vmin=-2., vmax=2., cmap="coolwarm", aspect="auto")
        plt.plot([[3, 5, 7, 9, 11]] * 2, [[0] * 5, [nb_neurons-1] * 5], 'r-')
        plt.xticks(2 * np.arange(vb.shape[0] // 2), xticks[:vb.shape[0] // 2])
        plt.yticks(nids, [""] * len(nids))
        plt.title("%s - odour B - value" % ms[i].routine_name, color="C%d" % (2 * i + 1))

        plt.subplot(nb_models * 2, 2, 3 + i * 4)
        wa = w[5:, w.shape[1] // 2]
        plt.imshow(wa.T, vmin=-2., vmax=2., cmap="coolwarm", aspect="auto")
        plt.plot([[31, 35, 39, 43, 47, 51]] * 2, [[0] * 6, [nb_neurons-1] * 6], 'r-')
        plt.xticks(np.arange(wa.shape[0])[::4], xticks[:wa.shape[0] // 4])
        plt.yticks(nids, yticks[nids])
        plt.title("%s - odour A - weights" % ms[i].routine_name, color="C%d" % (2 * i + 0))

        plt.subplot(nb_models * 2, 2, 4 + i * 4)
        wb = w[5:, 0]
        plt.imshow(wb.T, vmin=-2., vmax=2., cmap="coolwarm", aspect="auto")
        plt.plot([[7, 11, 15, 19, 23]] * 2, [[0] * 5, [nb_neurons-1] * 5], 'r-')
        plt.xticks(np.arange(wb.shape[0])[::4], xticks[:wb.shape[0] // 4])
        plt.yticks(nids, [""] * len(nids))
        plt.title("%s - odour B - weights" % ms[i].routine_name, color="C%d" % (2 * i + 1))

    plt.colorbar()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    from evaluation import evaluate, create_behaviour_map
    target = create_behaviour_map(cs_only=True)

    import pandas as pd

    pd.options.display.max_columns = 16
    pd.options.display.max_rows = 16
    pd.options.display.width = 1000

    neurons = [0, 1, 2, 3, 4, 5]
    nb_kcs = 10
    kc1, kc2 = nb_kcs // 2, nb_kcs // 2

    model = MotivationModel(learning_rule="dan-based", nb_apl=1, pn2kc_init="default", verbose=False,
                            timesteps=2, trials=28, nb_kc=nb_kcs, nb_kc_odour_1=kc1, nb_kc_odour_2=kc2)

    val, acc, prediction, models = evaluate(model, tolerance=.02, nids=neurons, percentage=True,
                                            behav_mean=target, cs_only=True, mbon_only=False, reversal=True, extinction=True)

    print("TARGET")
    print(target.T)
    print("PREDICTION")
    print(prediction.T)
    print("ACCURACY")
    print(acc.T)
    print(str(models))
    print("Score: %.2f" % val)
    print("#KC1: %d, #KC2: %d" % (kc1, kc2))

    # MBModel.plot_overlap(models, nids=neurons, score=acc)
    # MBModel.plot_timeline(models=models, nids=neurons, score=acc, target=target, nb_trials=13)
    plot_population(models, neurons)
