from modelling import MBModel

import matplotlib.pyplot as plt
import numpy as np


class MotivationModel(MBModel):
    def __init__(self, third_order=False, *args, **kwargs):
        kwargs.setdefault("nb_dan", 16)
        kwargs.setdefault("nb_mbon", 16)
        kwargs.setdefault("leak", .0)
        super().__init__(*args, **kwargs)

        p_dan_prim = 0
        p_dan_seco = 8
        p_mbon_pri = 16
        p_mbon_sec = 24
        self._v[:, p_dan_prim:p_dan_seco] = +2.  # P-DANs
        self._v[:, p_dan_seco:p_mbon_pri] = 1.  # S-DANs
        self._v[:, p_mbon_pri:p_mbon_sec] = 1.  # P-MBONs
        self._v[:, p_mbon_sec:] = 0.  # S-MBONs
        # self._v[:, 0] = +2.  # PPL1-γ1ped
        # self._v[:, 9] = 1.   # PPL1-γ1β'2a
        # self._v[:, 13] = 1.  # PAM-β'2a

        self.v_init[p_dan_prim:p_dan_seco] = 2.
        self.v_init[p_dan_seco:p_mbon_pri] = -7.
        self.v_init[p_mbon_pri:p_mbon_sec] = -3.
        self.v_init[p_mbon_sec:] = -1.

        #     DANs                          MBONs
        # --------------------------------------------------
        # 00: PPL1-γ1ped                16: MBON-γ1ped
        # 01: surprise                  17: anticipation
        # 02: fear                      18: anger
        # 03: trust                     19: disgust
        # 04: PAM-γ4<γ1γ2               20: MBON-γ4>γ1γ2
        # 05: anticipation              21: surprise
        # 06: anger                     22: fear
        # 07: disgust                   23: trust
        # 08: PPL1-γ2α'1                24: MBON-γ2α'1
        # 09: awe                       25: aggressiveness
        # 10: PAM-α1                    26: MBON-α1
        # 11: PPL1-α3                   27: MBON-α3
        # 12: PAM-β'2a                  28: MBON-γ5β'2a
        # 13: aggressiveness            29: awe
        # 14: PPL1-α'1                  30: MBON-α'1
        # 15: remorse                   31: love
        self._w_m2v = np.zeros((32, 32), dtype=float)
        self._w_m2v[p_mbon_pri:p_mbon_sec, p_dan_prim:p_dan_seco] = np.array([  # primary emotions depress their DANs
            [-1., +0., +0., +0., +0., +0., +0., +0.],  # MBON-γ1ped
            [+0., -1., +0., +0., +0., +0., +0., +0.],  # anticipation
            [+0., +0., -1., +0., +0., +0., +0., +0.],  # anger
            [+0., +0., +0., -1., +0., +0., +0., +0.],  # disgust
            [+0., +0., +0., +0., -1., +0., +0., +0.],  # MBON-γ4>γ1γ2
            [+0., +0., +0., +0., +0., -1., +0., +0.],  # surprise
            [+0., +0., +0., +0., +0., +0., -1., +0.],  # fear
            [+0., +0., +0., +0., +0., +0., +0., -1.],  # trust
        ])
        self._w_m2v[p_mbon_sec:, p_dan_seco:p_mbon_pri] = np.array([  # S-MBONs to S-DANs
            [+.0, +.0, +.0, +.0, +1., +.0, +.0, +.0],  # MBON-γ2α'1
            [+.0, +.0, +.0, +.0, +.0, +.0, +.0, +.0],  # aggressiveness
            [+.0, +.0, +.2, +.0, +.0, +.0, +.0, +.0],  # MBON-α1
            [+.0, +.0, +.2, +.0, +.0, +.0, +.0, +.0],  # MBON-α3
            [+1., +.0, +.0, +.0, +.0, +.0, +.0, +.0],  # MBON-γ5β'2a
            [+.0, +.0, +.0, +.0, +.0, +.0, +.0, +.0],  # awe
            [+.0, +.0, +.0, +.0, +.0, +.0, +.5, +.0],  # MBON-α'1
            [+.0, +.0, +.0, +.0, +.0, +.0, +.0, +.0],  # love
        ]) * 2.1
        self._w_m2v[p_mbon_pri:p_mbon_sec, p_mbon_sec:] = np.array([  # P-MBONs to S-MBONs
            [-.0, -.0, +.5, +.5, -.5, -.0, -.0, -.0],  # MBON-γ1ped
            [-.0, -.0, -.0, -.0, -.0, -.0, -.0, -.0],  # anticipation
            [-.0, -.0, -.0, -.0, -.0, -.0, -.0, -.0],  # anger
            [-.0, -.0, -.0, -.0, -.0, -.0, -.0, -.0],  # disgust
            [-.5, -.0, -.0, -.0, -.0, -.0, +1., -.0],  # MBON-γ4>γ1γ2
            [-.0, -.0, -.0, -.0, -.0, -.0, -.0, -.0],  # surprise
            [-.0, -.0, -.0, -.0, -.0, -.0, -.0, -.0],  # fear
            [-.0, -.0, -.0, -.0, -.0, -.0, -.0, -.0],  # trust
        ]) * .5
        self._w_m2v[p_mbon_sec:, p_mbon_sec:] = np.array([  # S-MBONs to S-MBONs
            [+.0, +.0, +.0, +.0, +.0, +.0, +.0, +.0],  # MBON-γ2α'1
            [+.0, +.0, +.0, +.0, +.0, +.0, +.0, +.0],  # aggressiveness
            [+1., +.0, +.0, +.0, +.0, +.0, +.0, +.0],  # MBON-α1
            [+.0, +.0, +.0, +.0, +.0, +.0, +.0, +.0],  # MBON-α3
            [+.0, +.0, +.0, +.0, +.0, +.0, +.0, +.0],  # MBON-γ5β'2a
            [+.0, +.0, +.0, +.0, +.0, +.0, +.0, +.0],  # awe
            [+.0, +.0, +.0, +.0, +1., +.0, +.0, +.0],  # MBON-α'1
            [+.0, +.0, +.0, +.0, +.0, +.0, +.0, +.0],  # love
        ]) * .5
        # self.w_d2k[:, int(1.5 * self.nb_dan):] *= 1.2
        self.w_d2k[[10, 14], [26, 30]] = 1
        self.names[0], self.names[4], self.names[8], self.names[10], self.names[12], self.names[14] = (
            "PPL1-γ1ped", "PAM-γ4<γ1γ2", "PPL1-γ2α'1", "PAM-α1", "PAM-β'2a", "PAM-α'1")
        self.names[16], self.names[20], self.names[24], self.names[26], self.names[28], self.names[30] = (
            "MBON-γ1ped", "MBON-γ4>γ1γ2", "MBON-γ2α'1", "MBON-α1", "MBON-γ5β'2a", "MBON-α'1")

        w = self._w_m2v + self.w_d2k
        # w = self._w_m2v
        w[:self.nb_dan//2, :] += self.w_u2d[1]
        nids = [0, 4, 8, 10, 12, 14, 16, 20, 24, 26, 28, 30]

        plt.figure("Synaptic weights", figsize=(10, 10))
        plt.imshow(w, vmin=-2, vmax=2, cmap="coolwarm")
        plt.xticks(nids, np.array(self.names)[nids])
        plt.yticks(nids, np.array(self.names)[nids])
        plt.tight_layout()
        plt.grid("both")
        plt.show()


def plot_population(ms, nids=None):
    title = "motivation-" + '-'.join(str(ms[0]).split("'")[1:-1:2])

    nb_models = len(ms)
    xticks = ["%d" % (i+1) for i in range(16)]

    plt.figure(title, figsize=(7.5, 10))

    for i in range(nb_models):
        yticks = np.array(ms[i].names)
        if nids == None:
            nids = np.arange(ms[i].nb_dan + ms[i].nb_mbon)[::8]
        v = ms[i]._v
        w = ms[i].w_k2m

        plt.subplot(nb_models * 2, 2, 1 + i * 4)
        va = v[1:].reshape((-1, 2, 2, 32))[:, ::2].reshape((-1, 32))
        plt.imshow(va.T, vmin=-2., vmax=2., cmap="coolwarm", aspect="auto")
        plt.plot([[15, 17, 19, 21, 23, 25]] * 2, [[0] * 6, [35] * 6], 'r-')
        plt.xticks(2 * np.arange(va.shape[0] // 2), xticks[:va.shape[0] // 2])
        plt.yticks(nids, yticks[nids])
        plt.title("%s - odour A - value" % ms[i].routine_name, color="C%d" % (2 * i + 0))

        plt.subplot(nb_models * 2, 2, 2 + i * 4)
        vb = v[1:].reshape((-1, 2, 2, 32))[:, 1::2].reshape((-1, 32))
        plt.imshow(vb.T, vmin=-2., vmax=2., cmap="coolwarm", aspect="auto")
        plt.plot([[3, 5, 7, 9, 11]] * 2, [[0] * 5, [35] * 5], 'r-')
        plt.xticks(2 * np.arange(vb.shape[0] // 2), xticks[:vb.shape[0] // 2])
        plt.yticks(nids, [""] * len(nids))
        plt.title("%s - odour B - value" % ms[i].routine_name, color="C%d" % (2 * i + 1))

        plt.subplot(nb_models * 2, 2, 3 + i * 4)
        wa = w[5:, 5]
        plt.imshow(wa.T, vmin=-2., vmax=2., cmap="coolwarm", aspect="auto")
        plt.plot([[31, 35, 39, 43, 47, 51]] * 2, [[0] * 6, [35] * 6], 'r-')
        plt.xticks(np.arange(wa.shape[0])[::4], xticks[:wa.shape[0] // 4])
        plt.yticks(nids, yticks[nids])
        plt.title("%s - odour A - weights" % ms[i].routine_name, color="C%d" % (2 * i + 0))

        plt.subplot(nb_models * 2, 2, 4 + i * 4)
        wb = w[5:, 0]
        plt.imshow(wb.T, vmin=-2., vmax=2., cmap="coolwarm", aspect="auto")
        plt.plot([[7, 11, 15, 19, 23]] * 2, [[0] * 5, [35] * 5], 'r-')
        plt.xticks(np.arange(wb.shape[0])[::4], xticks[:wb.shape[0] // 4])
        plt.yticks(nids, [""] * len(nids))
        plt.title("%s - odour B - weights" % ms[i].routine_name, color="C%d" % (2 * i + 1))

    plt.colorbar()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    from evaluation import evaluate, generate_behaviour_map
    target, target_s = generate_behaviour_map(cs_only=True)

    import pandas as pd

    pd.options.display.max_columns = 16
    pd.options.display.max_rows = 16
    pd.options.display.width = 1000

    neurons = [0, 4, 8, 10, 12, 14, 16, 20, 24, 26, 28, 30]
    nb_kcs = 10
    kc1, kc2 = nb_kcs // 2, nb_kcs // 2

    model = MotivationModel(learning_rule="dan-based", nb_apl=0, pn2kc_init="default", verbose=False, third_order=False,
                            timesteps=2, trials=28, nb_kc=nb_kcs, nb_kc_odour_1=kc1, nb_kc_odour_2=kc2)

    val, acc, prediction, models = evaluate(model, nids=neurons, behav_mean=target, behav_std=target_s,
                                            cs_only=True, reversal=True, extinction=True, liyans_frames=False)

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
