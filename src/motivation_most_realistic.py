from modelling import MBModel

import matplotlib.pyplot as plt
import numpy as np


class MotivationModel(MBModel):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("nb_dan", 24)
        kwargs.setdefault("nb_mbon", 24)
        kwargs.setdefault("leak", .0)
        super().__init__(*args, **kwargs)

        has_memory_neurons = False
        p_dan_abs = 0
        p_dan_rel = 8
        p_dan_ltm = 16
        p_mbon_abs = 24
        p_mbon_rel = 32
        p_mbon_ltm = 40

        self._v[:, p_dan_abs:p_dan_rel] = 0.  # A-DANs
        self._v[:, p_dan_rel:p_dan_ltm] = 0.  # R-DANs
        self._v[:, p_dan_ltm:p_mbon_abs] = 0.  # M-DANs
        self._v[:, p_mbon_abs:p_mbon_rel] = 0.  # A-MBONs
        self._v[:, p_mbon_rel:p_mbon_ltm] = 0.  # R-MBONs
        self._v[:, p_mbon_ltm:] = 0.  # M-MBONs
        # self._v[:, 0] = +2.  # PPL1-γ1ped
        # self._v[:, 9] = 1.   # PPL1-γ1β'2a
        # self._v[:, 13] = 1.  # PAM-β'2a

        self.v_init[p_dan_abs:p_dan_rel] = +1.
        self.v_init[p_dan_rel:p_dan_ltm] = +0.
        self.v_init[p_dan_ltm:p_mbon_abs] = 0.
        self.v_init[p_mbon_abs:p_mbon_rel] = -1.
        self.v_init[p_mbon_rel:p_mbon_ltm] = -1.
        self.v_init[p_mbon_ltm:] = -3.

        #     DANs                              MBONs                                  Memory type
        # ------------------------------------------------------------------------------------------
        # 00: PAM-γ4<γ1γ2 (joy)             24: MBON-γ1ped (joy)                          STM
        # 01: hope                          25: hope                                      STM
        # 02: anger                         26: anger                                     STM
        # 03: disgust                       27: disgust                                   STM
        # 04: PPL1-γ1ped (sadness)          28: MBON-γ4>γ1γ2 (sadness)                    STM
        # 05: surprise                      29: surprise                                  STM
        # 06: fear                          30: fear                                      STM
        # 07: trust                         31: trust                                     STM
        # 08: PAM-β'2a (relative joy)       32: MBON-γ2α'1 (relative joy)                 STM
        # 09: relative hope                 33: relative hope                             STM
        # 10: relative anger                34: relative anger                            STM
        # 11: relative disgust              35: relative disgust                          STM
        # 12: PPL1-γ2α'1 (relative sadness) 36: MBON-γ5β'2a (relative sadness)            STM
        # 13: relative surprise             37: relative surprise                         STM
        # 14: relative fear                 38: relative fear                             STM
        # 15: relative trust                39: relative trust                            STM
        # 16: love                          40: love                                      LTM
        # 17: optimism                      41: disapproval                               LTM
        # 18: aggressiveness                42: aggressiveness                            LTM
        # 19: contempt                      43: contempt                                  LTM
        # 20: remorse                       44: remorse                                   LTM
        # 21: PAM-β2 (disapproval)          45: MBON-β2β'2a (optimism)                    LTM
        # 22: awe                           46: awe                                       LTM
        # 23: submission                    47: submission                                LTM
        self._w_m2v = np.zeros((48, 48), dtype=float)
        # Absolute states depress their respective DANs
        self._w_m2v[p_mbon_abs:p_mbon_rel, p_dan_abs:p_dan_rel] = np.array([  # A-MBONs to A-DANs
            [+0., +0., +0., +0., -1., +0., +0., +0.],  # MBON-γ1ped
            [+0., +0., +0., +0., +0., -1., +0., +0.],  # hope
            [+0., +0., +0., +0., +0., +0., -1., +0.],  # anger
            [+0., +0., +0., +0., +0., +0., +0., -1.],  # disgust
            [-1., +0., +0., +0., +0., +0., +0., +0.],  # MBON-γ4>γ1γ2
            [+0., -1., +0., +0., +0., +0., +0., +0.],  # surprise
            [+0., +0., -1., +0., +0., +0., +0., +0.],  # fear
            [+0., +0., +0., -1., +0., +0., +0., +0.],  # trust
        ]) * .5
        # Absolute states depress their opposite relative MBONs
        self._w_m2v[p_mbon_abs:p_mbon_rel, p_mbon_rel:p_mbon_ltm] = np.array([  # A-MBONs to R-MBONs
            [-.0, -.0, -.0, -.0, -1., -.0, -.0, -.0],  # MBON-γ1ped
            [-.0, -.0, -.0, -.0, -.0, -1., -.0, -.0],  # hope
            [-.0, -.0, -.0, -.0, -.0, -.0, -1., -.0],  # anger
            [-.0, -.0, -.0, -.0, -.0, -.0, -.0, -1.],  # disgust
            [-1., -.0, -.0, -.0, -.0, -.0, -.0, -.0],  # MBON-γ4>γ1γ2
            [-.0, -1., -.0, -.0, -.0, -.0, -.0, -.0],  # surprise
            [-.0, -.0, -1., -.0, -.0, -.0, -.0, -.0],  # fear
            [-.0, -.0, -.0, -1., -.0, -.0, -.0, -.0],  # trust
        ]) * .5

        # Relative states depress their respective DANs and enhance their opposite DANs
        self._w_m2v[p_mbon_rel:p_mbon_ltm, p_dan_rel:p_dan_ltm] = np.array([  # R-MBONs to R-DANs
            [+1., +.0, +.0, +.0, -.0, +.0, +.0, +.0],  # MBON-γ2α'1
            [+.0, +1., +.0, +.0, +.0, -.0, +.0, +.0],  # hope
            [+.0, +.0, +1., +.0, +.0, +.0, -.0, +.0],  # anger
            [+.0, +.0, +.0, +1., +.0, +.0, +.0, -.0],  # disgust
            [-.0, +.0, +.0, +.0, +1., +.0, +.0, +.0],  # MBON-γ5β'2a
            [+.0, -.0, +.0, +.0, +.0, +1., +.0, +.0],  # surprise
            [+.0, +.0, -.0, +.0, +.0, +.0, +1., +.0],  # fear
            [+.0, +.0, +.0, -.0, +.0, +.0, +.0, +1.],  # trust
        ])

        if has_memory_neurons:
            # Relative states enhance their memory DANs
            self._w_m2v[p_mbon_rel:p_mbon_ltm, p_dan_ltm:p_mbon_abs] = np.array([  # R-MBONs to M-DANs
                [+1., +.0, +.0, +.0, -.0, +.0, +.0, +.0],  # MBON-γ2α'1
                [+.0, +1., +.0, +.0, +.0, +.0, +.0, +.0],  # hope
                [+.0, +.0, +1., +.0, +.0, +.0, +.0, +.0],  # anger
                [+.0, +.0, +.0, +1., +.0, +.0, +.0, +.0],  # disgust
                [-.0, +.0, +.0, +.0, +1., +.0, +.0, +.0],  # MBON-γ5β'2a
                [+.0, +.0, +.0, +.0, +.0, +1., +.0, +.0],  # surprise
                [+.0, +.0, +.0, +.0, +.0, +.0, +1., +.0],  # fear
                [+.0, +.0, +.0, +.0, +.0, +.0, +.0, +1.],  # trust
            ]) * .5

            # Memory states enhance their respective relative DANs
            self._w_m2v[p_mbon_ltm:, p_dan_rel:p_dan_ltm] = np.array([  # M-MBONs to R-DANs
                [+1., +.0, +.0, +.0, -.0, +.0, +.0, +.0],  # MBON-γ2α'1
                [+.0, +1., +.0, +.0, +.0, +.0, +.0, +.0],  # hope
                [+.0, +.0, +1., +.0, +.0, +.0, +.0, +.0],  # anger
                [+.0, +.0, +.0, +1., +.0, +.0, +.0, +.0],  # disgust
                [-.0, +.0, +.0, +.0, +1., +.0, +.0, +.0],  # MBON-γ5β'2a
                [+.0, +.0, +.0, +.0, +.0, +1., +.0, +.0],  # surprise
                [+.0, +.0, +.0, +.0, +.0, +.0, +1., +.0],  # fear
                [+.0, +.0, +.0, +.0, +.0, +.0, +.0, +1.],  # trust
            ]) * .1
            # Memory states depress their adjacent memory DANs
            self._w_m2v[p_mbon_ltm:, p_dan_ltm:p_mbon_abs] = np.array([  # M-MBONs to M-DANs
                [+.0, +.0, +.0, -1., +.0, -1., +.0, +.0],  # MBON-γ2α'1
                [+.0, +.0, +.0, +.0, -1., +.0, -1., +.0],  # hope
                [+.0, +.0, +.0, +.0, +.0, -1., +.0, -1.],  # anger
                [-1., +.0, +.0, +.0, +.0, +.0, -1., +.0],  # disgust
                [+.0, -1., +.0, +.0, +.0, +.0, +.0, -1.],  # MBON-γ5β'2a
                [-1., +.0, -1., +.0, +.0, +.0, +.0, +.0],  # surprise
                [+.0, -1., +.0, -1., +.0, +.0, +.0, +.0],  # fear
                [+.0, +.0, -1., +.0, -1., +.0, +.0, +.0],  # trust
            ]) * .2

        self._w_d2k = np.zeros((self.nb_dan + self.nb_mbon, self.nb_dan + self.nb_mbon), dtype=float)
        # Absolute DANs depress their opposite MBONs
        self._w_d2k[p_dan_abs:p_dan_rel, p_mbon_abs:p_mbon_rel] = -np.array([
            [float(m == (d + ((p_dan_rel-p_dan_abs) // 2)) % (p_dan_rel-p_dan_abs))
             for m in range(p_mbon_rel-p_mbon_abs)]
            for d in range(p_dan_rel-p_dan_abs)
        ])
        # Relative DANs depress their opposite MBONs
        self._w_d2k[p_dan_rel:p_dan_ltm, p_mbon_rel:p_mbon_ltm] = -np.array([
            [float(m == (d + ((p_dan_ltm-p_dan_rel) // 2)) % (p_dan_ltm-p_dan_rel))
             for m in range(p_mbon_ltm-p_mbon_rel)]
            for d in range(p_dan_ltm-p_dan_rel)
        ]) * 1.

        if has_memory_neurons:
            # Relative DANs enhance their respective memory MBONs
            self._w_d2k[p_dan_rel:p_dan_ltm, p_mbon_ltm:] = np.array([
                [float(m == d)
                 for m in range(self._w_d2k.shape[1]-p_mbon_ltm)]
                for d in range(p_dan_ltm-p_dan_rel)
            ]) * .05
            # Memory DANs depress their opposite memory MBONs
            self._w_d2k[p_dan_ltm:p_mbon_abs, p_mbon_ltm:] += -np.array([
                [float(m == (d + ((p_mbon_abs-p_dan_ltm) // 2) - 1) % (p_mbon_abs-p_dan_ltm))
                 for m in range(self._w_d2k.shape[1]-p_mbon_ltm)]
                for d in range(p_dan_ltm-p_dan_rel)
            ]) * .05
            self._w_d2k[p_dan_ltm:p_mbon_abs, p_mbon_ltm:] += -np.array([
                [float(m == (d + ((p_mbon_abs-p_dan_ltm) // 2) + 1) % (p_mbon_abs-p_dan_ltm))
                 for m in range(self._w_d2k.shape[1]-p_mbon_ltm)]
                for d in range(p_dan_ltm-p_dan_rel)
            ]) * .05
            # Memory DANs enhance their respective relative MBONs
            self._w_d2k[p_dan_ltm:p_mbon_abs, p_mbon_rel:p_mbon_ltm] = np.array([
                [float(m == d)
                 for m in range(p_mbon_ltm-p_mbon_rel)]
                for d in range(p_mbon_abs-p_dan_ltm)
            ]) * .05
        # self.w_d2k[:, int(1.5 * self.nb_dan):] *= 1.2
        # self.w_d2k[[10, 14], [26, 30]] = 1

        u = np.zeros((self.nb_dan // 3, self.nb_dan + self.nb_mbon), dtype=float)
        u[:, :self.nb_dan] = np.eye(self.nb_dan // 3, self.nb_dan) * 2
        u[:, self.nb_dan//3:self.nb_dan] = np.eye(self.nb_dan // 3, self.nb_dan - self.nb_dan // 3) * 2
        self.w_u2d = np.array(u)

        self.names[0], self.names[4], self.names[8], self.names[12], self.names[16], self.names[20] = (
            "PAM-γ4<γ1γ2", "PPL1-γ1ped", "PAM-β'2a", "PPL1-γ2α'1", "like", "dislike")
        self.names[24], self.names[28], self.names[32], self.names[36], self.names[40], self.names[44] = (
            "MBON-γ1ped", "MBON-γ4>γ1γ2", "MBON-γ2α'1", "MBON-γ5β'2a", "like", "dislike")

        w = self._w_m2v + self.w_d2k
        # w = self._w_m2v
        w[:self.nb_dan//3, :] += self.w_u2d
        nids = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44]

        plt.figure("Synaptic weights", figsize=(10, 10))
        plt.imshow(w, vmin=-2, vmax=2, cmap="coolwarm")
        plt.xticks(nids, np.array(self.names)[nids])
        plt.yticks(nids, np.array(self.names)[nids])
        plt.tight_layout()
        plt.grid("both")
        # plt.show()


def plot_population(ms, nids=None, vmin=-2., vmax=2.):
    title = "motivation-" + '-'.join(str(ms[0]).split("'")[1:-1:2])

    nb_models = len(ms)
    xticks = ["%d" % (i+1) for i in range(16)]

    plt.figure(title, figsize=(7.5, 10))

    for i in range(nb_models):
        nb_timesteps = ms[i].nb_timesteps
        nb_trials = ms[i].nb_trials

        yticks = np.array(ms[i].names)
        if nids == None:
            nids = np.arange(ms[i].nb_dan + ms[i].nb_mbon)[::8]
        v = ms[i]._v
        w = ms[i].w_k2m

        plt.subplot(nb_models * 2, 2, 1 + i * 4)
        # trial, odour, time-step, neuron
        va = v[1:].reshape((-1, 2, 2, v.shape[-1]))[:, ::2].reshape((-1, v.shape[-1]))
        plt.imshow(va.T, vmin=vmin, vmax=vmax, cmap="coolwarm", aspect="auto")
        plt.plot([[15, 17, 19, 21, 23, 25]] * 2, [[0] * 6, [47] * 6], 'r-')
        plt.xticks(nb_timesteps * np.arange(nb_trials // 2) + nb_timesteps / 4, xticks[:nb_trials // 2])
        plt.yticks(nids, yticks[nids])
        plt.title("%s - odour A - value" % ms[i].routine_name, color="C%d" % (2 * i + 0))

        plt.subplot(nb_models * 2, 2, 2 + i * 4)
        vb = v[1:].reshape((-1, 2, 2, v.shape[-1]))[:, 1::2].reshape((-1, v.shape[-1]))
        plt.imshow(vb.T, vmin=vmin, vmax=vmax, cmap="coolwarm", aspect="auto")
        plt.plot([[3, 5, 7, 9, 11]] * 2, [[0] * 5, [47] * 5], 'r-')
        plt.xticks(nb_timesteps * np.arange(nb_trials // 2) + nb_timesteps / 4, xticks[:nb_trials // 2])
        plt.yticks(nids, [""] * len(nids))
        plt.title("%s - odour B - value" % ms[i].routine_name, color="C%d" % (2 * i + 1))

        plt.subplot(nb_models * 2, 2, 3 + i * 4)
        wa = w[1:, 0]
        plt.imshow(wa.T, vmin=vmin, vmax=vmax, cmap="coolwarm", aspect="auto")
        plt.plot([[31, 35, 39, 43, 47, 51]] * 2, [[0] * 6, [47] * 6], 'r-')
        plt.xticks(2 * nb_timesteps * np.arange(nb_trials // 2) + nb_timesteps, xticks[:nb_trials // 2])
        plt.yticks(nids, yticks[nids])
        plt.title("%s - odour A - weights" % ms[i].routine_name, color="C%d" % (2 * i + 0))

        plt.subplot(nb_models * 2, 2, 4 + i * 4)
        wb = w[1:, 5]
        plt.imshow(wb.T, vmin=vmin, vmax=vmax, cmap="coolwarm", aspect="auto")
        plt.plot([[7, 11, 15, 19, 23]] * 2, [[0] * 5, [47] * 5], 'r-')
        plt.xticks(2 * nb_timesteps * np.arange(nb_trials // 2) + nb_timesteps, xticks[:nb_trials // 2])
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

    neurons = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44]
    nb_kcs = 10
    kc1, kc2 = nb_kcs // 2, nb_kcs // 2

    model = MotivationModel(learning_rule="dan-based", nb_apl=0, pn2kc_init="default", verbose=False,
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
