from modelling import MBModel

import matplotlib.pyplot as plt
import numpy as np


class MotivationModel(MBModel):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("nb_dan", 16)
        kwargs.setdefault("nb_mbon", 16)
        kwargs.setdefault("leak", .0)
        super().__init__(*args, **kwargs)

        has_ltm = True

        p_dan_abs_s = 0
        p_dan_abs_e = 8
        p_dan_rel_s = 8
        p_dan_rel_e = 16
        p_dan_ltm_s = 8
        p_dan_ltm_e = 16
        p_mbon_abs_s = 16
        p_mbon_abs_e = 24
        p_mbon_rel_s = 24
        p_mbon_rel_e = 32
        p_mbon_ltm_s = 24
        p_mbon_ltm_e = 32

        self._v[:, p_dan_abs_s:p_dan_abs_e] = 0.  # A-DANs
        self._v[:, p_dan_rel_s:p_dan_rel_e] = -1.  # R-DANs
        # self._v[:, p_dan_ltm_s:p_dan_ltm_e] = 0.  # M-DANs
        self._v[:, p_mbon_abs_s:p_mbon_abs_e] = 0.  # A-MBONs
        self._v[:, p_mbon_rel_s:p_mbon_rel_e] = 0.  # R-MBONs
        # self._v[:, p_mbon_ltm_s:p_mbon_ltm_e] = -4.  # M-MBONs
        # self._v[:, 0] = +2.  # PPL1-γ1ped
        # self._v[:, 9] = 1.   # PPL1-γ1β'2a
        # self._v[:, 13] = 1.  # PAM-β'2a

        self.v_init[p_dan_abs_s:p_dan_abs_e] = +1.
        self.v_init[p_dan_rel_s:p_dan_rel_e] = -0.
        self.v_init[p_dan_ltm_s:p_dan_ltm_e] = 0.
        self.v_init[p_mbon_abs_s:p_mbon_abs_e] = -1.
        self.v_init[p_mbon_rel_s:p_mbon_rel_e] = -0.
        self.v_init[p_mbon_ltm_s:p_mbon_ltm_e] = -2.

        #     DANs                              MBONs                                  Memory type
        # ------------------------------------------------------------------------------------------
        # 00: PAM-γ4<γ1γ2 (joy)             16: MBON-γ1ped (joy)                          STM
        # 01: hope                          17: hope                                      STM
        # 02: anger                         18: anger                                     STM
        # 03: disgust                       19: disgust                                   STM
        # 04: PPL1-γ1ped (sadness)          20: MBON-γ4>γ1γ2 (sadness)                    STM
        # 05: surprise                      21: surprise                                  STM
        # 06: fear                          22: fear                                      STM
        # 07: trust                         23: trust                                     STM
        # 08: PAM-β'2a (relative joy)       24: MBON-γ2α'1 (relative joy)                 STM
        # 09: relative hope                 25: relative hope                             STM
        # 10: relative anger                26: relative anger                            STM
        # 11: relative disgust              27: relative disgust                          STM
        # 12: PPL1-γ2α'1 (relative sadness) 28: MBON-γ5β'2a (relative sadness)            STM
        # 13: relative surprise             29: relative surprise                         STM
        # 14: relative fear                 30: relative fear                             STM
        # 15: relative trust                31: relative trust                            STM
        self._w_m2v = np.zeros((self.nb_mbon + self.nb_dan, self.nb_mbon + self.nb_dan), dtype=float)
        # Absolute states depress their respective DANs
        self._w_m2v[p_mbon_abs_s:p_mbon_abs_e, p_dan_abs_s:p_dan_abs_e] = np.array([  # A-MBONs to A-DANs
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
        self._w_m2v[p_mbon_abs_s:p_mbon_abs_e, p_mbon_rel_s:p_mbon_rel_e] = np.array([  # A-MBONs to R-MBONs
            [-.0, -.0, -.0, -.0, -1., -.0, -.0, -.0],  # MBON-γ1ped (joy)
            [-.0, -.0, -.0, -.0, -.0, -1., -.0, -.0],  # hope
            [-.0, -.0, -.0, -.0, -.0, -.0, -1., -.0],  # anger
            [-.0, -.0, -.0, -.0, -.0, -.0, -.0, -1.],  # disgust
            [-1., -.0, -.0, -.0, -.0, -.0, -.0, -.0],  # MBON-γ4>γ1γ2 (sadness)
            [-.0, -1., -.0, -.0, -.0, -.0, -.0, -.0],  # surprise
            [-.0, -.0, -1., -.0, -.0, -.0, -.0, -.0],  # fear
            [-.0, -.0, -.0, -1., -.0, -.0, -.0, -.0],  # trust
        ]) * .5

        # Relative states enhance their opposite DANs
        self._w_m2v[p_mbon_rel_s:p_mbon_rel_e, p_dan_rel_s:p_dan_rel_e] = np.array([  # R-MBONs to R-DANs
            [+1., +.0, +.0, +.0, -.0, +.0, +.0, +.0],  # MBON-γ2α'1 (joy)
            [+.0, +1., +.0, +.0, +.0, -.0, +.0, +.0],  # hope
            [+.0, +.0, +1., +.0, +.0, +.0, -.0, +.0],  # anger
            [+.0, +.0, +.0, +1., +.0, +.0, +.0, -.0],  # disgust
            [-.0, +.0, +.0, +.0, +1., +.0, +.0, +.0],  # MBON-γ5β'2a (sadness)
            [+.0, -.0, +.0, +.0, +.0, +1., +.0, +.0],  # surprise
            [+.0, +.0, -.0, +.0, +.0, +.0, +1., +.0],  # fear
            [+.0, +.0, +.0, -.0, +.0, +.0, +.0, +1.],  # trust
        ])

        if has_ltm:
            # Memory states enhance their memory DANs
            self._w_m2v[p_mbon_ltm_s:p_mbon_ltm_e, p_dan_ltm_s:p_dan_ltm_e] += np.array([  # M-MBONs to M-DANs
                [+.0, +.0, +.0, +1., +.0, +.0, +.0, +.0],  # MBON-γ2α'1
                [+.0, +.0, +.0, +.0, +1., +.0, +.0, +.0],  # hope
                [+.0, +.0, +.0, +.0, +.0, +1., +.0, +.0],  # anger
                [+.0, +.0, +.0, +.0, +.0, +.0, +1., +.0],  # disgust
                [+.0, +.0, +.0, +.0, +.0, +.0, +.0, +1.],  # MBON-γ5β'2a
                [+1., +.0, +.0, +.0, +.0, +.0, +.0, +.0],  # surprise
                [+.0, +1., +.0, +.0, +.0, +.0, +.0, +.0],  # fear
                [+.0, +.0, +1., +.0, +.0, +.0, +.0, +.0],  # trust
            ]) * .5
            # Memory states enhance their respective relative DANs
            self._w_m2v[p_mbon_ltm_s:p_mbon_ltm_e, p_dan_rel_s:p_dan_rel_e] += np.array([  # M-MBONs to R-DANs
                [+.0, +.0, +.0, +.0, +.0, +.0, +.0, +1.],  # MBON-γ2α'1
                [+1., +.0, +.0, +.0, +.0, +.0, +.0, +.0],  # hope
                [+.0, +1., +.0, +.0, +.0, +.0, +.0, +.0],  # anger
                [+.0, +.0, +1., +.0, +.0, +.0, +.0, +.0],  # disgust
                [+.0, +.0, +.0, +1., +.0, +.0, +.0, +.0],  # MBON-γ5β'2a
                [+.0, +.0, +.0, +.0, +1., +.0, +.0, +.0],  # surprise
                [+.0, +.0, +.0, +.0, +.0, +1., +.0, +.0],  # fear
                [+.0, +.0, +.0, +.0, +.0, +.0, +1., +.0],  # trust
            ]) * .5

        self._w_d2k = np.zeros((self.nb_dan + self.nb_mbon, self.nb_dan + self.nb_mbon), dtype=float)
        # Absolute DANs depress their opposite MBONs
        self._w_d2k[p_dan_abs_s:p_dan_abs_e, p_mbon_abs_s:p_mbon_abs_e] += -np.array([
            [float(m == (d + ((p_dan_abs_e-p_dan_abs_s) // 2)) % (p_dan_abs_e-p_dan_abs_s))
             for m in range(p_mbon_abs_e-p_mbon_abs_s)]
            for d in range(p_dan_abs_e-p_dan_abs_s)
        ])
        # Relative DANs depress their opposite MBONs
        self._w_d2k[p_dan_rel_s:p_dan_rel_e, p_mbon_rel_s:p_mbon_rel_e] += -np.array([
            [float(m == (d + ((p_dan_rel_e-p_dan_rel_s) // 2)) % (p_dan_rel_e-p_dan_rel_s))
             for m in range(p_mbon_rel_e-p_mbon_rel_s)]
            for d in range(p_dan_rel_e-p_dan_rel_s)
        ]) * 1.

        if has_ltm:
            # Relative DANs enhance their respective memory MBONs
            self._w_d2k[p_dan_rel_s:p_dan_rel_e, p_mbon_ltm_s:p_mbon_ltm_e] += np.array([
                [float(m == (d + ((p_dan_rel_e-p_dan_rel_s) // 2) + 1) % (p_dan_rel_e-p_dan_rel_s))
                 for m in range(p_mbon_ltm_e-p_mbon_ltm_s)]
                for d in range(p_dan_rel_e-p_dan_rel_s)
            ]) * .1 * np.array([0., 1., 0., 0., 0., 1., 0., 0.])

        u = np.zeros((self.nb_dan // 2, self.nb_dan + self.nb_mbon), dtype=float)
        u[:, :self.nb_dan] = np.eye(self.nb_dan // 2, self.nb_dan) * 2
        u[:, self.nb_dan//2:self.nb_dan] = np.eye(self.nb_dan // 2, self.nb_dan - self.nb_dan // 2) * 2
        self.w_u2d = np.array(u)

        self.names[0], self.names[4], self.names[8], self.names[12] = (
            "PAM-γ4<γ1γ2", "PPL1-γ1ped", "PAM-β'2a", "PPL1-γ2α'1")
        self.names[16], self.names[20], self.names[24], self.names[28] = (
            "MBON-γ1ped", "MBON-γ4>γ1γ2", "MBON-γ2α'1", "MBON-γ5β'2a")
        # self.names[0], self.names[4], self.names[8], self.names[12] = (
        #     "DAN-yellow-out", "DAN-blue-out", "DAN-yellow-in", "DAN-blue-in")
        # self.names[16], self.names[20], self.names[24], self.names[28] = (
        #     "MBON-yellow-out", "MBON-blue-out", "MBON-yellow-in", "MBON-blue-in")

        w = self._w_m2v + self.w_d2k
        # w = self._w_m2v
        # w[:self.nb_dan//2, :] += self.w_u2d
        nids = [0, 4, 8, 12, 16, 20, 24, 28]

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
        ylim = ms[i].nb_dan + ms[i].nb_mbon - 1

        yticks = np.array(ms[i].names)
        if nids == None:
            nids = np.arange(ms[i].nb_dan + ms[i].nb_mbon)[::8]
        v = ms[i]._v
        w = ms[i].w_k2m

        plt.subplot(nb_models * 2, 2, 1 + i * 4)
        # trial, odour, time-step, neuron
        va = v[1:].reshape((-1, 2, 2, v.shape[-1]))[:, ::2].reshape((-1, v.shape[-1]))
        plt.imshow(va.T, vmin=vmin, vmax=vmax, cmap="coolwarm", aspect="auto")
        plt.plot([[15, 17, 19, 21, 23, 25]] * 2, [[0] * 6, [ylim] * 6], 'r-')
        plt.xticks(nb_timesteps * np.arange(nb_trials // 2) + nb_timesteps / 4, xticks[:nb_trials // 2])
        plt.yticks(nids, yticks[nids])
        plt.title("%s - odour A - value" % ms[i].routine_name, color="C%d" % (2 * i + 0))

        plt.subplot(nb_models * 2, 2, 2 + i * 4)
        vb = v[1:].reshape((-1, 2, 2, v.shape[-1]))[:, 1::2].reshape((-1, v.shape[-1]))
        plt.imshow(vb.T, vmin=vmin, vmax=vmax, cmap="coolwarm", aspect="auto")
        plt.plot([[3, 5, 7, 9, 11]] * 2, [[0] * 5, [ylim] * 5], 'r-')
        plt.xticks(nb_timesteps * np.arange(nb_trials // 2) + nb_timesteps / 4, xticks[:nb_trials // 2])
        plt.yticks(nids, [""] * len(nids))
        plt.title("%s - odour B - value" % ms[i].routine_name, color="C%d" % (2 * i + 1))

        plt.subplot(nb_models * 2, 2, 3 + i * 4)
        wa = w[1:, 0]
        plt.imshow(wa.T, vmin=vmin, vmax=vmax, cmap="coolwarm", aspect="auto")
        plt.plot([[31, 35, 39, 43, 47, 51]] * 2, [[0] * 6, [ylim] * 6], 'r-')
        plt.xticks(2 * nb_timesteps * np.arange(nb_trials // 2) + nb_timesteps, xticks[:nb_trials // 2])
        plt.yticks(nids, yticks[nids])
        plt.title("%s - odour A - weights" % ms[i].routine_name, color="C%d" % (2 * i + 0))

        plt.subplot(nb_models * 2, 2, 4 + i * 4)
        wb = w[1:, 5]
        plt.imshow(wb.T, vmin=vmin, vmax=vmax, cmap="coolwarm", aspect="auto")
        plt.plot([[7, 11, 15, 19, 23]] * 2, [[0] * 5, [ylim] * 5], 'r-')
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

    get_score = True

    neurons = [0, 4, 8, 12, 16, 20, 24, 28]
    nb_kcs = 10
    kc1, kc2 = nb_kcs // 2, nb_kcs // 2

    model = MotivationModel(learning_rule="dan-based", nb_apl=0, pn2kc_init="default", verbose=False,
                            timesteps=2, trials=28, nb_kc=nb_kcs, nb_kc_odour_1=kc1, nb_kc_odour_2=kc2)

    if get_score:
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
    else:
        val, acc, prediction, models = evaluate(model, behav_mean=pd.DataFrame({}),
                                                nids=neurons, cs_only=True, reversal=True, extinction=True)

    # MBModel.plot_overlap(models, nids=neurons, score=acc)
    # MBModel.plot_timeline(models=models, nids=neurons, score=acc, target=target, nb_trials=13)
    plot_population(models, neurons)
