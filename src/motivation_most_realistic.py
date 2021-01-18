from modelling import MBModel

import matplotlib.pyplot as plt
import numpy as np


class MotivationModel(MBModel):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("nb_dan", 16)
        kwargs.setdefault("nb_mbon", 16)
        kwargs.setdefault("leak", .0)
        has_fom = kwargs.pop("has_fom", True)
        has_bm = kwargs.pop("has_bm", True)
        has_rsom = kwargs.pop("has_rsom", True)
        has_ltm = kwargs.pop("has_ltm", True)
        has_mdm = kwargs.pop("has_mdm", True)
        is_single = kwargs.pop("is_single", False)
        has_real_names = kwargs.pop("has_real_names", False)
        super().__init__(*args, **kwargs)
        pars = []
        for a, b in [("FOM", has_fom), ("BM", has_bm), ("RSOM", has_rsom), ("LTM", has_ltm), ("MDM", has_mdm)]:
            if b:
                pars += [a]
        print(*pars)

        shock_magnitude = 2.
        odour_magnitude = 2.
        p_dan_abs_s, p_dan_abs_e = 0, 8
        p_dan_stm_s, p_dan_stm_e = 8, 16
        p_dan_ltm_s, p_dan_ltm_e = 8, 16
        p_mbon_abs_s, p_mbon_abs_e = 16, 24
        p_mbon_stm_s, p_mbon_stm_e = 24, 32
        p_mbon_ltm_s, p_mbon_ltm_e = 24, 32

        self.w_p2k *= odour_magnitude

        self._v[:, p_dan_abs_s:p_dan_abs_e] = self.v_init[p_dan_abs_s:p_dan_abs_e] = -0.5  # D-DANs
        self._v[:, p_dan_stm_s:p_dan_stm_e] = self.v_init[p_dan_stm_s:p_dan_stm_e] = -0.5  # R-DANs
        self._v[:, p_dan_ltm_s:p_dan_ltm_e] = self.v_init[p_dan_ltm_s:p_dan_ltm_e] = -0.5  # F-DANs
        self._v[:, p_mbon_abs_s:p_mbon_abs_e] = self.v_init[p_mbon_abs_s:p_mbon_abs_e] = -2.  # A-MBONs
        self._v[:, p_mbon_stm_s:p_mbon_stm_e] = self.v_init[p_mbon_stm_s:p_mbon_stm_e] = -1.  # H-MBONs
        self._v[:, p_mbon_ltm_s:p_mbon_ltm_e] = self.v_init[p_mbon_ltm_s:p_mbon_ltm_e] = -1.  # M-MBONs
        # self._v[:, 0] = +2.  # PPL1-γ1ped
        # self._v[:, 9] = 1.   # PPL1-γ1β'2a
        # self._v[:, 13] = 1.  # PAM-β'2a

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
        self._w_d2k = np.zeros((self.nb_dan + self.nb_mbon, self.nb_dan + self.nb_mbon), dtype=float)

        # first order memory (FOM) subcircuit
        if has_fom:
            # Absolute states depress their respective DANs
            self._w_m2v[p_mbon_abs_s:p_mbon_abs_e, p_dan_abs_s:p_dan_abs_e] = np.array([  # A-MBONs to D-DANs
                [+0., +0., +0., +0., -1., +0., +0., +0.],  # MBON-γ1ped
                [+0., +0., +0., +0., +0., -1., +0., +0.],  # hope
                [+0., +0., +0., +0., +0., +0., -1., +0.],  # anger
                [+0., +0., +0., +0., +0., +0., +0., -1.],  # disgust
                [-1., +0., +0., +0., +0., +0., +0., +0.],  # MBON-γ4>γ1γ2
                [+0., -1., +0., +0., +0., +0., +0., +0.],  # surprise
                [+0., +0., -1., +0., +0., +0., +0., +0.],  # fear
                [+0., +0., +0., -1., +0., +0., +0., +0.],  # trust
            ]) * .3
            # Absolute DANs depress their opposite MBONs
            self._w_d2k[p_dan_abs_s:p_dan_abs_e, p_mbon_abs_s:p_mbon_abs_e] += -np.array([
                [float(m == (d + ((p_dan_abs_e-p_dan_abs_s) // 2)) % (p_dan_abs_e-p_dan_abs_s))
                 for m in range(p_mbon_abs_e-p_mbon_abs_s)]
                for d in range(p_dan_abs_e-p_dan_abs_s)
            ])

        # blocking memory (BM) subcircuit
        if has_bm:
            # Absolute states depress their opposite relative MBONs
            self._w_m2v[p_mbon_abs_s:p_mbon_abs_e, p_mbon_stm_s:p_mbon_stm_e] = np.array([  # A-MBONs to H-MBONs
                [-.0, -.0, -.0, -.0, -1., -.0, -.0, -.0],  # MBON-γ1ped (joy)
                [-.0, -.0, -.0, -.0, -.0, -1., -.0, -.0],  # hope
                [-.0, -.0, -.0, -.0, -.0, -.0, -1., -.0],  # anger
                [-.0, -.0, -.0, -.0, -.0, -.0, -.0, -1.],  # disgust
                [-1., -.0, -.0, -.0, -.0, -.0, -.0, -.0],  # MBON-γ4>γ1γ2 (sadness)
                [-.0, -1., -.0, -.0, -.0, -.0, -.0, -.0],  # surprise
                [-.0, -.0, -1., -.0, -.0, -.0, -.0, -.0],  # fear
                [-.0, -.0, -.0, -1., -.0, -.0, -.0, -.0],  # trust
            ]) * .3

        # reciprocal second order memories (RSOM) subcircuit
        if has_rsom:
            # Relative states enhance their opposite DANs
            self._w_m2v[p_mbon_stm_s:p_mbon_stm_e, p_dan_stm_s:p_dan_stm_e] = np.array([  # H-MBONs to R-DANs
                [+1., +.0, +.0, +.0, -.0, +.0, +.0, +.0],  # MBON-γ2α'1 (joy)
                [+.0, +1., +.0, +.0, +.0, -.0, +.0, +.0],  # hope
                [+.0, +.0, +1., +.0, +.0, +.0, -.0, +.0],  # anger
                [+.0, +.0, +.0, +1., +.0, +.0, +.0, -.0],  # disgust
                [-.0, +.0, +.0, +.0, +1., +.0, +.0, +.0],  # MBON-γ5β'2a (sadness)
                [+.0, -.0, +.0, +.0, +.0, +1., +.0, +.0],  # surprise
                [+.0, +.0, -.0, +.0, +.0, +.0, +1., +.0],  # fear
                [+.0, +.0, +.0, -.0, +.0, +.0, +.0, +1.],  # trust
            ]) * 2.

            # Relative DANs depress their opposite MBONs
            self._w_d2k[p_dan_stm_s:p_dan_stm_e, p_mbon_stm_s:p_mbon_stm_e] += -np.array([
                [float(m == (d + ((p_dan_stm_e-p_dan_stm_s) // 2)) % (p_dan_stm_e-p_dan_stm_s))
                 for m in range(p_mbon_stm_e-p_mbon_stm_s)]
                for d in range(p_dan_stm_e-p_dan_stm_s)
            ]) * 1.

        if has_ltm:
            # Long-term memory (LTM) subcircuit
            self._w_m2v[p_mbon_ltm_s:p_mbon_ltm_e, p_dan_stm_s:p_dan_stm_e] += np.array([  # M-MBONs to R-DANs
                [+.0, +.0, +.0, +1.0, +.0, +.0, +.0, +.0],  # MBON-γ2α'1
                [+.0, +.0, +.0, +.0, +1.0, +.0, +.0, +.0],  # hope
                [+.0, +.0, +.0, +.0, +.0, +1.0, +.0, +.0],  # anger
                [+.0, +.0, +.0, +.0, +.0, +.0, +1.0, +.0],  # disgust
                [+.0, +.0, +.0, +.0, +.0, +.0, +.0, +1.0],  # MBON-γ5β'2a
                [+1.0, +.0, +.0, +.0, +.0, +.0, +.0, +.0],  # surprise
                [+.0, +1.0, +.0, +.0, +.0, +.0, +.0, +.0],  # fear
                [+.0, +.0, +1.0, +.0, +.0, +.0, +.0, +.0],  # trust
            ]) * .05
            # Relative DANs enhance their respective memory MBONs
            self._w_d2k[p_dan_stm_s:p_dan_stm_e, p_mbon_ltm_s:p_mbon_ltm_e] += np.array([
                [float(m == (d + ((p_dan_stm_e-p_dan_stm_s) // 2) + 1) % (p_dan_stm_e-p_dan_stm_s))
                 for m in range(p_mbon_ltm_e-p_mbon_ltm_s)]
                for d in range(p_dan_stm_e-p_dan_stm_s)
            ]) * .05 * (np.array([0., 1., 0., 0., 0., 1., 0., 0.]) if is_single else 1.)

        # Memory digestion mechanism (MDM)
        if has_mdm:
            self._w_d2k[p_dan_ltm_s:p_dan_ltm_e, p_mbon_stm_s:p_mbon_stm_e] += -np.array([
                [float(m == (d + ((p_dan_stm_e-p_dan_stm_s) // 2) - 1) % (p_dan_stm_e-p_dan_stm_s))
                 for m in range(p_mbon_ltm_e-p_mbon_ltm_s)]
                for d in range(p_dan_stm_e-p_dan_stm_s)
            ]) * .05 * (np.array([[1., 0., 0., 0., 1., 0., 0., 0.]]) if is_single else 1.)

        u = np.zeros((self.nb_dan // 2, self.nb_dan + self.nb_mbon), dtype=float)
        u[:, p_dan_abs_s:p_dan_abs_e] = np.eye(p_dan_abs_e-p_dan_abs_s) * shock_magnitude
        u[:, p_dan_stm_s:p_dan_stm_e] = np.eye(p_dan_stm_e-p_dan_stm_s) * shock_magnitude
        self.w_u2d = np.array(u)

        if has_real_names:
            self.names[0], self.names[4], self.names[8], self.names[12], self.names[13], self.names[9] = (
                "PAM-γ4<γ1γ2", "PPL1-γ1ped", "PAM-β'2a", "PPL1-γ2α'1", "PPL1-α'2α2", "PPL1-γ5")
            self.names[16], self.names[20], self.names[24], self.names[28], self.names[29], self.names[25] = (
                "MBON-γ1ped", "MBON-γ4>γ1γ2", "MBON-γ2α'1", "MBON-γ5β'2a", "MBON-β2β'2a", "MBON-α'1")
        else:
            self.names[0], self.names[4], self.names[8], self.names[12], self.names[13], self.names[9] = (
                r"d_{attract}", r"d_{avoid}", r"r_{attract}", r"r_{avoid}", r"f_{attract}", r"f_{avoid}")
            self.names[16], self.names[20], self.names[24], self.names[28], self.names[29], self.names[25] = (
                r"a_{attract}", r"a_{avoid}", r"h_{attract}", r"h_{avoid}", r"m_{attract}", r"m_{avoid}")

        self.neuron_ids = [0, 4, 16, 20, 8, 12, 24, 28, 13, 9, 29, 25]
        # self.neuron_ids = [4, 16, 8, 12, 24, 28]

        # w = self._w_m2v + self.w_d2k
        # # w = self._w_m2v
        # # w[:self.nb_dan//2, :] += self.w_u2d
        # plt.figure("Synaptic weights", figsize=(10, 10))
        # plt.imshow(w, vmin=-2, vmax=2, cmap="coolwarm")
        # plt.xticks(self.neuron_ids, [r'$%s$' % tick for tick in np.array(self.names)[self.neuron_ids]])
        # plt.yticks(self.neuron_ids, [r'$%s$' % tick for tick in np.array(self.names)[self.neuron_ids]])
        # plt.tight_layout()
        # plt.colorbar()
        # plt.grid("both")
        # plt.show()


def plot_population(ms, nids=None, vmin=-2., vmax=2., only_nids=False):
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


def plot_individuals(ms, nids=None, only_nids=True):
    title = "individuals-" + '-'.join(str(ms[0]).split("'")[1:-1:2])

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

        v = ms[i]._v

        # trial, odour, time-step, neuron
        va = v[1:].reshape((-1, nb_odours, nb_timesteps, v.shape[-1]))[:, 0].reshape((-1, v.shape[-1]))
        if only_nids:
            va = va[:, nids]

        names = np.array(ms[i].names)[nids]
        nb_neurons = va.shape[1]
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
                axa.set_xticks((nb_timesteps - 1) * np.arange(nb_trials // 4 - 1) + (nb_timesteps - 1) / 4)
                axa.set_xticklabels(xticks[:(nb_trials // 4 - 1)])
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
                axb = plt.subplot(2, nb_plots // 2, jn+1)
                axb.set_xticks((nb_timesteps - 1) * np.arange(nb_trials // 4 - 1) + (nb_timesteps - 1) / 4)
                axb.set_xticklabels(xticks[:(nb_trials // 4 - 1)])
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
                y_acq = vb[:6*nb_timesteps, j].reshape((-1, nb_timesteps))[:, 1:].reshape((-1,))
                axb.plot(y_acq, color=acolour, label="acquisition")
                y_shock = vb[np.array([2, 3, 4, 5, 6]) * nb_timesteps - 1, j]
                x_shock = np.array([2, 3, 4, 5, 6]) * (nb_timesteps - 1) - 1
                axb.plot(x_shock, y_shock, 'r.')
                subs.append(axb)
            y_for = vb[6*nb_timesteps:, j].reshape((-1, nb_timesteps))[:, 1:].reshape((-1,))
            subs[jn].plot(y_for, color=colour, label=label)
        # plt.plot(vb[:, 0])
        # # plt.imshow(vb.T, vmin=vmin, vmax=vmax, cmap="coolwarm", aspect="auto")
        # plt.plot([[3, 5, 7, 9, 11]] * 2, [[0] * 5, [ylim] * 5], 'r-')
        # plt.xticks(nb_timesteps * np.arange(nb_trials // 2) + nb_timesteps / 4, xticks[:nb_trials // 2])
        # plt.yticks(np.arange(len(nids)) if only_nids else nids, [r'$%s$' % tick for tick in yticks[nids]])
        # axb.yaxis.set_ticks_position('both')
        # axb.tick_params(labelleft=False, labelright=True)
        # plt.title("%s - odour B - value" % ms[i].routine_name, color="C%d" % (2 * i + 1))

    subs[len(subs)//2 - 1].legend(fontsize=8, bbox_to_anchor=(1.05, 1.35), loc='upper left',
                                  framealpha=0., labelspacing=1.)
    subs[-1].legend(fontsize=8, bbox_to_anchor=(1.05, 1.35), loc='upper left', framealpha=0., labelspacing=1)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    from evaluation import evaluate, generate_behaviour_map
    target, target_s = generate_behaviour_map(cs_only=True)

    import pandas as pd

    pd.options.display.max_columns = 16
    pd.options.display.max_rows = 16
    pd.options.display.width = 1000

    get_score = False

    nb_kcs = 10
    kc1, kc2 = nb_kcs // 2, nb_kcs // 2

    model = MotivationModel(learning_rule="dan-based", nb_apl=0, pn2kc_init="default", verbose=False,
                            timesteps=3, trials=28, nb_kc=nb_kcs, nb_kc_odour_1=kc1, nb_kc_odour_2=kc2,
                            is_single=True, has_fom=True, has_bm=True, has_rsom=True, has_ltm=True, has_mdm=True,
                            has_real_names=False)
    print(model.w_p2k)
    if get_score:
        val, acc, prediction, models = evaluate(model, nids=model.neuron_ids, behav_mean=target, behav_std=target_s,
                                                cs_only=True, reversal=True, no_shock=True, unpaired=True,
                                                liyans_frames=False)

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
        val, acc, prediction, models = evaluate(model, behav_mean=pd.DataFrame({}), nids=model.neuron_ids,
                                                cs_only=True, reversal=True, unpaired=True, no_shock=True)

    # MBModel.plot_overlap(models, nids=neurons, score=acc)
    # MBModel.plot_timeline(models=models, nids=neurons, score=acc, target=target, nb_trials=13)
    # plot_population(models, only_nids=True)
    plot_individuals(models, only_nids=True)
