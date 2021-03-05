from models_base import MBModel

import numpy as np


class IncentiveWheel(MBModel):
    def __init__(self, *args, **kwargs):
        """
        The incentive wheel model (IW) is a simplified version of the mushroom body from the Drosophila
        melanogaster brain, which is a hypothetical circuit of it. It contains the connections from the Kenyon cells
        (KCs) to the output neurons (MBONs), from the MBONs to the dopaminergic neurons (DANs) and from the DANs to the
        connections from the KCs to MBONs. It takes as input a routine and produces the responses and weights of the
        mushroom body for every time-step. Its parameters are discussed in the manuscript.

        :param has_sm: indicates if synapses of the SM sub-circuit are included
        :type has_sm: bool
        :param has_r: indicates if synapses of the RM sub-circuit are included
        :type has_r: bool
        :param has_rrm: indicates if synapses of the RRM sub-circuit are included
        :type has_rrm: bool
        :param has_rfm: indicates if synapses of the RFM sub-circuit are included
        :type has_rfm: bool
        :param has_ltm: indicates if synapses of the LTM sub-circuit are included
        :type has_ltm: bool
        :param has_mam: indicates if synapses of the MAM sub-circuit are included
        :type has_mam: bool
        :param has_real_names: indicates if real neuron names are to be used instead of code names
        :type has_real_names: bool
        """
        kwargs.setdefault("nb_dan", 16)
        kwargs.setdefault("nb_mbon", 16)
        kwargs.setdefault("leak", .0)
        has_sm = kwargs.pop("has_sm", True)
        has_r = kwargs.pop("has_r", True)
        has_rrm = kwargs.pop("has_rrm", True)
        has_ltm = kwargs.pop("has_ltm", True)
        has_mam = kwargs.pop("has_mam", True)
        has_real_names = kwargs.pop("has_real_names", False)
        super().__init__(*args, **kwargs)

        shock_magnitude = 2.
        odour_magnitude = 2.
        pds, pde = 0, 8
        pcs, pce = 8, 16
        pfs, pfe = 8, 16
        pss, pse = 16, 24
        prs, pre = 24, 32
        pms, pme = 24, 32

        self.us_dims = 8
        self.w_p2k *= odour_magnitude

        self._v[:, pds:pde] = self.bias[pds:pde] = -0.5  # D-DANs
        self._v[:, pcs:pce] = self.bias[pcs:pce] = -0.15  # C-DANs
        self._v[:, pfs:pfe] = self.bias[pfs:pfe] = -0.15  # F-DANs
        self._v[:, pss:pse] = self.bias[pss:pse] = -2.  # S-MBONs
        self._v[:, prs:pre] = self.bias[prs:pre] = -.5  # R-MBONs
        self._v[:, pms:pme] = self.bias[pms:pme] = -.5  # M-MBONs

        self._w_m2v = np.zeros((self.nb_mbon + self.nb_dan, self.nb_mbon + self.nb_dan), dtype=float)
        self._w_d2k = np.zeros((self.nb_dan + self.nb_mbon, self.nb_dan + self.nb_mbon), dtype=float)

        # susceptible memory (SM) sub-circuit
        if has_sm:
            # Susceptible states depress their respective discharging DANs
            self._w_m2v[pss:pse, pds:pde] = np.array([  # S-MBONs to D-DANs
                [+0., +0., +0., +0., -1., +0., +0., +0.],  # trust
                [+0., +0., +0., +0., +0., -1., +0., +0.],  # fear
                [+0., +0., +0., +0., +0., +0., -1., +0.],  # surprise
                [+0., +0., +0., +0., +0., +0., +0., -1.],  # sadness
                [-1., +0., +0., +0., +0., +0., +0., +0.],  # disgust
                [+0., -1., +0., +0., +0., +0., +0., +0.],  # anger
                [+0., +0., -1., +0., +0., +0., +0., +0.],  # anticipation
                [+0., +0., +0., -1., +0., +0., +0., +0.],  # joy
            ]) * .3
            # Discharging DANs depress their respective susceptible MBONs
            self._w_d2k[pds:pde, pss:pse] += -np.array([
                [float(m == (d + ((pde-pds) // 2)) % (pde-pds))
                 for m in range(pse-pss)]
                for d in range(pde-pds)
            ])

        # restrained memory (RM) sub-circuit
        if has_r:
            # Susceptible states depress their respective restrained MBONs
            self._w_m2v[pss:pse, prs:pre] = np.array([  # S-MBONs to R-MBONs
                [-.0, -.0, -.0, -.0, -1., -.0, -.0, -.0],  # trust
                [-.0, -.0, -.0, -.0, -.0, -1., -.0, -.0],  # fear
                [-.0, -.0, -.0, -.0, -.0, -.0, -1., -.0],  # surprise
                [-.0, -.0, -.0, -.0, -.0, -.0, -.0, -1.],  # sadness
                [-1., -.0, -.0, -.0, -.0, -.0, -.0, -.0],  # disgust
                [-.0, -1., -.0, -.0, -.0, -.0, -.0, -.0],  # anger
                [-.0, -.0, -1., -.0, -.0, -.0, -.0, -.0],  # anticipation
                [-.0, -.0, -.0, -1., -.0, -.0, -.0, -.0],  # joy
            ]) * 1.

        # reciprocal restrained memories (RRM) sub-circuit
        if has_rrm:
            # Restrained states enhance their respective charging DANs
            self._w_m2v[prs:pre, pcs:pce] = np.array([  # R-MBONs to C-DANs
                [+1., +.0, +.0, +.0, -.0, +.0, +.0, +.0],  # trust
                [+.0, +1., +.0, +.0, +.0, -.0, +.0, +.0],  # fear
                [+.0, +.0, +1., +.0, +.0, +.0, -.0, +.0],  # surprise
                [+.0, +.0, +.0, +1., +.0, +.0, +.0, -.0],  # sadness
                [-.0, +.0, +.0, +.0, +1., +.0, +.0, +.0],  # disgust
                [+.0, -.0, +.0, +.0, +.0, +1., +.0, +.0],  # anger
                [+.0, +.0, -.0, +.0, +.0, +.0, +1., +.0],  # anticipation
                [+.0, +.0, +.0, -.0, +.0, +.0, +.0, +1.],  # joy
            ]) * .5

            # Charging DANs depress their respective restrained MBONs
            self._w_d2k[pcs:pce, prs:pre] += -np.array([
                [float(m == (d + ((pce-pcs) // 2)) % (pce-pcs))
                 for m in range(pre-prs)]
                for d in range(pce-pcs)
            ]) * 1.

        if has_ltm:
            # Long-term memory (LTM) sub-circuit
            self._w_m2v[pms:pme, pcs:pce] += np.array([  # M-MBONs to C-DANs
                [+.0, +.0, +.0, +1.0, +.0, +.0, +.0, +.0],  # trust
                [+.0, +.0, +.0, +.0, +1.0, +.0, +.0, +.0],  # fear
                [+.0, +.0, +.0, +.0, +.0, +1.0, +.0, +.0],  # surprise
                [+.0, +.0, +.0, +.0, +.0, +.0, +1.0, +.0],  # sadness
                [+.0, +.0, +.0, +.0, +.0, +.0, +.0, +1.0],  # disgust
                [+1.0, +.0, +.0, +.0, +.0, +.0, +.0, +.0],  # anger
                [+.0, +1.0, +.0, +.0, +.0, +.0, +.0, +.0],  # anticipation
                [+.0, +.0, +1.0, +.0, +.0, +.0, +.0, +.0],  # joy
            ]) * .05
            # Charging DANs enhance their respective long-term memory MBONs
            self._w_d2k[pcs:pce, pms:pme] += np.array([
                [float(m == (d + ((pce-pcs) // 2) + 1) % (pce-pcs))
                 for m in range(pme-pms)]
                for d in range(pce-pcs)
            ]) * .05

        # Memory assimilation mechanism (MAM)
        if has_mam:
            self._w_d2k[pfs:pfe, prs:pre] += -np.array([
                [float(m == (d + ((pce-pcs) // 2) - 1) % (pce-pcs))
                 for m in range(pme-pms)]
                for d in range(pce-pcs)
            ]) * .05

        u = np.zeros((self.nb_dan // 2, self.nb_dan + self.nb_mbon), dtype=float)
        u[:, pds:pde] = np.eye(pde-pds) * shock_magnitude
        u[:, pcs:pce] = np.eye(pce-pcs) * shock_magnitude
        self.w_u2d = np.array(u)

        self.us_names = ["friendly", "predator", "unexpected", "failure",
                         "abominable", "enemy", "new territory", "possess"]
        if has_real_names:
            self.names[0], self.names[4], self.names[8], self.names[12], self.names[13], self.names[9] = (
                "PAM-γ4<γ1γ2", "PPL1-γ1ped", "PAM-β'2a", "PPL1-γ2α'1", "PPL1-α'2α2", "PPL1-γ5")
            self.names[16], self.names[20], self.names[24], self.names[28], self.names[29], self.names[25] = (
                "MBON-γ1ped", "MBON-γ4>γ1γ2", "MBON-γ2α'1", "MBON-γ5β'2a", "MBON-β2β'2a", "MBON-α'1")
        else:
            for i in range(8):
                self.names[i] = r"d_{%d}" % i
                self.names[8 + i] = r"f_{%d}" % i
                self.names[16 + i] = r"a_{%d}" % i
                self.names[24 + i] = r"m_{%d}" % i

        self.neuron_ids = [0, 4, 16, 20, 8, 12, 24, 28, 13, 9, 29, 25]
        # self.neuron_ids = [4, 16, 8, 12, 24, 28]

    def __repr__(self):
        s = "IncentiveWheel("
        s += "lr='" + self._learning_rule + "'"
        if self.nb_apl > 0:
            s += ", apl=%d" % self.nb_apl
        s += ")"
        return s
