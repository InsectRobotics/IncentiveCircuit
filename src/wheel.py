from models_base import MBModel

import numpy as np


class WheelModel(MBModel):
    def __init__(self, *args, **kwargs):
        """
        The wheel of motivations model (WOM) is a simplified version of the mushroom body from the Drosophila
        melanogaster brain, which is a hypothetical circuit of it. It contains the connections from the Kenyon cells
        (KCs) to the output neurons (MBONs), from the MBONs to the dopaminergic neurons (DANs) and from the DANs to the
        connections from the KCs to MBONs. It takes as input a routine and produces the responses and weights of the
        mushroom body for every time-step. Its parameters are discussed in the manuscript.

        :param has_fom: indicates if synapses of the FOM sub-circuit are included
        :type has_fom: bool
        :param has_bm: indicates if synapses of the BM sub-circuit are included
        :type has_bm: bool
        :param has_rsom: indicates if synapses of the RSOM sub-circuit are included
        :type has_rsom: bool
        :param has_rfm: indicates if synapses of the RFM sub-circuit are included
        :type has_rfm: bool
        :param has_ltm: indicates if synapses of the LTM sub-circuit are included
        :type has_ltm: bool
        :param has_mdm: indicates if synapses of the MDM sub-circuit are included
        :type has_mdm: bool
        :param has_real_names: indicates if real neuron names are to be used instead of code names
        :type has_real_names: bool
        """
        kwargs.setdefault("nb_dan", 16)
        kwargs.setdefault("nb_mbon", 16)
        kwargs.setdefault("leak", .0)
        has_fom = kwargs.pop("has_fom", True)
        has_bm = kwargs.pop("has_bm", True)
        has_rsom = kwargs.pop("has_rsom", True)
        has_ltm = kwargs.pop("has_ltm", True)
        has_mdm = kwargs.pop("has_mdm", True)
        has_real_names = kwargs.pop("has_real_names", False)
        super().__init__(*args, **kwargs)

        shock_magnitude = 2.
        odour_magnitude = 2.
        p_dan_abs_s, p_dan_abs_e = 0, 8
        p_dan_stm_s, p_dan_stm_e = 8, 16
        p_dan_ltm_s, p_dan_ltm_e = 8, 16
        p_mbon_abs_s, p_mbon_abs_e = 16, 24
        p_mbon_stm_s, p_mbon_stm_e = 24, 32
        p_mbon_ltm_s, p_mbon_ltm_e = 24, 32

        self.us_dims = 8
        self.w_p2k *= odour_magnitude

        self._v[:, p_dan_abs_s:p_dan_abs_e] = self.bias[p_dan_abs_s:p_dan_abs_e] = -0.5  # D-DANs
        self._v[:, p_dan_stm_s:p_dan_stm_e] = self.bias[p_dan_stm_s:p_dan_stm_e] = -0.15  # R-DANs
        self._v[:, p_dan_ltm_s:p_dan_ltm_e] = self.bias[p_dan_ltm_s:p_dan_ltm_e] = -0.15  # F-DANs
        self._v[:, p_mbon_abs_s:p_mbon_abs_e] = self.bias[p_mbon_abs_s:p_mbon_abs_e] = -2.  # A-MBONs
        self._v[:, p_mbon_stm_s:p_mbon_stm_e] = self.bias[p_mbon_stm_s:p_mbon_stm_e] = -.5  # H-MBONs
        self._v[:, p_mbon_ltm_s:p_mbon_ltm_e] = self.bias[p_mbon_ltm_s:p_mbon_ltm_e] = -.5  # M-MBONs
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
            ]) * 1.

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
            ]) * .5

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
            ]) * .05

        # Memory digestion mechanism (MDM)
        if has_mdm:
            self._w_d2k[p_dan_ltm_s:p_dan_ltm_e, p_mbon_stm_s:p_mbon_stm_e] += -np.array([
                [float(m == (d + ((p_dan_stm_e-p_dan_stm_s) // 2) - 1) % (p_dan_stm_e-p_dan_stm_s))
                 for m in range(p_mbon_ltm_e-p_mbon_ltm_s)]
                for d in range(p_dan_stm_e-p_dan_stm_s)
            ]) * .05

        u = np.zeros((self.nb_dan // 2, self.nb_dan + self.nb_mbon), dtype=float)
        u[:, p_dan_abs_s:p_dan_abs_e] = np.eye(p_dan_abs_e-p_dan_abs_s) * shock_magnitude
        u[:, p_dan_stm_s:p_dan_stm_e] = np.eye(p_dan_stm_e-p_dan_stm_s) * shock_magnitude
        self.w_u2d = np.array(u)

        self.us_names = ["painkiller", "bright", "sugar", "hot", "shocked", "dark", "quinine", "cold"]
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
        s = "WheelModel("
        s += "lr='" + self._learning_rule + "'"
        if self.nb_apl > 0:
            s += ", apl=%d" % self.nb_apl
        s += ")"
        return s
