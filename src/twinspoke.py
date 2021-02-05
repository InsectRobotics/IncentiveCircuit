from models_base import MBModel

import numpy as np


class TwinSpokeModel(MBModel):
    def __init__(self, *args, **kwargs):
        """
        The twin-spoke model (TSM) is a simplified version of the mushroom body from the Drosophila melanogaster brain,
        which is a hypothetical sub-circuit in it. It contains the connections from the Kenyon cells (KCs) to the output
        neurons (MBONs), from the MBONs to the dopaminergic neurons (DANs) and from the DANs to the connections from the
        KCs to MBONs. It takes as input a routine and produces the responses and weights of the mushroom body for every
        time-step. Its parameters are discussed in the manuscript.

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
        :param as_subcircuit: indicates whether the model is going to be used as a sub-circuit instead of as a whole
        :type as_subcircuit: bool
        """
        kwargs.setdefault("nb_dan", 6)
        kwargs.setdefault("nb_mbon", 6)
        kwargs.setdefault("leak", .0)
        has_fom = kwargs.pop("has_fom", True)
        has_bm = kwargs.pop("has_bm", True)
        has_rsom = kwargs.pop("has_rsom", True)
        has_rfm = kwargs.pop("has_rfm", True)
        has_ltm = kwargs.pop("has_ltm", True)
        has_mdm = kwargs.pop("has_mdm", True)
        has_real_names = kwargs.pop("has_real_names", False)
        as_subcircuits = kwargs.pop("as_subcircuits", False)
        super().__init__(*args, **kwargs)

        shock_magnitude = 2.
        odour_magnitude = 2.
        ltm_speed = .3 if as_subcircuits else .05

        p_dan_abs_s, p_dan_abs_e = 0, 2
        p_dan_stm_s, p_dan_stm_e = 2, 4
        p_dan_ltm_s, p_dan_ltm_e = 4, 6
        p_mbon_abs_s, p_mbon_abs_e = 6, 8
        p_mbon_stm_s, p_mbon_stm_e = 8, 10
        p_mbon_ltm_s, p_mbon_ltm_e = 10, 12

        self.us_dims = 2
        self.w_p2k *= odour_magnitude

        self._v[:, p_dan_abs_s:p_dan_abs_e] = self.bias[p_dan_abs_s:p_dan_abs_e] = -0.5  # D-DANs
        self._v[:, p_dan_stm_s:p_dan_stm_e] = self.bias[p_dan_stm_s:p_dan_stm_e] = -0.15  # R-DANs
        self._v[:, p_dan_ltm_s:p_dan_ltm_e] = self.bias[p_dan_ltm_s:p_dan_ltm_e] = -0.15  # F-DANs
        self._v[:, p_mbon_abs_s:p_mbon_abs_e] = self.bias[p_mbon_abs_s:p_mbon_abs_e] = -2.  # A-MBONs
        self._v[:, p_mbon_stm_s:p_mbon_stm_e] = self.bias[p_mbon_stm_s:p_mbon_stm_e] = -.5  # H-MBONs
        self._v[:, p_mbon_ltm_s:p_mbon_ltm_e] = self.bias[p_mbon_ltm_s:p_mbon_ltm_e] = -.5  # M-MBONs
        if as_subcircuits:
            self._v[:, p_mbon_stm_s:p_mbon_stm_e] = self.bias[p_mbon_stm_s:p_mbon_stm_e] = -2.  # H-MBONs
            self._v[:, p_mbon_ltm_s:p_mbon_ltm_e] = self.bias[p_mbon_ltm_s:p_mbon_ltm_e] = -4.  # M-MBONs

        self._w_m2v = np.zeros((self.nb_mbon + self.nb_dan, self.nb_mbon + self.nb_dan), dtype=float)
        self._w_d2k = np.zeros((self.nb_dan + self.nb_mbon, self.nb_dan + self.nb_mbon), dtype=float)

        # first order memory (FOM) sub-circuit
        if has_fom:
            # Absolute states depress their respective DANs
            self._w_m2v[p_mbon_abs_s:p_mbon_abs_e, p_dan_abs_s:p_dan_abs_e] = np.array([  # A-MBONs to D-DANs
                [+0., -1.],  # MBON-γ1ped (a_at)
                [-1., +0.]  # MBON-γ4>γ1γ2 (a_av)
            ]) * .3
            # Absolute DANs depress their opposite MBONs
            self._w_d2k[p_dan_abs_s:p_dan_abs_e, p_mbon_abs_s:p_mbon_abs_e] += -np.array([
                [float(m == (d + ((p_dan_abs_e-p_dan_abs_s) // 2)) % (p_dan_abs_e-p_dan_abs_s))
                 for m in range(p_mbon_abs_e-p_mbon_abs_s)]
                for d in range(p_dan_abs_e-p_dan_abs_s)
            ])

        # blocking memory (BM) sub-circuit
        if has_bm:
            # Absolute states depress their opposite relative MBONs
            self._w_m2v[p_mbon_abs_s:p_mbon_abs_e, p_mbon_stm_s:p_mbon_stm_e] = np.array([  # A-MBONs to H-MBONs
                [-.0, -1.],  # MBON-γ1ped (a_at)
                [-1., -.0]  # MBON-γ4>γ1γ2 (a_av)
            ]) * 1.

        # reciprocal second order memories (RSOM) sub-circuit
        if has_rsom:
            # Relative states enhance their opposite DANs
            self._w_m2v[p_mbon_stm_s:p_mbon_stm_e, p_dan_stm_s:p_dan_stm_e] = np.array([  # H-MBONs to R-DANs
                [+1., +.0],  # MBON-γ2α'1 (h_at)
                [+.0, +1.]  # MBON-γ5β'2a (h_av)
            ]) * .5

            # Relative DANs depress their opposite MBONs
            self._w_d2k[p_dan_stm_s:p_dan_stm_e, p_mbon_stm_s:p_mbon_stm_e] += -np.array([
                [float(m == (d + ((p_dan_stm_e-p_dan_stm_s) // 2)) % (p_dan_stm_e-p_dan_stm_s))
                 for m in range(p_mbon_stm_e-p_mbon_stm_s)]
                for d in range(p_dan_stm_e-p_dan_stm_s)
            ]) * 1.

        if has_ltm:
            # Long-term memory (LTM) sub-circuit
            self._w_m2v[p_mbon_ltm_s:p_mbon_ltm_e, p_dan_stm_s:p_dan_stm_e] += np.array([  # M-MBONs to R-DANs
                [+1.0, +.0],  # MBON-γ2α'1 (h_at)
                [+.0, +1.0],  # MBON-γ5β'2a (h_av)
            ]) * .05
            # Relative DANs enhance their respective memory MBONs
            self._w_d2k[p_dan_stm_s:p_dan_stm_e, p_mbon_ltm_s:p_mbon_ltm_e] += np.array([
                [float(m == (d + ((p_dan_stm_e-p_dan_stm_s) // 2) + 1) % (p_dan_stm_e-p_dan_stm_s))
                 for m in range(p_mbon_ltm_e-p_mbon_ltm_s)]
                for d in range(p_dan_stm_e-p_dan_stm_s)
            ]) * ltm_speed

        # reciprocal forgetting memories (RFM) sub-circuit
        if has_rfm:
            # Relative states enhance their opposite DANs
            self._w_m2v[p_mbon_ltm_s:p_mbon_ltm_e, p_dan_ltm_s:p_dan_ltm_e] = np.array([  # H-MBONs to R-DANs
                [+1., +.0],  # MBON-β2β'2a (m_at)
                [+.0, +1.]  # MBON-α'1 (m_av)
            ]) * .5

            # Relative DANs depress their opposite MBONs
            self._w_d2k[p_dan_ltm_s:p_dan_ltm_e, p_mbon_ltm_s:p_mbon_ltm_e] += -np.array([
                [float(m == (d + ((p_dan_ltm_e-p_dan_ltm_s) // 2)) % (p_dan_ltm_e-p_dan_ltm_s))
                 for m in range(p_mbon_ltm_e-p_mbon_ltm_s)]
                for d in range(p_dan_ltm_e-p_dan_ltm_s)
            ]) * 1.

        # Memory digestion mechanism (MDM)
        if has_mdm:
            self._w_d2k[p_dan_ltm_s:p_dan_ltm_e, p_mbon_stm_s:p_mbon_stm_e] += -np.array([
                [float(m == (d + ((p_dan_stm_e-p_dan_stm_s) // 2) - 1) % (p_dan_stm_e-p_dan_stm_s))
                 for m in range(p_mbon_ltm_e-p_mbon_ltm_s)]
                for d in range(p_dan_stm_e-p_dan_stm_s)
            ]) * ltm_speed

        u = np.zeros((2, self.nb_dan + self.nb_mbon), dtype=float)
        u[:, p_dan_abs_s:p_dan_abs_e] = np.eye(p_dan_abs_e-p_dan_abs_s) * shock_magnitude
        u[:, p_dan_stm_s:p_dan_stm_e] = np.eye(p_dan_stm_e-p_dan_stm_s) * shock_magnitude
        self.w_u2d = np.array(u)

        self.us_names = ["sugar", "shock"]
        if has_real_names:
            self.names[0], self.names[4], self.names[8], self.names[12], self.names[13], self.names[9] = (
                "PAM-γ4<γ1γ2", "PPL1-γ1ped", "PAM-β'2a", "PPL1-γ2α'1_2", "PPL1-γ2α'1_1", "PAM-β2")
            self.names[16], self.names[20], self.names[24], self.names[28], self.names[29], self.names[25] = (
                "MBON-γ1ped", "MBON-γ4>γ1γ2", "MBON-γ2α'1", "MBON-γ5β'2a", "MBON-β2β'2a", "MBON-α'1")
        else:
            self.names = [
                r"d_{at}", r"d_{av}", r"r_{at}", r"r_{av}", r"f_{at}", r"f_{av}",
                r"a_{at}", r"a_{av}", r"h_{at}", r"h_{av}", r"m_{at}", r"m_{av}"
            ]

        self.neuron_ids = [0, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11]

    def __repr__(self):
        s = "TwinSpokeModel("
        s += "lr='" + self._learning_rule + "'"
        if self.nb_apl > 0:
            s += ", apl=%d" % self.nb_apl
        s += ")"
        return s
