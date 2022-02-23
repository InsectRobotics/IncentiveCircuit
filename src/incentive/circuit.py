"""
Examples:
---------
>>> ic = IncentiveCircuit(learning_rule="dpr", nb_kc=10)
>>> ic(reversal=True)
"""

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2021, Insect Robotics Group," \
                "Institute of Perception, Action and Behaviour," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "GPLv3+"
__version__ = "v1.1-alpha"
__maintainer__ = "Evripidis Gkanias"

from .models_base import MBModel

import numpy as np


class IncentiveCircuit(MBModel):
    def __init__(self, has_sm=True, has_rm=True, has_ltm=True, has_rrm=True, has_rfm=True, has_mam=True,
                 has_real_names=False, as_microcircuits=False, ltm_speed=None,
                 b_d=-0.5, b_c=-0.15, b_f=-0.15, b_s=-2, b_r=-0.5, b_m=-0.5,
                 w_s2d=.3, w_d2s=1., w_s2r=1., w_r2c=.5, w_c2r=1., w_m2c=.3, w_c2m=.3, w_m2f=.5, w_f2m=1., w_f2r=.3,
                 *args, **kwargs):
        """
        The Incentive Circuit (IC) is a simplified version of the mushroom body from the Drosophila melanogaster
        brain, which is a hypothetical sub-circuit in it. It contains the connections from the Kenyon cells (KCs) to the
        output neurons (MBONs), from the MBONs to the dopaminergic neurons (DANs) and from the DANs to the connections
        from the KCs to MBONs. It takes as input a routine and produces the responses and weights of the mushroom body
        for every time-step. Its parameters are discussed in the manuscript.

        Parameters
        ----------
        has_sm: bool, optional
            indicates if synapses of the SM sub-circuit are included. Default is True.
        has_rm: bool, optional
            indicates if synapses of the RM sub-circuit are included. Default is True.
        has_rrm: bool, optional
            indicates if synapses of the RRM sub-circuit are included. Default is True.
        has_rfm: bool, optional
            indicates if synapses of the RFM sub-circuit are included. Default is True.
        has_ltm: bool, optional
            indicates if synapses of the LTM sub-circuit are included. Default is True.
        has_mam: bool, optional
            indicates if synapses of the MAM sub-circuit are included. Default is True.
        has_real_names: bool, optional
            indicates if real neuron names are to be used instead of code names. Default is False.
        as_microcircuits: bool, optional
            indicates whether the model is going to be used as a sub-circuit instead of as a whole. Default is False.
        """
        kwargs.setdefault("nb_dan", 6)
        kwargs.setdefault("nb_mbon", 6)
        kwargs.setdefault("leak", .0)
        super().__init__(*args, **kwargs)

        shock_magnitude = 2.
        odour_magnitude = 2.
        # ltm_speed = .5 if as_microcircuits else .05
        if ltm_speed is not None:
            w_c2m = w_m2c = w_f2r = ltm_speed

        pds, pde = 0, 2
        pcs, pce = 2, 4
        pfs, pfe = 4, 6
        pss, pse = 6, 8
        prs, pre = 8, 10
        pms, pme = 10, 12

        self.us_dims = 2
        self.w_p2k *= odour_magnitude

        self._v[:, pds:pde] = self.bias[pds:pde] = b_d  # D-DANs
        self._v[:, pcs:pce] = self.bias[pcs:pce] = b_c  # C-DANs
        self._v[:, pfs:pfe] = self.bias[pfs:pfe] = b_f  # F-DANs
        self._v[:, pss:pse] = self.bias[pss:pse] = b_s  # S-MBONs
        self._v[:, prs:pre] = self.bias[prs:pre] = b_r  # R-MBONs
        self._v[:, pms:pme] = self.bias[pms:pme] = b_m  # M-MBONs
        if as_microcircuits:
            self._v[:, prs:pre] = self.bias[prs:pre] = -2.  # R-MBONs
            self._v[:, pms:pme] = self.bias[pms:pme] = -4.  # M-MBONs

        self._w_m2v = np.zeros((self.nb_mbon + self.nb_dan, self.nb_mbon + self.nb_dan), dtype=float)
        self._w_d2k = np.zeros((self.nb_dan + self.nb_mbon, self.nb_dan + self.nb_mbon), dtype=float)

        # susceptible memory (SM) sub-circuit
        if has_sm:
            # Susceptible memories depress their opposite DANs
            self._w_m2v[pss:pse, pds:pde] = np.array([  # S-MBONs to D-DANs
                [+0., -1.],  # MBON-γ1ped (s_at)
                [-1., +0.]  # MBON-γ4>γ1γ2 (s_av)
            ]) * w_s2d
            # # Susceptible memories excite their respective DANs
            # self._w_m2v[pss:pse, pds:pde] = np.array([  # S-MBONs to D-DANs
            #     [+1., -0.],  # MBON-γ1ped (s_at)
            #     [-0., +1.]  # MBON-γ4>γ1γ2 (s_av)
            # ]) * .7

            # Discharging DANs depress their respective susceptible MBONs
            self._w_d2k[pds:pde, pss:pse] += -np.array([
                [float(m == (d + ((pde-pds) // 2)) % (pde-pds))
                 for m in range(pse-pss)]
                for d in range(pde-pds)
            ]) * w_d2s

        # restrained memory (RM) sub-circuit
        if has_rm:
            # Susceptible memories depress their opposite restrained MBONs
            self._w_m2v[pss:pse, prs:pre] = np.array([  # S-MBONs to R-MBONs
                [-.0, -1.],  # MBON-γ1ped (s_at)
                [-1., -.0]  # MBON-γ4>γ1γ2 (s_av)
            ]) * w_s2r

        # reciprocal restrained memories (RRM) sub-circuit
        if has_rrm:
            # Restrained memories enhance their same-valence DANs
            self._w_m2v[prs:pre, pcs:pce] = np.array([  # R-MBONs to C-DANs
                [+1., +0.],  # MBON-γ2α'1 (r_at)
                [+0., +1.]  # MBON-γ5β'2a (r_av)
            ]) * w_r2c

            # # Restrained memories inhibit their opposite DANs
            # self._w_m2v[prs:pre, pcs:pce] = np.array([  # R-MBONs to C-DANs
            #     [+0., -1.],  # MBON-γ2α'1 (r_at)
            #     [-1., +0.]  # MBON-γ5β'2a (r_av)
            # ]) * .5

            # Charging DANs depress their opposite restrained MBONs
            self._w_d2k[pcs:pce, prs:pre] += -np.array([
                [float(m == (d + ((pce-pcs) // 2)) % (pce-pcs))
                 for m in range(pre-prs)]
                for d in range(pce-pcs)
            ]) * w_c2r

        if has_ltm:
            # Long-term memory (LTM) sub-circuit
            self._w_m2v[pms:pme, pcs:pce] += np.array([  # M-MBONs to C-DANs
                [+1.0, +.0],  # MBON-β2β'2a (m_at)
                [+.0, +1.0],  # MBON-α'1 (m_av)
            ]) * w_m2c

            # # Long-term memory (LTM) sub-circuit
            # self._w_m2v[pms:pme, pcs:pce] += np.array([  # M-MBONs to C-DANs
            #     [+.0, -1.0],  # MBON-β2β'2a (m_at)
            #     [-1.0, +.0],  # MBON-α'1 (m_av)
            # ]) * ltm_speed

            # Charging DANs enhance their respective memory MBONs
            self._w_d2k[pcs:pce, pms:pme] += np.array([
                [float(m == (d + ((pce-pcs) // 2) + 1) % (pce-pcs))
                 for m in range(pme-pms)]
                for d in range(pce-pcs)
            ]) * w_c2m

        # reciprocal forgetting memories (RFM) sub-circuit
        if has_rfm:
            # Relative states enhance their opposite DANs
            self._w_m2v[pms:pme, pfs:pfe] = np.array([  # M-MBONs to F-DANs
                [+1., +.0],  # MBON-β2β'2a (m_at)
                [+.0, +1.]  # MBON-α'1 (m_av)
            ]) * w_m2f

            # Forgetting DANs depress their opposite long-term memory MBONs
            self._w_d2k[pfs:pfe, pms:pme] += -np.array([
                [float(m == (d + ((pfe-pfs) // 2)) % (pfe-pfs))
                 for m in range(pme-pms)]
                for d in range(pfe-pfs)
            ]) * w_f2m

        # Memory assimilation mechanism (MAM)
        if has_mam:
            self._w_d2k[pfs:pfe, prs:pre] += -np.array([
                [float(m == (d + ((pce-pcs) // 2) - 1) % (pce-pcs))
                 for m in range(pme-pms)]
                for d in range(pce-pcs)
            ]) * w_f2r

        u = np.zeros((2, self.nb_dan + self.nb_mbon), dtype=float)
        u[:, pds:pde] = np.eye(pde-pds) * shock_magnitude
        u[:, pcs:pce] = np.eye(pce-pcs) * shock_magnitude
        # u[:, pfs:pfe] = np.eye(pfe-pfs) * shock_magnitude / 2
        self.w_u2d = np.array(u)

        self.us_names = ["sugar", "shock"]
        if has_real_names:
            self.names[0], self.names[4], self.names[8], self.names[12], self.names[13], self.names[9] = (
                "PAM-γ4<γ1γ2", "PPL1-γ1ped", "PAM-β'2a", "PPL1-γ2α'1_2", "PPL1-γ2α'1_1", "PAM-β2β'2a")
            self.names[16], self.names[20], self.names[24], self.names[28], self.names[29], self.names[25] = (
                "MBON-γ1ped", "MBON-γ4>γ1γ2", "MBON-γ2α'1", "MBON-γ5β'2a", "MBON-β2β'2a", "MBON-α'1")
        else:
            self.names = [
                r"d_{at}", r"d_{av}", r"c_{at}", r"c_{av}", r"f_{at}", r"f_{av}",
                r"s_{at}", r"s_{av}", r"r_{at}", r"r_{av}", r"m_{at}", r"m_{av}"
            ]

        self.neuron_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    def __repr__(self):
        s = f"IncentiveCircuit(lr='{self._learning_rule}')"
        return s
