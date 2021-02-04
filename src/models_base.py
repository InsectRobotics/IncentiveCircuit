from routines import no_shock_routine, unpaired_routine, reversal_routine

import numpy as np
import matplotlib.pyplot as plt

from copy import copy

synapse_counts = {
    "PPL1-γ1ped": {},
    "PPL1-γ2α'1": {
        "MBON-γ2α'1": [256, 229],
    },
    "PAM-β'2a": {
        "MBON-γ5β'2a": [60, 31, 23, 31, 25, 52, 56, 46, 42, 41]
    },
    "MBON-γ1ped": {
        "PAM-β'2a": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    },
    "MBON-γ3β'1": {
        "PAM-β'2a": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    },
    "MBON-γ4γ5": {
        "PAM-β'2a": [2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    },
    "MBON-γ2α'1": {
        "PAM-β'2a": [10, 3, 14, 6, 7, 8, 5, 6, 0, 0, 2, 2, 3, 1, 5, 2, 9, 7, 13, 0]
    },
    "MBON-γ5β'2a": {
        "PAM-β'2a": [1, 1, 1, 1, 2, 3, 1, 2, 0, 0]
    },
    "MBON-α'2": {
        "PAM-β'2a": [2, 6, 7, 9, 2, 6, 2, 3, 13, 0]
    },
    "MBON-β'2a": {
        "PAM-β'2a": [2, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    }
}


class MBModel(object):
    def __init__(self, nb_kc=2, nb_mbon=3, nb_dan=3, learning_rule="default", pn2kc_init="default", nb_apl=0, trials=17,
                 timesteps=3, nb_kc_odour_1=1, nb_kc_odour_2=1, eta=1., leak=.8, verbose=False):
        self.eta_w = eta
        self.eta_v = eta
        self.b_init = 1.
        self.vd_init = self.b_init
        self.vm_init = self.b_init
        self.nb_trials = trials
        self.nb_timesteps = timesteps
        self.__leak = leak
        self.us_dims = 8

        self._t = 0
        self._learning_rule = learning_rule
        self.__routine_name = ""
        self.__verbose = verbose

        self.w_p2k = np.array([
            [1.] * nb_kc_odour_1 + [0.] * (nb_kc - nb_kc_odour_1),
            [0.] * (nb_kc - nb_kc_odour_2) + [1.] * nb_kc_odour_2
        ])
        nb_pn, nb_kc = self.w_p2k.shape
        if pn2kc_init in ["default"]:
            self.w_p2k *= 1. / np.array([[nb_kc_odour_1], [nb_kc_odour_2]], dtype=self.w_p2k.dtype)
        elif pn2kc_init in ["simple"]:
            self.w_p2k *= nb_pn / nb_kc
        elif pn2kc_init in ["sqrt_pn", "pn_sqrt"]:
            self.w_p2k *= np.square(nb_pn) / nb_kc
        elif pn2kc_init in ["sqrt_kc", "kc_sqrt"]:
            self.w_p2k *= nb_pn / np.sqrt(nb_kc)

        v = []
        u = np.zeros((nb_dan // 2, nb_dan + nb_mbon), dtype=float)
        u[:, :nb_dan] = np.eye(nb_dan // 2, nb_dan)
        u[:, nb_dan//2:nb_dan] = np.eye(nb_dan // 2, nb_dan - nb_dan // 2) * 2

        if nb_dan == 3:
            d1 = 2.  # PPL1-γ1ped
            d2 = 0.  # PPL1-γ2α'1
            d3 = -1.  # PAM-β'2a
            v += [d1, d2, d3]
        else:
            v += [0.] * nb_dan

        if nb_mbon == 3:
            m1 = 0.  # MBON-γ1ped
            m2 = 0.  # MBON-γ2α'1
            m3 = 0.  # MBON-γ5β'2a
            v += [m1, m2, m3]
        else:
            v += [0.] * nb_mbon

        self._v = np.array([np.array(v).T] * (trials * timesteps + 1))
        self._v_apl = np.zeros((trials * timesteps + 1, nb_apl), dtype=self._v.dtype)
        self.v_init = self._v[0]
        self.v_max = np.array([+2.] * (nb_dan+nb_mbon))
        self.v_min = np.array([-2.] * (nb_dan+nb_mbon))
        self.w_max = np.array([+2.] * (nb_dan+nb_mbon))
        self.w_min = np.array([-2.] * (nb_dan+nb_mbon))

        self.w_k2m = np.array([[[0.] * nb_dan + [1.] * nb_mbon] * nb_kc] * (trials * timesteps + 1))
        self.w_u2d = np.array(u)
        self.k2m_init = self.w_k2m[0, 0]

        # MBON-to-MBON and MBON-to-DAN synaptic weights
        if nb_dan == 3 and nb_mbon == 3:
            self._w_m2v = np.array([  # M to V_m
                # d1,  d2,  d3,  m1,  m2,  m3
                [-0., -0., -0., -0., -0., -0.],  # d1: PPL1-γ1ped
                [-0., -0., -0., -0., -0., -0.],  # d2: PPL1-γ2α'1
                [-0., -0., -0., -0., -0., -0.],  # d3: PAM-β'2'a
                [-1., -0., -0., -0., -0., -5.],  # m1: MBON-γ1ped
                [+0., +0., +1., +0., +0., +0.],  # m2: MBON-γ2α'1
                [+0., +0., +0., +0., +0., +0.]  # m3: MBON-γ5β'2a
            ])
        else:
            self._w_m2v = np.zeros((nb_dan+nb_mbon, nb_dan+nb_mbon), dtype=float)
        self._w_d2k = np.array([
            ([0.] * nb_dan + [-float(m == d) for m in range(nb_mbon)]) for d in range(nb_dan+nb_mbon)], dtype=float)

        self.w_k2a = np.ones((nb_kc, nb_apl), dtype=float) / nb_kc
        if nb_apl > 0:
            self.w_d2a = np.array([[-1.] * nb_dan + [0.] * nb_mbon] * nb_apl).T / nb_dan
        else:
            self.w_d2a = np.zeros((6, nb_apl), dtype=float)
        self.w_a2k = -np.ones((nb_apl, nb_kc), dtype=float) / nb_kc
        self.w_a2d = np.array([[-1.] * nb_dan + [0.] * nb_mbon] * nb_apl)

        self.nb_pn = nb_pn
        self.nb_kc = nb_kc
        self.nb_mbon = nb_mbon
        self.nb_dan = nb_dan
        self.nb_apl = nb_apl

        self.csa = np.array([1., 0.]) * 2
        self.csb = np.array([0., 1.]) * 2
        self.__shock_on = []
        self.__cs_on = []

        self.names = ["DAN-%d" % (i+1) for i in range(nb_dan)] + ["MBON-%d" % (i+1) for i in range(nb_mbon)]
        self.neuron_ids = []

    @property
    def w_m2v(self):
        return np.eye(*self._w_m2v.shape) + self._w_m2v

    @property
    def w_d2k(self):
        return self._w_d2k

    @property
    def cs_on(self):
        return self.__cs_on

    @property
    def us_on(self):
        return self.__shock_on

    @property
    def routine_name(self):
        return self.__routine_name

    @routine_name.setter
    def routine_name(self, v):
        self.__routine_name = v

    @property
    def verbose(self):
        return self.__verbose

    def _a(self, v, v_max=None, v_min=None):
        if v_max is None:
            v_max = self.v_max
        if v_min is None:
            v_min = self.v_min
        return leaky_relu(v, alpha=self.__leak, v_max=v_max, v_min=v_min)

    def __call__(self, *args, **kwargs):
        if kwargs.get('no_shock', False):
            routine = no_shock_routine(self)
        elif kwargs.get('unpaired', False):
            routine = unpaired_routine(self)
        elif kwargs.get('reversal', False):
            routine = reversal_routine(self)
        else:
            routine = kwargs.get('routine', reversal_routine(self))
        repeat = kwargs.get('repeat', 4)

        for _, _, cs, us in routine:

            # create dynamic memory for repeating loop
            w_k2m_pre, w_k2m_post = self.w_k2m[self._t].copy(), self.w_k2m[self._t].copy()
            v_pre, v_post = self._v[self._t].copy(), self._v[self._t].copy()

            # feed forward responses: PN(CS) -> KC
            k = cs @ self.w_p2k

            eta = float(1) / float(repeat)
            for r in range(repeat):

                # feed forward responses: KC -> MBON, US -> DAN
                mb = k @ w_k2m_pre + us @ self.w_u2d + self.v_init

                # Step 1: internal values update
                v_post = self.update_values(k, v_pre, mb)

                # Step 2: synaptic weights update
                w_k2m_post = self.update_weights(k, v_post, w_k2m_pre)

                # update dynamic memory for repeating loop
                v_pre += eta * (v_post - v_pre)
                w_k2m_pre += eta * (w_k2m_post - w_k2m_pre)

            # store values and weights in history
            self._v[self._t + 1], self.w_k2m[self._t + 1] = v_post, w_k2m_post
            # self._v_apl[self._t + 1] = v_apl_post

    def update_weights(self, kc, v, w_k2m):
        eta_w = self.eta_w / float(self.nb_timesteps - 1)
        # eta_w = np.sqrt(self.eta_w / float(self.nb_timesteps))

        k = kc[..., np.newaxis]

        if self.nb_apl > 0:
            v_apl = self._v_apl[self._t + 1] = k[..., 0] @ self.w_k2a
            k += (v_apl @ self.w_a2k)[..., np.newaxis]
        else:
            k = np.maximum(k, 0)  # do not learn negative values

        D = np.maximum(v, 0).dot(self.w_d2k)

        if self._learning_rule in ["dopaminergic", "dlr", "dan-base", "default"]:
            # When DAN > 0 and KC > W - W_rest increase the weight (if DAN < 0 it is reversed)
            # When DAN > 0 and KC < W - W_rest decrease the weight (if DAN < 0 it is reversed)
            # When DAN = 0 no learning happens
            w_new = w_k2m + eta_w * D * (k + w_k2m - self.k2m_init)
        elif self._learning_rule in ["rescorla-wagner", "rw", "kc-based"]:
            # When KC > 0 and DAN > W - W_rest increase the weight (if KC < 0 it is reversed)
            # When KC > 0 and DAN < W - W_rest decrease the weight (if KC < 0 it is reversed)
            # When KC = 0 no learning happens
            w_new = w_k2m + eta_w * k * (D - w_k2m + self.k2m_init)
        else:
            w_new = w_k2m

        return np.maximum(w_new, 0)

    def update_values(self, kc, v, mb):
        reduce = 1. / float(self.nb_timesteps)
        # eta_v = 1. / float(self.nb_timesteps)
        eta_v = np.power(self.eta_v * reduce, reduce)

        if self.nb_apl > 0 and False:
            v_apl = self._v_apl[self._t+1] = np.maximum(kc @ self.w_k2a + (mb @ self.w_m2v - v) @ self.w_d2a, 0)
            mb += v_apl @ self.w_a2d
        nr = mb
        v_temp = (1 - eta_v) * v + eta_v * (nr.dot(self.w_m2v) - v)
        return self._a(v_temp)

    def copy(self):
        return copy(self)

    def __copy__(self):
        m = self.__class__()
        for att in self.__dict__:
            m.__dict__[att] = copy(self.__dict__[att])

        return m

    def __repr__(self):
        s = "MBModel("
        if self.__routine_name != "":
            s += "routine='" + self.__routine_name + "'"
        else:
            s += "routine=None"
        if self._learning_rule != "default":
            s += ", lr='" + self._learning_rule + "'"
        if self.nb_apl > 0:
            s += ", apl=%d" % self.nb_apl
        s += ")"
        return s


def leaky_relu(v, alpha=.1, v_max=np.inf, v_min=-np.inf):
    return np.clip(v, np.maximum(alpha * v, v_min), v_max)


def plot_relation(kc_vals, levels=None, origin='lower'):

    if levels is None:
        levels = [.00, .15, .30, .45, .60, .75, .90, 1.00]
        # levels = [.60, .75, .90]

    n, m = kc_vals.shape
    plt.figure("kcs-overlap", figsize=(6, 5))
    img = plt.imshow(kc_vals, extent=[-.5, n-.5, -.5, m-.5], vmin=0, vmax=1, cmap="Greys", origin=origin)
    # cof = plt.contourf(kc_vals, extent=[0, n-1, 0, m-1], vmin=0, vmax=1, cmap="Greys", origin=origin)
    col = plt.contour(kc_vals, levels=levels, cmap="magma", origin='lower')
    # cbar = plt.colorbar()
    plt.clabel(col, inline=True, fontsize=8)
    cbar = plt.colorbar(img)
    cbar.add_lines(col)
    cbar.ax.set_yticks(levels)
    cbar.ax.set_ylabel("Score")

    plt.ylim([.5, n-.5])
    plt.xlim([.5, m-.5])
    plt.ylabel("#KCs for odour A")
    plt.xlabel("#KCs for odour B")

    plt.tight_layout()
    plt.show()
