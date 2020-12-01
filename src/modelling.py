from imaging import plot_overlap

import numpy as np
import pandas as pd
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
    def __init__(self, eta=.9, vd_init=1., vm_init=1., trials=17, repeats=1, degrade_factor=.8, leak=.8, normalise=False,
                 nb_kc=2, nb_kc_odour_1=1, nb_kc_odour_2=1, nb_apl=0, model_1=True, model_2=True, lam=0., timesteps=1,
                 no_value=False, weights_degrade=False, unidimensional=False, bias_degrade=False, us_plasticity=False,
                 constraints=False, weight_constraints=False, learning_rule="default", pn2kc_init="default",
                 verbose=True):
        self.repeats = repeats
        self.eta_w = eta if not no_value else 1.
        self.eta_v = eta if not no_value else 1.
        self.b_init = 1.
        self.vd_init = vd_init * self.b_init
        self.vm_init = vm_init * self.b_init
        self.nb_trials = trials
        self.nb_timesteps = timesteps
        self.c_forgetting = degrade_factor if weights_degrade or bias_degrade else 1.
        self.__leak = leak
        self.__lambda = lam

        self._t = 0
        self.__no_value = no_value
        self.__unidimensional = unidimensional
        self.__weights_degrade = weights_degrade
        self.__bias_degrade = bias_degrade
        self.__us_plasticity = us_plasticity
        self.__normalised = normalise
        self.__eligibility_trace = lam > 0.
        self.__constraints = constraints
        self.__weights_constraints = weight_constraints
        self.__model_1 = model_1
        self.__model_2 = model_2
        self.__routine_name = ""
        self.__learning_rule = learning_rule
        self.__verbose = verbose

        self.w_p2k = np.array([
            [1.] * nb_kc_odour_1 + [0.] * (nb_kc - nb_kc_odour_1),
            [0.] * (nb_kc - nb_kc_odour_2) + [1.] * nb_kc_odour_2
        ])
        nb_pn, nb_kc = self.w_p2k.shape
        if pn2kc_init in ["default"]:
            self.w_p2k *= nb_pn / np.array([[nb_kc_odour_1], [nb_kc_odour_2]], dtype=self.w_p2k.dtype)
        elif pn2kc_init in ["simple"]:
            self.w_p2k *= nb_pn / nb_kc
        elif pn2kc_init in ["sqrt_pn", "pn_sqrt"]:
            self.w_p2k *= np.square(nb_pn) / nb_kc
        elif pn2kc_init in ["sqrt_kc", "kc_sqrt"]:
            self.w_p2k *= nb_pn / np.sqrt(nb_kc)

        self.nb_apl = nb_apl

        d1 = [(+0. + 2. * float(model_1) + 0.0 * float(model_2)) * vd_init] * nb_kc  # PPL1-γ1ped
        d2 = [(+0. + 0. * float(model_1) + 0.0 * float(model_2)) * vd_init] * nb_kc  # PPL1-γ2α'1
        d3 = [(+0. + 0. * float(model_1) - 1.0 * float(model_2)) * vd_init] * nb_kc  # PAM-β'2a

        m1 = [(+1. - 1. * float(model_1) - 0.0 * float(model_2)) * vm_init] * nb_kc  # MBON-γ1ped
        m2 = [(+1. + 0. * float(model_1) - 1.0 * float(model_2)) * vm_init] * nb_kc  # MBON-γ2α'1
        m3 = [(+1. + 0. * float(model_1) - 1.0 * float(model_2)) * vm_init] * nb_kc  # MBON-γ5β'2a

        self._v = np.array([np.array([d1, d2, d3, m1, m2, m3]).T] * (trials * timesteps + 1))
        self._v_apl = np.zeros((trials * timesteps + 1, nb_apl), dtype=self._v.dtype)
        self.r = np.zeros(((trials * timesteps + 1), 6), dtype=self._v.dtype)
        self.w_diff = np.zeros(((trials * timesteps + 1), 6), dtype=self._v.dtype)
        self.v_init = self._v[0, 0]
        self.md = self._v.copy()
        if not constraints:
            self.v_max = np.array([2., 2., 2., 2., 2., 4.])
            self.v_min = np.array([-2., -2., -2., -2., -2., -2.])
        else:
            self.v_max = np.array([2., 2., 2., 1., 2., 2.])
            self.v_min = np.array([-2., -2., -2., -2., -2., -0.])
        if not weight_constraints:
            self.w_max = np.array([+2.0, +2.0, +2.0, +2.0, +2.0, +2.0])
            self.w_min = np.array([-2.0, -2.0, -2.0, -2.0, -2.0, -2.0])
        else:
            self.w_max = np.array([+2.0, +2.0, +2.0, +1.0, +2.0, +2.0])
            self.w_min = np.array([-2.0, -2.0, -2.0, -2.0, -2.0, +1.0])
        self.z = np.zeros((trials * timesteps + 1, nb_kc), dtype=self.w_p2k.dtype)

        self.w_k2m = np.array([[
            # d1,  d2,  d3,  m1,  m2,  m3
            [+0., +0., +0., +1., +1., +1.]
        ] * nb_kc] * (trials * timesteps + 1))
        self.w_u2d = np.array([[
            # d1,  d2,  d3,  m1,  m2,  m3
            [+1., +1., +0., +0., +0., +0.],
            [+1., +1., +0., +0., +0., +0.]
        ]] * (trials * timesteps + 1))
        self.w_u2d[0] *= 0.
        self.k2m_init = self.w_k2m[0, 0]
        self.b_k2m = np.array([0.] * 3 + [1.] * 3)

        # Synaptic weights for model Ai
        self.w1_m2v = np.array([  # M to V_m
            # d1,  d2,  d3,  m1,  m2,  m3
            [-0., -0., -0., -0., -0., -0.],  # d1: PPL1-γ1ped
            [-0., -0., -0., -0., -0., -0.],  # d2: PPL1-γ2α'1
            [-0., -0., -0., -0., -0., -0.],  # d3: PAM-β'2'a
            [-1., -0., -0., -0., -0., -.5],  # m1: MBON-γ1ped
            [+0., +0., +0., +0., +0., +0.],  # m2: MBON-γ2α'1
            [+0., +0., +0., +0., +0., +0.]  # m3: MBON-γ5β'2a
        ])
        self.w1_d2k = np.array([  # V_d to W_k2m
            # d1,  d2,  d3,  m1,  m2,  m3
            [-0., -0., -0., -1., -0., -0.],  # d1: PPL1-γ1ped
            [-0., -0., -0., -0., -0., -0.],  # d2: PPL1-γ2α'1
            [-0., -0., -0., -0., -0., -0.],  # d3: PAM-β'2'a
            [-0., -0., -0., -0., -0., -0.],  # m1: MBON-γ1ped
            [+0., +0., +0., +0., +0., +0.],  # m2: MBON-γ2α'1
            [+0., +0., +0., +0., +0., +0.]  # m3: MBON-γ5β'2a
        ])

        # Synaptic weights for model Aii
        self.w2_m2v = np.array([  # M to V_m
            # d1,  d2,  d3,  m1,  m2,  m3
            [-0., -0., -0., -0., -0., -0.],  # d1: PPL1-γ1ped
            [-0., -0., -0., -0., -0., -0.],  # d2: PPL1-γ2α'1
            [-0., -0., -0., -0., -0., -0.],  # d3: PAM-β'2'a
            [-0., -0., -0., -0., -0., -0.],  # m1: MBON-γ1ped
            [+0., +0., +1., +0., +0., +0.],  # m2: MBON-γ2α'1
            [+0., +0., +0., +0., +0., +0.]  # m3: MBON-γ5β'2a
        ])
        self.w2_d2k = np.array([  # V_d to W_k2m
            # d1,  d2,  d3,  m1,  m2,  m3
            [-0., -0., -0., -0., -0., -0.],  # d1: PPL1-γ1ped
            [-0., -0., -0., -0., -1., -0.],  # d2: PPL1-γ2α'1
            [-0., -0., -0., -0., -0., -1.],  # d3: PAM-β'2'a
            [-0., -0., -0., -0., -0., -0.],  # m1: MBON-γ1ped
            [+0., +0., +0., +0., +0., +0.],  # m2: MBON-γ2α'1
            [+0., +0., +0., +0., +0., +0.]  # m3: MBON-γ5β'2a
        ])
        self.w_k2a = np.ones((nb_kc, nb_apl), dtype=float) / nb_kc
        if nb_apl > 0:
            self.w_d2a = np.array([[-1., -1., -1., 0., 0., 0.]] * nb_apl).T / 3.
        else:
            self.w_d2a = np.zeros((6, nb_apl), dtype=float)
        self.w_a2k = -np.ones((nb_apl, nb_kc), dtype=float)
        self.w_a2d = np.array([[-1., -1., -1., 0., 0., 0.]] * nb_apl)

        self.nb_pn = nb_pn
        self.nb_kc = nb_kc
        self.nb_mbon = 3
        self.nb_dan = 3

        self.csa = np.array([1., 0.])
        self.csb = np.array([0., 1.])
        self.usa = np.array([1., 0.])
        self.usb = np.array([0., 1.])
        self._mask = np.ones((nb_kc, 1), dtype=self.w_k2m.dtype)
        self.__shock_on = []
        self.__cs_on = []

        self.names = ["PPL1-γ1ped", "PPL1-γ2α'1", "PAM-β'2a",
                      "MBON-γ1ped", "MBON-γ2α'1", "MBON-γ5β'2a"]

    @property
    def w_m2v(self):
        w = np.eye(6)
        if self.__model_1:
            w += self.w1_m2v
        if self.__model_2:
            w += self.w2_m2v
        return w

    @property
    def w_d2k(self):
        w = 0.
        if self.__model_1:
            w += self.w1_d2k
        if self.__model_2:
            w += self.w2_d2k
        return w

    def _a(self, v, v_max=None, v_min=None):
        if v_max is None:
            v_max = self.v_max
        if v_min is None:
            v_min = self.v_min
        return leaky_relu(v, alpha=self.__leak, v_max=v_max, v_min=v_min)
    #
    # def __call__(self, *args, **kwargs):
    #     self._t = 0
    #     self.r[0] = self._v[0][0] - self._v[0][1]
    #     mb = self.w_k2m[0, 0].copy()
    #     for trial in range(1, 10):
    #         for cs, us_ in zip([self.csa, self.csb], [self.usa, self.usb]):
    #             # stop when the trial iteration exceeds the limit
    #             if self._t >= self.nb_trials * self.nb_timesteps:
    #                 break
    #             for timestep in range(self.nb_timesteps):
    #                 print("Trial: %d / %d" % (self._t, 1 + self.nb_trials * self.nb_timesteps))
    #                 # shock is presented only after the 4th sec of the trial
    #                 us = us_ * float(4 <= 5 * (timestep + 1) / self.nb_timesteps)
    #
    #                 # create dynamic memory for repeating loop
    #                 w_k2m_pre, w_k2m_post = self.w_k2m[self._t].copy(), self.w_k2m[self._t].copy()
    #                 v_pre, v_post = self._v[self._t].copy(), self._v[self._t].copy()
    #
    #                 # feed forward responses: PN(CS) -> KC
    #                 k = cs.dot(self.w_p2k)
    #                 # if self.__normalised:
    #                 #     k -= k.mean()
    #                 # k = self._a(k, v_max=self.v_max.max(), v_min=self.v_min.min())
    #                 print("CS/US", cs, us)
    #                 print("KC", k)
    #
    #                 if self.__eligibility_trace:
    #                     self.z[self._t+1] = self.z[self._t] * self.__lambda
    #                     self.z[self._t+1][k > 0] = 1.
    #
    #                 for e in range(self.repeats):
    #                     print("\nTRIAL %d %s" % (self._t // 2 + 1, ["CS-", "CS+"][self._t % 2]))
    #
    #                     # feed forward responses: KC -> MBON, US -> DAN
    #                     mb = k.dot(w_k2m_pre) + us.dot(self.w_u2d[self._t]) + self.v_init * float(self.__no_value)
    #                     print("MB", mb)
    #
    #                     # Step 1: internal values update
    #                     v_post = self.update_values(k, v_pre, mb)
    #                     print("POST", v_post[0])
    #
    #                     # Step 2: synaptic weights update
    #                     w_k2m_post = self.update_weights(k, v_post, w_k2m_post, z=self.z[self._t+1])
    #
    #                     # update dynamic memory for repeating loop
    #                     v_pre, w_k2m_pre = v_post, w_k2m_post
    #
    #                 if self.repeats > 0:
    #                     # store values and weights in history
    #                     self._v[self._t + 1], self.w_k2m[self._t + 1], self.md[self._t + 1] = (
    #                         v_post, w_k2m_post, np.array(k[..., np.newaxis] > 0, dtype=mb.dtype).dot(mb[np.newaxis]))
    #
    #                     # calculate responses
    #                     self.r[self._t + 1] = self.get_response(k, self._t + 1)
    #
    #                     # calculate weights difference
    #                     self.w_diff[self._t + 1] = self.integrate(self.w_k2m[self._t + 1], k)
    #
    #                 self._t += 1

    def __call__(self, *args, **kwargs):
        self.r[0] = self._v[0][0] - self._v[0][1]
        mb = self.w_k2m[0, 0].copy()

        if kwargs.get('extinction', False):
            routine = self.__extinction_routine()
        elif kwargs.get('reversal', True):
            routine = self.__reversal_routine()
        else:
            routine = []

        w_k2m_pre = self.w_k2m[0].copy()
        w_k2m_post = self.w_k2m[0].copy()
        v_pre = self._v[0].copy()
        k = 0

        for _, _, repeat, cs, us in routine:

            if repeat == 0:
                # create dynamic memory for repeating loop
                w_k2m_pre, w_k2m_post = self.w_k2m[self._t].copy(), self.w_k2m[self._t].copy()
                v_pre, v_post = self._v[self._t].copy(), self._v[self._t].copy()

                # feed forward responses: PN(CS) -> KC
                k = cs @ self.w_p2k

                if self.__eligibility_trace:
                    self.z[self._t + 1] = self.z[self._t] * self.__lambda
                    self.z[self._t + 1][k > 0] = 1.

            # feed forward responses: KC -> MBON, US -> DAN
            mb = k @ w_k2m_pre + us @ self.w_u2d[self._t] + self.v_init * float(self.__no_value)

            # Step 1: internal values update
            v_post = self.update_values(k, v_pre, mb)

            # Step 2: synaptic weights update
            w_k2m_post = self.update_weights(k, v_post, w_k2m_post, z=self.z[self._t + 1])

            # update dynamic memory for repeating loop
            v_pre, w_k2m_pre = v_post, w_k2m_post

            if self.repeats > 0 and repeat >= self.repeats - 1:
                # store values and weights in history
                self._v[self._t + 1], self.w_k2m[self._t + 1], self.md[self._t + 1] = (
                    v_post, w_k2m_post, np.array(k[..., np.newaxis] > 0, dtype=mb.dtype).dot(mb[np.newaxis]))
                # self._v_apl[self._t + 1] = v_apl_post

                # calculate responses
                self.r[self._t + 1] = self.get_response(k, self._t + 1)

                # calculate weights difference
                self.w_diff[self._t + 1] = self.integrate(self.w_k2m[self._t + 1], k)

    def __reversal_routine(self):
        self.__cs_on = np.arange(self.nb_trials * 2)
        self.__us_on = np.array([3, 5, 7, 9, 11, 14, 16, 18, 20, 22, 24])
        self.__routine_name = "reversal"
        return self.__routine(odour=self.__cs_on, shock=self.__us_on)

    def __extinction_routine(self):
        self.__cs_on = np.arange(self.nb_trials * 2)
        self.__us_on = np.array([3, 5, 7, 9, 11])
        self.__routine_name = "extinction"
        return self.__routine(odour=self.__cs_on, shock=self.__us_on)

    def __routine(self, odour=None, shock=None):
        self._t = 0
        if odour is None:
            odour = np.arange(self.nb_trials * 2)
        if shock is None:
            shock = np.arange(self.nb_trials * 2)

        for trial in range(1, self.nb_trials // 2 + 2):
            for cs_, us_ in zip([self.csa, self.csb], [self.usa, self.usb]):
                if self._t >= self.nb_trials * self.nb_timesteps:
                    break

                trial_ = self._t // self.nb_timesteps

                # odour is presented only in specific trials
                cs = cs_ * float(trial_ in odour)

                # shock is presented only in specific trials
                us__ = us_ * float(trial_ in shock)

                for timestep in range(self.nb_timesteps):

                    # shock is presented only after the 4th sec of the trial
                    us = us__ * float(4 <= 5 * (timestep + 1) / self.nb_timesteps)

                    if self.__verbose:
                        print("Trial: %d / %d, %s" % (trial_ + 1, self.nb_trials, ["CS-", "CS+"][self._t % 2]))
                    for e in range(self.repeats):
                        yield trial, timestep, e, cs, us

                    self._t += 1

    def update_weights(self, kc, v, w_k2m, z=None):
        eta_f = 1 - self.c_forgetting  # forgetting rate
        eta_w = np.sqrt(self.eta_w / float(self.repeats * self.nb_timesteps))

        # make sure that we keep the sign of the weights ('-' for depression, #+# for potentiation)
        if self.__eligibility_trace and z is not None:
            gat = z[..., np.newaxis]
        else:
            gat = kc[..., np.newaxis]

        if self.nb_apl > 0:
            v_apl = self._v_apl[self._t + 1] = gat[..., 0] @ self.w_k2a
            gat += (v_apl @ self.w_a2k)[..., np.newaxis]
        elif self.__normalised:
            gat -= gat.mean()
        else:
            gat = np.maximum(gat, 0)  # do not learn negative values

        dop = np.maximum(v, 0).dot(self.w_d2k)

        if self.__learning_rule in ["hybrid"]:
            # When DAN > 0 and KC > 0 : w <-- w_rest-
            # When DAN = 0 and KC > 0 : w <-- w_rest
            # When DAN > 0 and KC = 0 : w <-- w_rest+
            # When DAN = 0 and KC = 0 : w <-- w
            k = gat
            d = -dop
            b = self.k2m_init
            # print(k.shape, d.shape, b.shape)

            w_new = (k * (b - d) +
                     (1 - k) * d * (b + d) +
                     (1 - k) * (1 - d) * w_k2m)
        elif self.__learning_rule in ["hybrid-2"]:
            # When DAN > 0 and KC > 0 : w <-- w_rest-
            # When DAN = 0 and KC > 0 : w <-- w_rest
            # When DAN > 0 and KC = 0 : w <-- w_rest
            # When DAN = 0 and KC = 0 : w <-- w
            k = gat
            d = -dop
            b = self.k2m_init
            # print(k.shape, d.shape, b.shape)

            w_new = (k * (b - d) +
                     (1 - k) * d * b +
                     (1 - k) * (1 - d) * w_k2m)
        elif self.__learning_rule in ["hybrid-3"]:
            w_new = w_k2m + eta_w * dop * (gat + dop + w_k2m - self.k2m_init)
        elif self.__learning_rule in ["hybrid-4"]:
            w_new = w_k2m + eta_w * (dop * gat + (dop - gat) * (w_k2m - self.k2m_init))
        elif self.__learning_rule in ["dan-based"]:
            # When DAN > 0 and KC > W - W_rest increase the weight (if DAN < 0 it is reversed)
            # When DAN > 0 and KC < W - W_rest decrease the weight (if DAN < 0 it is reversed)
            # When DAN = 0 no learning happens
            w_new = w_k2m + eta_w * dop * (gat + w_k2m - self.k2m_init)
        elif self.__learning_rule in ["kc-based", "default"]:
            # When KC > 0 and DAN > W - W_rest increase the weight (if KC < 0 it is reversed)
            # When KC > 0 and DAN < W - W_rest decrease the weight (if KC < 0 it is reversed)
            # When KC = 0 no learning happens
            w_new = w_k2m + eta_w * gat * (dop - w_k2m + self.k2m_init)
        else:
            w_new = w_k2m

        # U2D plasticity
        if self.__us_plasticity:
            self.w_u2d = np.maximum(self.w_u2d - .1 * v.mean(axis=0), 0)

        # degrade the bias based on the current activity
        if self.__bias_degrade:
            self.v_init -= eta_f * v.mean(axis=0) * np.array([1., 1., 1., 0., 0., 0.])

        # weights degrade
        if self.__weights_degrade:
            w_new = self.c_forgetting * w_new + eta_f * self.k2m_init

        # weights constraint
        if self.__weights_constraints:
            w_new = np.clip(w_new, self.w_min, self.w_max)
        return np.maximum(w_new, 0)

    def update_values(self, kc, v, mb):
        reduce = 1. / float(self.repeats * self.nb_timesteps)
        eta_v = np.power(self.eta_v * reduce, reduce)

        if self.nb_apl > 0 and False:
            v_apl = self._v_apl[self._t+1] = np.maximum(kc @ self.w_k2a + (mb @ self.w_m2v - v[0]) @ self.w_d2a, 0)
            kc += v_apl @ self.w_a2k
            mb += v_apl @ self.w_a2d
        dim = np.maximum(kc[..., np.newaxis], 0)
        nr = mb
        # nr = np.minimum(mb, 0)
        if self.__unidimensional:
            dim = self._mask
        v_temp = (1 - eta_v) * v + eta_v * dim * (nr.dot(self.w_m2v) - v)
        return self._a(v_temp)

    def get_response(self, kc, t=None):
        if t is None:
            t = self._t
        return self.integrate(self._v[t], kc)

    def integrate(self, v, kc):
        return kc.dot(v) - (1 - kc).dot(v) * float(not self.__unidimensional)

    def plot(self, mode="trial", **kwargs):

        if mode == "trial":
            self._plot_trials(**kwargs)
        elif mode == "overlap":

            self._plot_overlap(**kwargs)
            # # build dataset
            # pattern = np.zeros(100)
            # pattern[25:50] = np.sin(np.linspace(0, np.pi, 25))
            #
            # ap, am, bp, bm = {}, {}, {}, {}
            # for j, name in enumerate(self.names):
            #     bp[name] = np.concatenate([v * pattern for v in self.r[1:, j]]).reshape((-1, 1))
            #     ap[name] = np.full_like(bp[name], np.nan)
            #     am[name] = np.full_like(bp[name], np.nan)
            #     bm[name] = np.full_like(bp[name], np.nan)
            #
            # df = pd.DataFrame({"A+": ap, "A-": am, "B+": bp, "B-": bm})
            # plot_overlap(df, experiment="B+")

    def _plot_trials(self, show_responses=False, show_values=True, show_cs=True, show_us=True, show_combined=False,
                     show_weights=True, show_eligibility_traces=True, show_internal_values=True):

        if not self.__eligibility_trace and show_eligibility_traces:
            show_eligibility_traces = False
        if self.nb_timesteps < 2:
            show_cs = show_us = False
            show_combined = True

        plt.figure(str(self) + "-responses", figsize=(6, 4))
        w_k2p = self.w_p2k.T  # / self.w_p2k.sum(axis=1)
        transform = lambda x, norm: np.tensordot(x, np.array(w_k2p > 0, dtype=float), axes=(1, 0)) / (
            np.sum(np.array(~np.isclose(w_k2p, 0), dtype=float), axis=0) if norm else 1.)
        # v = np.tensordot(self._v, w_k2p, axes=(1, 0))
        # w = np.tensordot(self.w_k2m, w_k2p, axes=(1, 0))
        # md = np.tensordot(self.md, w_k2p, axes=(1, 0))
        v = transform(self._v, norm=True)
        w = transform(self.w_k2m, norm=True)
        md = transform(self.md, norm=True)
        xlim = [.5, self.nb_trials-.5]
        ylim = [
            np.min([w.min() * float(show_weights), .5 * v.sum(axis=1).min() * float(show_values),
                    md.min() * float(show_internal_values), -1]) * 1.1,
            np.max([w.max() * float(show_weights), .5 * v.sum(axis=1).max() * float(show_values),
                    md.max() * float(show_internal_values), +1]) * 1.1
        ]

        nb_timesteps = np.minimum(self.nb_timesteps, 2)
        x_cs = np.r_[[[.75 + i, .75 + i, 1. + i, 1. + i] for i in self.__cs_on]]
        y_cs = np.r_[[ylim + ylim[::-1]] * len(self.__cs_on)]
        x_us = np.array([self.__us_on] * 2) + .9
        y_us = [[ylim[0]] * 7, [ylim[1]] * 7]
        xs = ((np.arange(nb_timesteps * self.nb_trials) + 1) / nb_timesteps).reshape((-1, nb_timesteps))
        xs[:, 0] += .25
        xs[:, 1] -= .1
        xw = ((np.arange(self.nb_timesteps * (self.nb_trials + 1))) / self.nb_timesteps).reshape((-1, self.nb_timesteps))
        for i in range(self.nb_timesteps):
            xw[:, i] += .75 - i / self.nb_timesteps + .25 * i / self.nb_timesteps
        xw = xw.flatten()[self.nb_timesteps-1:]

        nb_neurons = len(self.names)
        k = np.array([self.csa, self.csb]).dot(self.w_p2k) > 0

        for j, name in enumerate(self.names):
            plt.subplot(231 + j)
            plt.plot(xlim, [0, 0], c=(.8, .8, .8), lw=2)
            plt.fill_between(x_cs[0::2].flatten(), np.full_like(y_cs[0::2].flatten(), ylim[0]), y_cs[0::2].flatten(),
                             facecolor="C0", alpha=0.2)
            plt.fill_between(x_cs[1::2].flatten(), np.full_like(y_cs[1::2].flatten(), ylim[0]), y_cs[1::2].flatten(),
                             facecolor="C1", alpha=0.2)
            plt.plot(x_us, y_us, 'r-')

            cs_k = np.arange(self.nb_timesteps)[:int(np.round(.4 * self.nb_timesteps))]
            us_k = np.arange(self.nb_timesteps)[int(np.round(.4 * self.nb_timesteps)):]

            for i in range(2):
                x_i = xs[i::2].flatten()
                md_j_i = md[1:].reshape((-1, self.nb_timesteps, nb_neurons, 2))[i::2, :, j, i].flatten()
                v_j_i = v[1:].reshape((-1, self.nb_timesteps, nb_neurons, 2))[i::2, :, j, i].flatten()
                r_j_i = self.r[1:].reshape((-1, self.nb_timesteps, nb_neurons))[i::2, :, j].flatten()
                w_j_i = w[:, j, i]

                if show_internal_values:
                    plt.plot(x_i, md_j_i, 'C%d:' % i)

                if show_values:
                    if show_cs and not show_combined:
                        v_j_i_cs = v_j_i.reshape((-1, self.nb_timesteps))[:, cs_k].sum(axis=1).flatten()
                        plt.plot(x_i[0::2], v_j_i_cs, 'C%d-' % (i + 2))
                    if show_us and not show_combined:
                        v_j_i_us = v_j_i.reshape((-1, self.nb_timesteps))[:, us_k].sum(axis=1).flatten()
                        plt.plot(x_i[1::2], v_j_i_us, 'C%d--' % (i + 2))
                    if show_combined:
                        plt.plot(x_i, v_j_i, 'C%d-' % (i + 2))
                    if show_weights:
                        plt.plot(xw[1+2*i::], np.r_[w_j_i[0], w_j_i[(1+i):-(1+i)]], 'C%d--' % i)
                if show_responses:
                    plt.plot(x_i, r_j_i, 'C%d-' % (i + 4))
                    plt.plot(xs[i::].flatten(), np.r_[self.w_diff[0, j, i], w[(1+i):-2:2, j]], 'C%d--' % i)
                if show_eligibility_traces:
                    plt.plot(xs.flatten(), self.z[1:, k[i]], 'C%d-' % (i + 2), alpha=1. / self.w_p2k.shape[1])
            plt.title(name)
            plt.xticks(np.arange(self.nb_trials+1)+.5,
                       np.array([["", str(t + 1)] for t in range((self.nb_trials+1) // 2)]).flatten())
            plt.xlim(xlim)
            plt.ylim(ylim)
        plt.tight_layout()
        plt.show()

    def _plot_overlap(self, show_responses=False, show_values=True, show_cs=True, show_us=True, show_combined=False,
                     show_weights=True, show_eligibility_traces=True, show_internal_values=True, score=None):
        df = self.as_dataframe()
        plot_overlap(df, experiment="B+", title=str(self) + "-overlap", phase2=self.__routine_name, score=score)

    def as_dataframe(self):
        w_k2p = self.w_p2k.T
        transform = lambda x, norm: np.tensordot(x, np.array(w_k2p > 0, dtype=float), axes=(1, 0)) / (
            np.sum(np.array(~np.isclose(w_k2p, 0), dtype=float), axis=0) if norm else 1.)

        v = transform(self._v, norm=True)
        v = v[1:].reshape((-1, self.nb_timesteps, v.shape[1], v.shape[2]))

        pattern = np.exp(-np.square(np.arange(100) - 27.5) / 32)
        pattern[25:50] = 1.
        cs_pattern = np.exp(-np.square(np.arange(100) - 30) / 200) * pattern
        us_pattern = np.exp(-np.square(np.arange(100) - 45) / 2) * pattern

        ap, am, bp, bm = {}, {}, {}, {}
        for j, name in enumerate(self.names):
            bp[name] = np.concatenate([vv[0, 0] * cs_pattern + vv[1, 1] * us_pattern for vv in v[..., j, :]]).reshape((-1, 1))
            ap[name] = np.full_like(bp[name], np.nan)
            am[name] = np.full_like(bp[name], np.nan)
            bm[name] = np.full_like(bp[name], np.nan)

        return pd.DataFrame({"A+": ap, "A-": am, "B+": bp, "B-": bm})

    def copy(self):
        return copy(self)

    def __copy__(self):
        m = MBModel()
        m.repeats = copy(self.repeats)
        m.eta_w = copy(self.eta_w)
        m.eta_v = copy(self.eta_v)
        m.b_init = copy(self.b_init)
        m.vd_init = copy(self.vd_init)
        m.vm_init = copy(self.vm_init)
        m.nb_trials = copy(self.nb_trials)
        m.nb_timesteps = copy(self.nb_timesteps)
        m.c_forgetting = copy(self.c_forgetting)
        m.__leak = copy(self.__leak)
        m.__lambda = copy(self.__lambda)

        m._t = copy(self._t)
        m.__no_value = copy(self.__no_value)
        m.__unidimensional = copy(self.__unidimensional)
        m.__weights_degrade = copy(self.__weights_degrade)
        m.__bias_degrade = copy(self.__bias_degrade)
        m.__us_plasticity = copy(self.__us_plasticity)
        m.__normalised = copy(self.__normalised)
        m.__eligibility_trace = copy(self.__eligibility_trace)
        m.__constraints = copy(self.__constraints)
        m.__weights_constraints = copy(self.__weights_constraints)
        m.__model_1 = copy(self.__model_1)
        m.__model_2 = copy(self.__model_2)
        m.__routine_name = copy(self.__routine_name)
        m.__learning_rule = copy(self.__learning_rule)
        m.__verbose = copy(self.__verbose)
        m.nb_pn = self.nb_pn
        m.nb_kc = self.nb_kc
        m.nb_mbon = self.nb_mbon
        m.nb_dan = self.nb_dan

        m.w_p2k = copy(self.w_p2k)

        m._v = copy(self._v)
        m.r = copy(self.r)
        m.w_diff = copy(self.w_diff)
        m.v_init = copy(self.v_init)
        m.md = copy(self.md)

        m.v_max = copy(self.v_max)
        m.v_min = copy(self.v_min)
        m.w_max = copy(self.w_max)
        m.w_min = copy(self.w_min)
        m.z = copy(self.z)

        m.w_k2m = copy(self.w_k2m)
        m.w_u2d = copy(self.w_u2d)
        m.k2m_init = copy(self.k2m_init)
        m.b_k2m = copy(self.b_k2m)

        m.w1_m2v = copy(self.w1_m2v)
        m.w1_d2k = copy(self.w1_d2k)
        m.w2_m2v = copy(self.w2_m2v)
        m.w2_d2k = copy(self.w2_d2k)

        m._v_apl = copy(self._v_apl)
        m.nb_apl = copy(self.nb_apl)
        m.w_a2k = copy(self.w_a2k)
        m.w_a2d = copy(self.w_a2d)
        m.w_k2a = copy(self.w_k2a)
        m.w_d2a = copy(self.w_d2a)

        m.csa = copy(self.csa)
        m.csb = copy(self.csb)
        m.usa = copy(self.usa)
        m.usb = copy(self.usb)
        m._mask = copy(self._mask)
        m.__shock_on = copy(self.__shock_on)
        m.__cs_on = copy(self.__cs_on)

        m.names = copy(self.names)

        return m

    def __repr__(self):
        s = ""
        if self.__no_value:
            s += "novalue"
        else:
            s += "value"
        if self.__unidimensional:
            s += "-unidimensional"
        if self.__weights_degrade:
            s += "-weightsdegrade"
        if self.__bias_degrade:
            s += "-biasdegrade"
        if self.__us_plasticity:
            s += "-usplasticity"
        if self.__eligibility_trace:
            s += "-eligibility"
        if self.__constraints:
            s += "-constrained"
        if self.__weights_constraints:
            s += "-wconstrained"
        if self.__normalised:
            s += "-normalised"
        if self.nb_apl > 0:
            s += "-apl%d" % self.nb_apl
        if self.__learning_rule != "default":
            s += "-" + self.__learning_rule
        if self.__routine_name != "":
            s += "-" + self.__routine_name
        return s

    @classmethod
    def plot_overlap(cls, models: list, score=None):
        dfs, phases = [], []
        for mdl in models:
            dfs.append(mdl.as_dataframe())
            phases.append(mdl.__routine_name)
        title_comps = str(models[0]).split("-")[:-1] + ["overlap"]

        plot_overlap(dfs, experiment="B+", title="-".join(title_comps), phase2=phases, score=score)

    @classmethod
    def plot_timeline(cls, models: list = None, raw_data=None, integration_func=np.mean, score=None, target=None,
                      show_weights=True, show_values=True, show_cs=True, show_us=True, show_trace=False, nb_trials=None):

        cs_on = np.arange(56)
        us_on = np.array([3, 5, 7, 9, 11, 14, 16, 18, 20, 22, 24])
        if models is not None and raw_data is None:
            title_comps = str(models[0]).split("-")[:-1] + ["timeline"]
            data = cls._data_from_models(models, integration_func=integration_func)
            cs_on = models[0].__cs_on
            us_on = models[0].__us_on
        elif raw_data is not None:
            title_comps = ["timeline", "from", "data"]
            data = _data_from_raw(raw_data)
            show_weights = False
        else:
            raise AttributeError("Either 'model' or 'data' has to be given as input.")
        if nb_trials is not None:
            nb_trials *= 2

        y_min, y_max = None, None

        for name in data:
            ndata = data[name]

            vs = []  # Read the values in the data depending on the available phases
            if show_trace:
                if "reversal" in ndata and "v" in ndata["reversal"]:
                    vs += ndata["reversal"]["v"]
                if "extinction" in ndata and "v" in ndata["extinction"]:
                    vs += ndata["extinction"]["v"]
            else:
                if "reversal" in ndata and "v_cs" in ndata["reversal"]:
                    vs += ndata["reversal"]["v_cs"]
                if "reversal" in ndata and "v_cs" in ndata["reversal"]:
                    vs += ndata["reversal"]["v_us"]
                if "extinction" in ndata and "v_cs" in ndata["extinction"]:
                    vs += ndata["extinction"]["v_cs"]
                if "extinction" in ndata and "v_us" in ndata["extinction"]:
                    vs += ndata["extinction"]["v_us"]
            # transform the lists of data with uneven lengths into an even NumPy array filled with NaNs
            vs = _list2array(vs)
            if nb_trials is None:
                nb_trials = np.max([len(vss) for vss in vs]) * 2
            if len(vs) > 0:
                # compute the minimum and maximum values in the data
                v_min = np.nanmin(vs - vs[:, :1])
                v_max = np.nanmax(vs - vs[:, :1])
            else:
                v_min = v_max = 0.

            ws = []  # Read the weights in the data depending on the available phases
            if "reversal" in ndata and "w" in ndata["reversal"]:
                ws += ndata["reversal"]["w"]
            if "extinction" in ndata and "w" in ndata["extinction"]:
                ws += ndata["extinction"]["w"]
            ws = _list2array(ws)
            if len(ws) > 0:
                # compute the minimum and maximum weights in the data
                w_min = np.nanmin(ws - ws[:, :1])
                w_max = np.nanmax(ws - ws[:, :1])
            else:
                w_min = w_max = 0.

            t_min = np.min([w_min * float(show_weights), v_min * float(show_values), -1]) * 1.1
            t_max = np.max([w_max * float(show_weights), v_max * float(show_values), +1]) * 1.1

            if y_min is None or y_max is None:
                y_min = t_min
                y_max = t_max
            else:
                y_min = np.minimum(t_min, y_min)
                y_max = np.maximum(t_max, y_max)

        xlim = [.5, nb_trials - .5]
        ylim = [y_min, y_max]

        x_cs_ = np.r_[[[.75 + i, .75 + i, 1. + i, 1. + i] for i in cs_on]]
        y_cs_ = np.r_[[ylim + ylim[::-1]] * len(cs_on)]
        x_us_ = np.array([us_on] * 2) + .9
        y_us_ = np.array([[y_min] * 11, [y_max] * 11])

        plt.figure("-".join(title_comps), figsize=(10.5, 5.25))

        cs_mark = "-"
        us_mark = "--"
        w_mark = ":"
        for j, name in enumerate(data):
            smark = "-." if "extinction" in data[name] else "-"
            plt.subplot(231 + j)
            plt.plot(xlim, [0, 0], c=(.8, .8, .8), lw=2)
            plt.fill_between(x_cs_[0::2].flatten(), np.full_like(y_cs_[0::2].flatten(), ylim[0]),
                             y_cs_[0::2].flatten(),
                             facecolor="C0", alpha=0.2)
            plt.fill_between(x_cs_[1::2].flatten(), np.full_like(y_cs_[1::2].flatten(), ylim[0]),
                             y_cs_[1::2].flatten(),
                             facecolor="C1", alpha=0.2)
            plt.plot(x_us_[x_us_ < 12].reshape((2, -1)), y_us_[x_us_ < 12].reshape((2, -1)), 'r-', lw=.5)
            plt.plot(x_us_[x_us_ > 12].reshape((2, -1)), y_us_[x_us_ > 12].reshape((2, -1)), 'r%s' % smark, lw=.5)

            ndata = data[name]
            for mode in ndata:
                mdata = ndata[mode]
                shift = 0 if mode == "reversal" else 2

                for i in range(2):
                    amark = 'o'
                    emark = 's'
                    aw = 1.
                    ew = 1.
                    colour = 'C%d' % (i + shift)
                    if target is not None:
                        trg = target[name]["acquisition (%s)" % ["A", "B"][i]]
                        if np.isnan(trg):
                            amark = ','
                        elif trg > 0:
                            amark = '^'
                        elif trg < 0:
                            amark = 'v'
                        if not np.isnan(trg):
                            aw = np.absolute(trg)
                    if score is not None:
                        score_ = 1 - score[name]["acquisition (%s)" % ["A", "B"][i]]
                        plt.text(xlim[1] / 20, 1.05 * y_max, "A", horizontalalignment="center", fontsize=8)
                        plt.plot([xlim[1] / 20], [(0.9 - i / 6) * y_max],
                                 '%s%s' % (colour, amark), alpha=score_ * aw)
                    if show_values:
                        if target is not None:
                            trg = target[name]["%s (%s)" % (mode, ["A", "B"][i])]
                            if np.isnan(trg):
                                emark = ','
                            elif trg > 0:
                                emark = '^'
                            elif trg < 0:
                                emark = 'v'
                            if not np.isnan(trg):
                                ew = np.absolute(trg)
                        if score is not None:
                            score_ = 1 - score[name]["%s (%s)" % (mode, ["A", "B"][i])]
                            mode_ = 2 if mode == "reversal" else 3
                            plt.text(mode_ * xlim[1] / 20, 1.05 * y_max, ["R", "E"][mode_-2],
                                     horizontalalignment="center", fontsize=8)
                            plt.plot([mode_ * xlim[1] / 20], [(0.9 - i / 6) * y_max],
                                     '%s%s' % (colour, emark), alpha=score_ * ew)
                        if show_trace:
                            x_i, v_i = mdata["x"][i], mdata["v"][i]
                            if "s" in mdata:
                                s_i = mdata["s"][i]
                                plt.fill_between(x_i, v_i - s_i, v_i + s_i, facecolor=colour, alpha=0.2)
                            plt.plot(x_i, v_i, '%s%s' % (colour, cs_mark))
                        elif show_cs:
                            x_cs, v_cs = np.array(mdata["x_cs"][i]), np.array(mdata["v_cs"][i])
                            if "s_cs" in mdata:
                                s_cs = np.array(mdata["s_cs"][i]) / 2
                                plt.fill_between(x_cs, v_cs - s_cs, v_cs + s_cs, facecolor=colour, alpha=0.2)
                            plt.plot(x_cs, v_cs - v_cs[0], '%s%s' % (colour, cs_mark),
                                     label="Odour %s (%s phase)" % (["A", "B"][i], mode))
                        if show_us and not show_trace:
                            x_us, v_us = mdata["x_us"][i], mdata["v_us"][i]
                            plt.plot(x_us, v_us - v_us[0], '%s%s' % (colour, us_mark),
                                     label="Shock %s (%s phase)" % (["A", "B"][i], mode))
                        if show_weights:
                            xw, w_i = mdata["x_w"][i], mdata["w"][i]
                            plt.plot(xw[1 + 2 * i::], np.r_[w_i[0], w_i[(1 + i):-(1 + i)]],
                                     '%s%s' % (colour, w_mark),
                                     label="Weight (%s phase)" % mode)

            plt.ylabel(name)
            if j // 3 > 0:
                plt.xlabel("Trials")
            plt.xticks(np.arange(nb_trials + 1) + .5,
                       np.array([["", str(t + 1)] for t in range(6)] + [["", ""]] +
                                [[str(t + 1), ""] for t in range(6, (nb_trials + 2) // 2)]).flatten()[
                       :(nb_trials + 1)])
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.grid(axis='y')

        plt.subplot(234)
        leg = plt.legend(ncol=4, loc='upper left', bbox_to_anchor=(-0.03, -0.25))
        leg.set_in_layout(False)
        plt.tight_layout(rect=[None, .13, None, None])
        plt.show()

    @classmethod
    def _data_from_models(cls, models, integration_func=np.mean):
        transform = lambda x, w_, norm: np.tensordot(x, np.array(w_ > 0, dtype=float), axes=(1, 0)) / (
            np.sum(np.array(~np.isclose(w_, 0), dtype=float), axis=0) if norm else 1.)

        neurons = models[0].names
        nb_neurons = len(neurons)
        nb_timesteps = models[0].nb_timesteps

        cs_k = np.arange(nb_timesteps)[:int(np.round(.4 * nb_timesteps))]
        us_k = np.arange(nb_timesteps)[int(np.round(.4 * nb_timesteps)):]

        data = {}
        for m in models[::-1]:
            w_k2p = np.linalg.pinv(m.w_p2k)

            v = transform(m._v, w_k2p, norm=False)
            w = transform(m.w_k2m, w_k2p, norm=False) - m.k2m_init[np.newaxis, :, np.newaxis] * m.nb_kc / m.nb_pn

            nb_timesteps = np.maximum(m.nb_timesteps, 2)
            xw = ((np.arange(nb_timesteps * (m.nb_trials + 1))) / nb_timesteps).reshape(
                (-1, nb_timesteps))
            for i in range(m.nb_timesteps):
                xw[:, i] += .75 - i / nb_timesteps + .25 * i / nb_timesteps
            xw = xw.flatten()[nb_timesteps - 1:]
            xs = xw.copy() - 1
            mode = m.__routine_name

            for j, name in enumerate(neurons):
                ndata = data[name] = data.get(name, {})
                mdata = ndata[mode] = {}

                mdata["x"] = [None] * 2
                mdata["x_cs"] = [None] * 2
                mdata["x_us"] = [None] * 2
                mdata["x_w"] = [None] * 2
                mdata["v"] = [None] * 2
                mdata["v_cs"] = [None] * 2
                mdata["v_us"] = [None] * 2
                mdata["w"] = [None] * 2
                for i in range(2):
                    s, e = i, xs.shape[0] - 1 + i
                    mdata["x"][i] = x_i = xs[s:e].reshape((-1, nb_timesteps))[i::2].flatten()
                    mdata["x_cs"][i] = x_i[int(1 * nb_timesteps / 3)::nb_timesteps]
                    mdata["x_us"][i] = x_i[int(2 * nb_timesteps / 3)::nb_timesteps]
                    mdata["x_w"][i] = xw
                    mdata["v"][i] = v_i = v[1:].reshape((-1, nb_timesteps, nb_neurons, 2))[i::2, :, j, i].flatten()
                    mdata["v_cs"][i] = integration_func(v_i.reshape((-1, nb_timesteps))[:, cs_k], axis=1).flatten()
                    mdata["v_us"][i] = integration_func(v_i.reshape((-1, nb_timesteps))[:, us_k], axis=1).flatten()
                    mdata["w"][i] = w[:, j, i]
        return data


def leaky_relu(v, alpha=.8, v_max=np.inf, v_min=-np.inf):
    return np.clip(v, np.maximum(alpha * v, v_min), v_max)


def _data_from_raw(raw_data, integration_func=np.mean):
    from evaluation import _get_trend, _get_trial

    names = np.array(raw_data.index.sort_values())[[4, 5, 3, 0, 1, 2]]
    data = {}
    for name in names:
        ndata = data[name] = {}
        mdata = ndata["reversal"] = {
            "x": [None] * 2,
            "x_cs": [None] * 2,
            "x_us": [None] * 2,
            "v": [None] * 2,
            "v_cs": [None] * 2,
            "v_us": [None] * 2,
            "s": [None] * 2,
            "s_cs": [None] * 2,
            "s_us": [None] * 2
        }

        nb_timesteps = 100
        nb_trials = 17
        trials = [[], []]
        for trial in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
            # A-
            trial_0 = _get_trial(raw_data[name], trial, odour="A")
            if trial_0.shape[0] > 0 and trial_0[0].shape[0] > 0:
                trials[0].append(trial_0)

            # B+
            trial_1 = _get_trial(raw_data[name], trial, odour="B")
            if trial_1.shape[0] > 0 and trial_1[0].shape[0] > 0:
                trials[1].append(trial_1)

        xw = ((np.arange(nb_timesteps * (nb_trials + 1))) / nb_timesteps).reshape(
            (-1, nb_timesteps))
        # for i in range(nb_timesteps):
        #     xw[:, i] += .75 - i / nb_timesteps + .25 * i / nb_timesteps
        xw = xw.flatten()[nb_timesteps - 1:]
        xs = xw.copy() - 1

        for j in range(2):
            s, e = j, xs.shape[0] - 1 + j
            mdata["x"][j] = x_i = xs[s:e].reshape((-1, nb_timesteps))[j::2].flatten()
            mdata["x_cs"][j] = x_i[int(1 * nb_timesteps / 3)::nb_timesteps]
            mdata["x_us"][j] = x_i[int(2 * nb_timesteps / 3)::nb_timesteps]
            mdata["v_cs"][j] = []
            mdata["v_us"][j] = []
            mdata["v"][j] = []
            mdata["s_cs"][j] = []
            mdata["s_us"][j] = []
            mdata["s"][j] = []
            for i in range(len(trials[j])):
                trial_i = np.array(np.array(trials[j])[[0, i]])
                if len(trial_i) < 2:
                    v_cs, s_cs = np.nan, np.nan
                    v_us, s_us = np.nan, np.nan
                    v, s = np.nan, np.nan
                else:
                    v_cs, s_cs = _get_trend(trial_i, cs_only=True, us_only=False, integration=integration_func)
                    v_us, s_us = _get_trend(trial_i, cs_only=False, us_only=True, integration=integration_func)
                    v, s = _get_trend(trial_i, cs_only=False, us_only=False, integration=integration_func, axis=(0, 2))
                mdata["v_cs"][j].append(v_cs)
                mdata["v_us"][j].append(v_us)
                mdata["v"][j].append(v)
                mdata["s_cs"][j].append(s_cs)
                mdata["s_us"][j].append(s_us)
                mdata["s"][j].append(s)

    return data


def _list2array(v, fill_val=np.nan):
    if len(v) < 1:
        out = np.array(v)
    else:
        lens = np.array([len(vv) for vv in v])
        mask = lens[:, None] > np.arange(lens.max())
        out = np.full(mask.shape, fill_val)
        out[mask] = np.concatenate(v)
    return out


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


if __name__ == '__main__':
    from evaluation import evaluate, create_behaviour_map
    from imaging import load_draft_data
    # from evaluation import behaviour as target
    target = create_behaviour_map(cs_only=True)

    pd.options.display.max_columns = 6
    pd.options.display.max_rows = 6
    pd.options.display.width = 1000

    nb_kcs = 10
    repeat_all = False

    nb_kc1_best, nb_kc2_best = 0, 0
    val_best, acc_best, pre_best, models_best = 0, None, None, None
    vals = np.zeros((nb_kcs+1, nb_kcs+1))
    for kc1 in (range(nb_kcs+1) if repeat_all else [nb_kcs//2]):
        for kc2 in (range(nb_kcs+1) if repeat_all else [nb_kcs//2]):
            # model = MBModel()
            # model = MBModel(no_value=True)
            # model = MBModel(auto_degrade=True, degrade_factor=.8)
            # model = MBModel(unidimensional=True, normalise=False, vm_init=0., vd_init=0.)
            # model = MBModel(unidimensional=True, normalise=True, vm_init=0., vd_init=0.)
            model = MBModel(no_value=True, unidimensional=True, weights_degrade=False, bias_degrade=False,
                            lam=.0, timesteps=25, us_plasticity=False, constraints=False, verbose=False,
                            weight_constraints=False, normalise=False, nb_apl=1, learning_rule="kc-based",
                            # trials=17,
                            trials=28,
                            pn2kc_init="default",
                            # nb_kc=4, nb_kc_odour_1=2, nb_kc_odour_2=2)  # 0.1
                            nb_kc=nb_kcs, nb_kc_odour_1=kc1, nb_kc_odour_2=kc2)  # 0.4
                            # nb_kc=20, nb_kc_odour_1=10, nb_kc_odour_2=10)  # 0.2
            vals[kc1, kc2], acc, prediction, models = evaluate(model, tolerance=.02, percentage=True, behav=target,
                                                               cs_only=True, mbon_only=False,
                                                               reversal=True, extinction=True)
            val = vals[kc1, kc2]
            if val > val_best:
                val_best = val
                acc_best = acc
                pre_best = prediction
                models_best = models
                nb_kc1_best = kc1
                nb_kc2_best = kc2
            print("Odour A: %d, Odour B: %d -- Score: %.4f" % (kc1, kc2, val * 100))

    # best_vals = np.full_like(vals, np.nan)
    # best_vals[vals == val_best] = 1
    # plt.figure("kcs-overlap", figsize=(6, 5))
    # plt.imshow(vals * 100, vmin=0, vmax=100, cmap="Greys", origin='lower')
    # cbar = plt.colorbar()
    # plt.imshow(best_vals, cmap="Reds_r", origin='lower')
    # plt.ylabel("#KCs for odour A")
    # plt.xlabel("#KCs for odour B")
    # cbar.ax.set_ylabel("Score (%)")
    # plt.tight_layout()
    # plt.show()

    # if repeat_all:
    #     plot_relation(vals)

    print("TARGET")
    print(target.T)
    print("PREDICTION")
    print(pre_best.T)
    print("ACCURACY")
    print(acc_best.T)
    print("Score: %.2f" % val_best)
    print("#KC1: %d, #KC2: %d" % (nb_kc1_best, nb_kc2_best))

    # MBModel.plot_overlap(models_best, score=acc_best)
    MBModel.plot_timeline(models=models_best, score=acc_best, target=target, nb_trials=13)
    # MBModel.plot_timeline(raw_data=load_draft_data()["B+"], nb_trials=13)
    # for model in models:
    #     # model.plot("trial", show_internal_values=False)
    #     model.plot("overlap", show_internal_values=False, score=acc)
