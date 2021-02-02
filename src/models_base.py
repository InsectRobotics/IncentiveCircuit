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
        self.__routine_name = ""
        self.__learning_rule = learning_rule
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
        self.w_diff = np.zeros(((trials * timesteps + 1), nb_dan + nb_mbon), dtype=self._v.dtype)
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

    def _a(self, v, v_max=None, v_min=None):
        if v_max is None:
            v_max = self.v_max
        if v_min is None:
            v_min = self.v_min
        return leaky_relu(v, alpha=self.__leak, v_max=v_max, v_min=v_min)

    def __call__(self, *args, **kwargs):
        if kwargs.get('no_shock', False):
            routine = self.__no_shock_routine()
        elif kwargs.get('unpaired', False):
            routine = self.__unpaired_routine()
        elif kwargs.get('reversal', True):
            routine = self.__reversal_routine()
        else:
            routine = []
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

            # calculate weights difference
            self.w_diff[self._t + 1] = self.integrate(self.w_k2m[self._t + 1], k)

    def __reversal_routine(self):
        self.__cs_on = np.arange(self.nb_trials * 2)
        self.__us_on = np.array([3, 5, 7, 9, 11, 14, 16, 18, 20, 22, 24])
        self.__routine_name = "reversal"
        return self.__routine(odour=self.__cs_on, shock=self.__us_on)

    def __unpaired_routine(self):
        self.__cs_on = np.arange(self.nb_trials * 2)
        self.__us_on = np.array([3, 5, 7, 9, 11, 14, 16, 18, 20, 22, 24])
        self.__routine_name = "unpaired"
        return self.__routine(odour=self.__cs_on, shock=self.__us_on, paired=[3, 5, 7, 9, 11])
        # return self.__routine(odour=self.__cs_on, shock=self.__us_on, paired=[2, 3, 4, 5, 6])

    def __no_shock_routine(self):
        self.__cs_on = np.arange(self.nb_trials * 2)
        self.__us_on = np.array([3, 5, 7, 9, 11])
        self.__routine_name = "no shock"
        return self.__routine(odour=self.__cs_on, shock=self.__us_on)

    def __routine(self, odour=None, shock=None, paired=None):
        self._t = 0
        if odour is None:
            odour = np.arange(self.nb_trials * 2)
        if shock is None:
            shock = np.arange(self.nb_trials * 2)
        if paired is None:
            paired = np.arange(self.nb_trials * 2)

        for trial in range(1, self.nb_trials // 2 + 2):
            for cs_ in [self.csa, self.csb]:
                if self._t >= self.nb_trials * self.nb_timesteps:
                    break

                trial_ = self._t // self.nb_timesteps

                # odour is presented only in specific trials
                cs__ = cs_ * float(trial_ in odour)

                # shock is presented only in specific trials
                us__ = np.zeros(self.us_dims, dtype=float)
                if self.us_dims > 2:
                    us__[4] = float(trial_ in shock)
                else:
                    us__[1] = float(trial_ in shock)

                for timestep in range(self.nb_timesteps):

                    # we skip odour in the first timestep of the trial
                    cs = cs__ * float(timestep > 0)
                    if trial_ in paired:
                        # shock is presented only after the 4th sec of the trial
                        us = us__ * float(4 <= 5 * (timestep + 1) / self.nb_timesteps)
                    else:
                        us = us__ * float(timestep < 1)
                    # print(self.__routine_name, trial, timestep, cs, us)
                    if self.__verbose:
                        print("Trial: %d / %d, %s" % (trial_ + 1, self.nb_trials, ["CS-", "CS+"][self._t % 2]))
                    yield trial, timestep, cs, us

                    self._t += 1

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

        if self.__learning_rule in ["dopaminergic", "dlr", "dan-base", "default"]:
            # When DAN > 0 and KC > W - W_rest increase the weight (if DAN < 0 it is reversed)
            # When DAN > 0 and KC < W - W_rest decrease the weight (if DAN < 0 it is reversed)
            # When DAN = 0 no learning happens
            w_new = w_k2m + eta_w * D * (k + w_k2m - self.k2m_init)
        elif self.__learning_rule in ["rescorla-wagner", "rw", "kc-based"]:
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

    def integrate(self, v, kc):
        return kc.dot(v)

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
        if self.__learning_rule != "default":
            s += ", lr='" + self.__learning_rule + "'"
        if self.nb_apl > 0:
            s += ", apl=%d" % self.nb_apl
        s += ")"
        return s

    def as_dataframe(self, nids=None, reconstruct=True):
        if nids is None:
            nids = np.arange(self._v.shape[1])
        v = self._v.copy()[1:, nids].reshape((-1, self.nb_timesteps, len(nids)))

        pattern = np.exp(-np.square(np.arange(100) - 27.5) / 32)
        pattern[25:50] = 1.
        cs_pattern = np.exp(-np.square(np.arange(100) - 30) / 200) * pattern
        us_pattern = np.exp(-np.square(np.arange(100) - 45) / 2) * pattern

        ap, am, bp, bm = {}, {}, {}, {}
        for j, name in enumerate(np.array(self.names)[nids]):
            if reconstruct:
                bp[name] = np.concatenate([vv[0] * cs_pattern + vv[1] * us_pattern for vv in v[..., j]]).reshape((-1, 1))
                ap[name] = np.full_like(bp[name], np.nan)
                am[name] = np.full_like(bp[name], np.nan)
                bm[name] = np.full_like(bp[name], np.nan)
            else:
                bp[name] = v[..., j]
                ap[name] = np.full_like(bp[name], np.nan)
                bm[name] = np.full_like(bp[name], np.nan)
                ap[name] = np.full_like(bp[name], np.nan)

        return pd.DataFrame({"A+": ap, "A-": am, "B+": bp, "B-": bm})

    @classmethod
    def _data_from_models(cls, models, nids=None, integration_func=np.mean):
        transform = lambda x, w_, norm: np.tensordot(x, np.array(w_ > 0, dtype=float), axes=(1, 0)) / (
            np.sum(np.array(~np.isclose(w_, 0), dtype=float), axis=0) if norm else 1.)

        if nids is None:
            nids = np.arange(len(models[0].names))
        neurons = np.array(models[0].names)[nids]
        nb_neurons = len(neurons)
        nb_timesteps = models[0].nb_timesteps

        cs_k = np.arange(nb_timesteps)[:int(np.round(.4 * nb_timesteps))]
        us_k = np.arange(nb_timesteps)[int(np.round(.4 * nb_timesteps)):]

        data = {}
        for m in models[::-1]:
            w_k2p = np.linalg.pinv(m.w_p2k)

            v = m._v[:, nids].copy()
            w = (transform(m.w_k2m[..., nids], w_k2p, norm=False) * m.nb_pn / m.nb_kc -
                 m.k2m_init[np.newaxis, nids, np.newaxis])

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
                    mdata["v"][i] = v_i = v[1:].reshape((-1, nb_timesteps, nb_neurons))[i::2, :, j].flatten()
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
