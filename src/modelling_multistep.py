from imaging import plot_overlap
from modelling import synapse_counts

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class MBModel(object):
    def __init__(self, eta=.7, vd_init=0., vm_init=1., trials=17, iterations=1, repeats=1):
        self.repeats = repeats
        self.eta = eta
        self.b_init = 1.
        self.vd_init = vd_init * self.b_init
        self.vm_init = vm_init * self.b_init
        self.nb_trials = trials
        self.nb_iterations = iterations
        self._t = 0

        d1 = [vd_init, vd_init]  # PPL1-γ1ped
        d2 = [vd_init, vd_init]  # PPL1-γ2α'1
        d3 = [vd_init, vd_init]  # PAM-β'2a

        m1 = [vm_init, vm_init]  # MBON-γ1ped
        m2 = [vm_init, vm_init]  # MBON-γ2α'1
        m3 = [vm_init, vm_init]  # MBON-γ5β'2a

        self._v = np.array([np.array([d1, d2, d3, m1, m2, m3]).T] * (trials * iterations + 1))
        self.r = np.zeros(((trials * iterations + 1), 6), dtype=self._v.dtype)
        self.w_diff = np.zeros(((trials * iterations + 1), 6), dtype=self._v.dtype)
        self.v_init = self._v[0, 0]
        self.md = self._v.copy()

        self.w_k2m = np.array([[
            # d1,  d2,  d3,  m1,  m2,  m3
            [+0., +0., +0., +1., +1., +1.],
            [+0., +0., +0., +1., +1., +1.]
        ]] * (trials * iterations + 1))
        self.w_u2d = np.array([[
            # d1,  d2,  d3,  m1,  m2,  m3
            [+1., +1., +0., +0., +0., +0.],
            [+1., +1., +0., +0., +0., +0.]
        ]] * (trials * iterations + 1))
        for t in [0, 1, 2, 4, 6, 8, 10, 12, 13, 15]:
            self.w_u2d[t] *= 0.
        self.k2m_init = self.w_k2m[0, 0]

        # Synaptic weights for model Ai
        self.w1_m2v = np.array([  # M to V_m
            # d1,  d2,  d3,  m1,  m2,  m3
            [-0., -0., -0., -0., -0., -0.],  # d1: PPL1-γ1ped
            [-0., -0., -0., -0., -0., -0.],  # d2: PPL1-γ2α'1
            [-0., -0., -0., -0., -0., -0.],  # d3: PAM-β'2'a
            [-1., -0., -0., -0., -0., -1.],  # m1: MBON-γ1ped
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

        self.csa = np.array([1., 0.])
        self.csb = np.array([0., 1.])
        self.usa = np.array([1., 0.])
        self.usb = np.array([0., 1.])

        self.names = ["PPL1-γ1ped", "PPL1-γ2α'1", "PAM-β'2a",
                      "MBON-γ1ped", "MBON-γ2α'1", "MBON-γ5β'2a"]

    @property
    def w_m2v(self):
        return np.eye(6) + self.w1_m2v + self.w2_m2v

    @property
    def w_d2k(self):
        return self.w1_d2k + self.w2_d2k

    def __call__(self, *args, **kwargs):
        self._t = 0
        self.r[0] = self._v[0][0] - self._v[0][1]
        cs_pattern = np.zeros((self.nb_iterations, 1), dtype=float)
        cs_pattern[int(np.floor(.25 * self.nb_iterations)):int(np.ceil(.50 * self.nb_iterations))] = 1.
        us_pattern = np.zeros((self.nb_iterations, 1), dtype=float)
        us_pattern[int(np.floor(.48 * self.nb_iterations)):int(np.ceil(.49 * self.nb_iterations))] = 1.
        mb = self.w_k2m[0, 0].copy()
        for trial in range(1, 10):
            for cs_id, us_id in zip([self.csa, self.csb], [self.usa, self.usb]):
                for cs, us in zip(cs_pattern * cs_id, us_pattern * us_id):
                    # stop when the trial iteration exceeds the limit
                    if self._t >= self.nb_trials * self.nb_iterations:
                        break

                    # create dynamic memory for repeating loop
                    w_k2m_pre, w_k2m_post = self.w_k2m[self._t].copy(), self.w_k2m[self._t].copy()
                    v_pre, v_post = self._v[self._t].copy(), self._v[self._t].copy()

                    for e in range(self.repeats):
                        # feed forward responses: KC -> MBON, US -> DAN
                        mb = cs.dot(w_k2m_pre) + us.dot(self.w_u2d[self._t])

                        # Step 1: internal values update
                        v_post = self.update_values(cs, v_pre, mb)

                        # Step 2: synaptic weights update
                        w_k2m_post = self.update_weights(cs, v_post, w_k2m_post)

                        # update dynamic memory for repeating loop
                        v_pre, w_k2m_pre = v_post, w_k2m_post

                    if self.repeats > 0:
                        # store values and weights in history
                        self._v[self._t + 1], self.w_k2m[self._t + 1], self.md[self._t + 1] = (
                            v_post, w_k2m_post, cs[..., np.newaxis].dot(mb[np.newaxis]))

                        # calculate responses
                        self.r[self._t + 1] = self.get_response(cs, self._t + 1)

                        # calculate weights difference
                        self.w_diff[self._t + 1] = self.integrate(self.w_k2m[self._t + 1], cs)

                    self._t += 1

    def update_weights(self, cs, v, w_k2m):
        return np.maximum(w_k2m + self.eta * cs[..., np.newaxis] * (v.dot(self.w_d2k) - w_k2m + self.k2m_init), 0)

    def update_values(self, cs, v, mb):
        v_temp = v + self.eta * cs[..., np.newaxis] * (mb.dot(self.w_m2v) - v + self.v_init)
        return leaky_relu(v_temp)

    def get_response(self, cs, t=None):
        if t is None:
            t = self._t
        return self.integrate(self._v[t], cs)

    @staticmethod
    def integrate(v, cs):
        return cs.dot(v) - (1 - cs).dot(v)

    def plot(self, mode="trial", **kwargs):

        if mode == "trial":
            self._plot_model(**kwargs)
        elif mode == "overlap":
            # build dataset
            pattern = np.zeros(100)
            pattern[25:50] = np.sin(np.linspace(0, np.pi, 25))

            ap, am, bp, bm = {}, {}, {}, {}
            for j, name in enumerate(self.names):
                bp[name] = np.concatenate([v * pattern for v in self.r[1:, j]]).reshape((-1, 1))
                ap[name] = np.full_like(bp[name], np.nan)
                am[name] = np.full_like(bp[name], np.nan)
                bm[name] = np.full_like(bp[name], np.nan)

            df = pd.DataFrame({"A+": ap, "A-": am, "B+": bp, "B-": bm})
            plot_overlap(df, experiment="B+")

    def _plot_model(self, responses=True, values=False):
        ni = self.nb_iterations
        plt.figure("responses", figsize=(6, 4))
        ylim = [-2, np.maximum(3 * self.b_init, 1)]
        for j, name in enumerate(self.names):
            plt.subplot(231 + j)
            xs = np.r_[[[1.25 + i, 1.25 + i, 1.50 + i, 1.50 + i] for i in range(self.nb_trials)]]
            ys = np.r_[[ylim + ylim[::-1]] * self.nb_trials]
            plt.fill_between(xs[0::2].flatten(), np.full_like(ys[0::2].flatten(), ylim[0]), ys[0::2].flatten(),
                             facecolor="C0", alpha=0.2)
            plt.fill_between(xs[1::2].flatten(), np.full_like(ys[1::2].flatten(), ylim[0]), ys[1::2].flatten(),
                             facecolor="C1", alpha=0.2)
            plt.plot(np.array([[4, 6, 8, 10, 12, 15, 17]] * 2) - .1, [[ylim[0]] * 7, [ylim[1]] * 7], 'r-')
            ai = np.concatenate([(i * ni + np.arange(ni)) for i in range(1, 17, 2)])
            bi = np.concatenate([i * ni + np.arange(ni) for i in range(2, 18, 2)])

            plt.plot(np.arange(18 * ni)[ai] / ni, self.md[ai, 0, j], 'C0:')
            plt.plot(np.arange(18 * ni)[bi] / ni, self.md[bi, 1, j], 'C1:')
            if values:
                plt.plot(np.arange(18 * ni)[ai] / ni, self._v[ai, 0, j], 'C2-')
                plt.plot(np.arange(18 * ni)[bi] / ni, self._v[bi, 1, j], 'C3-')
                plt.plot(np.arange(18 * ni)[ai] / ni, np.r_[self.w_k2m[:ni, 0, j], self.w_k2m[ai[:-ni], 0, j]], 'C2--')
                plt.plot(np.arange(18 * ni)[bi] / ni, np.r_[self.w_k2m[:ni, 1, j], self.w_k2m[bi[:-ni], 1, j]], 'C3--')
            if responses:
                plt.plot(np.arange(18 * ni)[ai] / ni, self.r[ai, j], 'C0-')
                plt.plot(np.arange(18 * ni)[bi] / ni, self.r[bi, j], 'C1-')
                plt.plot(np.arange(18 * ni)[ai] / ni, np.r_[self.w_diff[:ni, 0, j], self.w_diff[ai[:-ni], 0, j]], 'C0--')
                plt.plot(np.arange(18 * ni)[bi] / ni, np.r_[self.w_diff[:ni, 1, j], self.w_diff[bi[:-ni], 1, j]], 'C1--')
            plt.title(name)
            plt.xlim([0.5, 17.5])
            plt.ylim(ylim)
            plt.xticks(np.arange(self.nb_trials+1)+.5, [
                "", "1", "", "2", "", "3", "", "4", "", "5", "", "6", "", "7", "", "8", "", "9"])
        plt.tight_layout()
        plt.show()


class NoValueModel(MBModel):
    def update_values(self, cs, v, mb, t=None):
        if t is None:
            v = v[self._t]
        v_temp = cs[..., np.newaxis] * mb.dot(self.w_m2v) + (1 - cs[..., np.newaxis]) * v
        return np.maximum(v_temp, .8 * v_temp)


class AutoDegrade(MBModel):
    def __init__(self, *args, degrade_factor=.9, **kwargs):
        super(AutoDegrade, self).__init__(*args, **kwargs)
        self.c_forgetting = degrade_factor

    def update_weights(self, cs, v, w_k2m, t=None):
        if t is not None:
            v = v[self._t]
            w_k2m = w_k2m[self._t]
        w_post = w_k2m + self.eta * cs[..., np.newaxis] * (v.dot(self.w_d2k) - w_k2m + self.k2m_init)
        w_post = self.c_forgetting * w_post + (1 - self.c_forgetting) * self.k2m_init
        return np.maximum(w_post, 0)


class UnidimensionalModel(MBModel):
    def __init__(self, normalise=False, *args, **kwargs):
        super(UnidimensionalModel, self).__init__(*args, **kwargs)
        self._mask = np.ones((2, 1), dtype=self.w_k2m.dtype)
        if normalise:
            self.csa = (self.csa - .5)
            self.csb = (self.csb - .5)

    def update_values(self, cs, v, mb):
        v_temp = v + self.eta * self._mask * (mb.dot(self.w_m2v) - v + self.v_init)
        return leaky_relu(v_temp, .8)

    def integrate(self, v, cs):
        return cs.dot(v)


def leaky_relu(v, alpha=.8):
    return np.maximum(v, alpha * v)


if __name__ == '__main__':
    # model = MBModel()
    # model = NoValueModel()
    # model = AutoDegrade(degrade_factor=.8)
    # model = UnidimensionalModel(normalise=False, vm_init=0., vd_init=0.)
    model = UnidimensionalModel(normalise=True, iterations=100, vm_init=1., vd_init=1.)
    model()
    # model.plot("trial")
    model.plot("trial", responses=False, values=True)


