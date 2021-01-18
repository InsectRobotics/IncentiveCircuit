import numpy as np
import matplotlib.pyplot as plt


def relu(x, a=0.01):
    return np.maximum(x, a * x)


class Model(object):
    def __init__(self, kc=None, us=None, nb_timesteps=None, apl=False):
        if kc is not None:
            self.kc = kc
            nb_timesteps = self.kc.shape[0]
        if us is not None:
            self.us = us
        if nb_timesteps is None or nb_timesteps < 1:
            nb_timesteps = 1
        if kc is None:
            self.kc = np.zeros((nb_timesteps, 2), dtype=float)
        if us is None:
            self.us = np.zeros(nb_timesteps, dtype=float)
        self.m = np.zeros_like(kc)
        self.d = np.zeros_like(kc)
        self.nb_timesteps = nb_timesteps

        self.w_u2d = np.array([1, 0])
        self.w_k2m = np.full((nb_timesteps + 1, kc.shape[1], len(mbons)), 0.5)
        self.w_m2d = np.array([[0, 1],
                               [0, 0]])
        self.w_d2km = -eta * np.array([[[1, 0], [0, 1]],
                                       [[1, 0], [0, 1]]])

        self.b_k = np.array([0, 0], dtype=float)
        self.b_d = np.array([0, 0], dtype=float)
        self.b_m = np.array([0, 0], dtype=float)
        self.b_w = 0.01
        self.has_apl = apl

        self.__t = 0

    def fprop(self):
        self.kc[self.__t] = relu(self.kc[self.__t])
        if self.has_apl:
            self.kc[self.__t] = relu(self.kc[self.__t] - self.kc[self.__t].mean())
        self.m[self.__t] = relu((self.kc[self.__t] - self.b_k) @ self.w_k2m[self.__t])
        self.d[self.__t] = relu(self.us[self.__t] * self.w_u2d + (self.m[self.__t] - self.b_m) @ self.w_m2d)

        self.w_k2m[self.__t + 1] = np.clip(
            self.w_k2m[self.__t] + ((self.kc[self.__t] - self.b_k) * ((self.d[self.__t] - self.b_d) @ self.w_d2km).T).T
            + self.b_w, 0., 2.)
        self.__t += 1


if __name__ == '__main__':
    trials_p = 1  # number of pre-training trials
    trials_a = 5  # number of acquisition trials
    trials_t = 1  # number of rest trials
    trials_r = 2  # number of reversal trials
    trials_e = 6  # number of no shock trials

    eta = .3
    apl = False

    # pre-training
    kc_p, us_p, fl_p = [[1, 0], [0, 1]] * trials_p, [0, 0] * trials_p, ["p", "p"] * trials_p
    # acquisition
    kc_a, us_a, fl_a = [[1, 0], [0, 1]] * trials_a, [0, 1] * trials_a, ["a", "a"] * trials_a
    # rest
    kc_t, us_t, fl_t = [[1, 0]] * trials_t, [0] * trials_t, ["t"] * trials_t
    # reversal
    kc_r, us_r, fl_r = [[0, 1], [1, 0]] * trials_r, [0, 1] * trials_r, ["r", "r"] * trials_r
    # no shock
    kc_e, us_e, fl_e = [[1, 0], [0, 1]] * trials_e, [0, 0] * trials_e, ["e", "e"] * trials_e

    paradigms = {
        "MCH-reversal": {
            "CS": np.array(kc_p + kc_a + kc_t + kc_r, dtype=float),
            "US": np.array(us_p + us_a + us_t + us_r, dtype=float),
            "flag": np.array(fl_p + fl_a + fl_t + fl_r)
        },
        "MCH-no-shock": {
            "CS": np.array(kc_p + kc_a + kc_e, dtype=float),
            "US": np.array(us_p + us_a + us_e, dtype=float),
            "flag": np.array(fl_p + fl_a + fl_e)
        },
        "OCT-reversal": {
            "CS": 1 - np.array(kc_p + kc_a + kc_t + kc_r, dtype=float),
            "US": np.array(us_p + us_a + us_t + us_r, dtype=float),
            "flag": np.array(fl_p + fl_a + fl_t + fl_r)
        },
        "OCT-no-shock": {
            "CS": 1 - np.array(kc_p + kc_a + kc_e, dtype=float),
            "US": np.array(us_p + us_a + us_e, dtype=float),
            "flag": np.array(fl_p + fl_a + fl_e)
        }
    }

    mbons = ["MBON-γ2α'1", "MBON-γ5β'2a"]
    dans = ["PPL1-γ2α'1", "PAM-β'2a"]

    for paradigm in paradigms:
        model = Model(kc=paradigms[paradigm]["CS"], us=paradigms[paradigm]["US"], apl=apl)

        for t in range(model.nb_timesteps):
            model.fprop()

        t = np.arange(model.nb_timesteps)
        plt.figure(paradigm, figsize=(7, 5))
        plt.subplot(221)
        plt.plot(t[int("OCT" in paradigm)::2], model.m[int("OCT" in paradigm)::2, 0])
        plt.plot(t[int("MCH" in paradigm)::2], model.m[int("MCH" in paradigm)::2, 0])
        plt.plot(t, model.w_k2m[1:, 0, 0], "C2--")
        plt.plot(t, model.w_k2m[1:, 1, 0], "C3--")
        plt.ylim([-0.1, 1.5])
        plt.ylabel(mbons[0])
        plt.subplot(222)
        plt.plot(t[int("OCT" in paradigm)::2], model.m[int("OCT" in paradigm)::2, 1])
        plt.plot(t[int("MCH" in paradigm)::2], model.m[int("MCH" in paradigm)::2, 1])
        plt.plot(t, model.w_k2m[1:, 0, 1], "C2--")
        plt.plot(t, model.w_k2m[1:, 1, 1], "C3--")
        plt.ylim([-0.1, 1.5])
        plt.ylabel(mbons[1])
        plt.subplot(223)
        plt.plot(t, (model.us[:, None] * model.w_u2d[None, :])[:, 0], "y--")
        plt.plot(t[int("OCT" in paradigm)::2], model.d[int("OCT" in paradigm)::2, 0])
        plt.plot(t[int("MCH" in paradigm)::2], model.d[int("MCH" in paradigm)::2, 0])
        plt.ylim([-0.1, 1.5])
        plt.ylabel(dans[0])
        plt.subplot(224)
        plt.plot(t, (model.us[:, None] * model.w_u2d[None, :])[:, 1], "y--")
        plt.plot(t[int("OCT" in paradigm)::2], model.d[int("OCT" in paradigm)::2, 1])
        plt.plot(t[int("MCH" in paradigm)::2], model.d[int("MCH" in paradigm)::2, 1])
        plt.ylim([-0.1, 1.5])
        plt.ylabel(dans[1])

        plt.tight_layout()
        plt.show()

        # break
