import numpy as np
import matplotlib.pyplot as plt


def relu(x, a=0.01):
    return np.maximum(x, a * x)


if __name__ == '__main__':
    trials_p = 1  # number of pre-training trials
    trials_a = 5  # number of acquisition trials
    trials_t = 1  # number of rest trials
    trials_r = 2  # number of reversal trials
    trials_e = 6  # number of no-shock trials

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
        # "OCT-reversal": {
        #     "CS": 1 - np.array(kc_p + kc_a + kc_t + kc_r, dtype=float),
        #     "US": np.array(us_p + us_a + us_t + us_r, dtype=float),
        #     "flag": np.array(fl_p + fl_a + fl_t + fl_r)
        # },
        # "OCT-no-shock": {
        #     "CS": 1 - np.array(kc_p + kc_a + kc_e, dtype=float),
        #     "US": np.array(us_p + us_a + us_e, dtype=float),
        #     "flag": np.array(fl_p + fl_a + fl_e)
        # }
    }

    mbons = ["MBON-γ1ped", "MBON-γ2α'1", "MBON-γ5β'2a", "MBON-?"]
    dans = ["PPL1-γ1ped", "PPL1-γ2α'1", "PAM-β'2a", "PAM-?"]

    for paradigm in paradigms:
        kc = paradigms[paradigm]["CS"]
        us = paradigms[paradigm]["US"]
        m = np.zeros((kc.shape[0], 4), dtype=float)
        d = np.zeros((kc.shape[0], 4), dtype=float)
        w_u2d = np.array([1, 1, 0, 0])
        w_k2m = np.full((kc.shape[0] + 1, kc.shape[1], len(mbons)), 0.5)
        w_m2m = np.array([[0, 0, -1, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, -1, 0, 0]])
        w_m2d = np.array([[-1, 0, 0, 0],
                          [0, 0, 1, 0],
                          [0, 1, 0, 0],
                          [0, 0, 0, -1]])
        w_d2km = -eta * np.array([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                                  [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]])

        b_k = np.array([0, 0], dtype=float)
        b_d = np.array([0, 0, 0, 0], dtype=float)
        b_m = np.array([0, 0, 0, .5], dtype=float)
        b_w = 0.01

        for t in range(0):
            kc[0] = relu(kc[0])
            if apl:
                kc[0] = relu(kc[0] - kc[0].mean())
            m[0] = (kc[0] - b_k) @ w_k2m[0]
            m[0] = relu(m[0] + (m[0] - b_m) @ w_m2m)
            d[0] = relu(us[0] * w_u2d + (m[0] - b_m) @ w_m2d)
            w_k2m[0] = np.clip(w_k2m[0] + (kc - b_k)[0, :, None] * ((d[0] - b_d) @ w_d2km) + b_w, 0., 1.5)

        for t in range(len(kc)):
            kc[t] = relu(kc[t])
            if apl:
                kc[t] = relu(kc[t] - kc[t].mean())
            m[t] = (kc[t] - b_k) @ w_k2m[t]
            m[t] = relu(m[t] + (m[t] - b_m) @ w_m2m)
            d[t] = relu(us[t] * w_u2d + (m[t] - b_m) @ w_m2d)
            w_k2m[t + 1] = np.clip(w_k2m[t] + (kc - b_k)[t, :, None] * ((d[t] - b_d) @ w_d2km) + b_w, 0., 1.5)

        t = np.arange(len(kc))
        plt.figure(paradigm, figsize=(7, 5))
        plt.subplot(241)
        plt.plot(t[int("OCT" in paradigm)::2], m[int("OCT" in paradigm)::2, 0])
        plt.plot(t[int("MCH" in paradigm)::2], m[int("MCH" in paradigm)::2, 0])
        plt.plot(t, w_k2m[1:, 0, 0], "C2--")
        plt.plot(t, w_k2m[1:, 1, 0], "C3--")
        plt.ylim([-0.1, 1.5])
        plt.ylabel(mbons[0])
        plt.subplot(242)
        plt.plot(t[int("OCT" in paradigm)::2], m[int("OCT" in paradigm)::2, 1])
        plt.plot(t[int("MCH" in paradigm)::2], m[int("MCH" in paradigm)::2, 1])
        plt.plot(t, w_k2m[1:, 0, 1], "C2--")
        plt.plot(t, w_k2m[1:, 1, 1], "C3--")
        plt.ylim([-0.1, 1.5])
        plt.ylabel(mbons[1])
        plt.subplot(243)
        plt.plot(t[int("OCT" in paradigm)::2], m[int("OCT" in paradigm)::2, 2])
        plt.plot(t[int("MCH" in paradigm)::2], m[int("MCH" in paradigm)::2, 2])
        plt.plot(t, w_k2m[1:, 0, 2], "C2--")
        plt.plot(t, w_k2m[1:, 1, 2], "C3--")
        plt.ylim([-0.1, 1.5])
        plt.ylabel(mbons[2])
        plt.subplot(244)
        plt.plot(t[int("OCT" in paradigm)::2], m[int("OCT" in paradigm)::2, 3])
        plt.plot(t[int("MCH" in paradigm)::2], m[int("MCH" in paradigm)::2, 3])
        plt.plot(t, w_k2m[1:, 0, 3], "C2--")
        plt.plot(t, w_k2m[1:, 1, 3], "C3--")
        plt.ylim([-0.1, 1.5])
        plt.ylabel(mbons[3])
        plt.subplot(245)
        plt.plot(t, (us[:, None] * w_u2d[None, :])[:, 0], "y--")
        plt.plot(t[int("OCT" in paradigm)::2], d[int("OCT" in paradigm)::2, 0])
        plt.plot(t[int("MCH" in paradigm)::2], d[int("MCH" in paradigm)::2, 0])
        plt.ylim([-0.1, 1.5])
        plt.ylabel(dans[0])
        plt.subplot(246)
        plt.plot(t, (us[:, None] * w_u2d[None, :])[:, 1], "y--")
        plt.plot(t[int("OCT" in paradigm)::2], d[int("OCT" in paradigm)::2, 1])
        plt.plot(t[int("MCH" in paradigm)::2], d[int("MCH" in paradigm)::2, 1])
        plt.ylim([-0.1, 1.5])
        plt.ylabel(dans[1])
        plt.subplot(247)
        plt.plot(t, (us[:, None] * w_u2d[None, :])[:, 2], "y--")
        plt.plot(t[int("OCT" in paradigm)::2], d[int("OCT" in paradigm)::2, 2])
        plt.plot(t[int("MCH" in paradigm)::2], d[int("MCH" in paradigm)::2, 2])
        plt.ylim([-0.1, 1.5])
        plt.ylabel(dans[2])
        plt.subplot(248)
        plt.plot(t, (us[:, None] * w_u2d[None, :])[:, 3], "y--")
        plt.plot(t[int("OCT" in paradigm)::2], d[int("OCT" in paradigm)::2, 3])
        plt.plot(t[int("MCH" in paradigm)::2], d[int("MCH" in paradigm)::2, 3])
        plt.ylim([-0.1, 1.5])
        plt.ylabel(dans[3])

        plt.tight_layout()
        plt.show()

        # break
