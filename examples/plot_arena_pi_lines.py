# #!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright 2021, School of Informatics, the University of Edinburgh"
__licence__ = "MIT"
__version__ = "1.1-alpha"
__maintainer__ = "Evripidis Gkanias"
__email__ = "ev.gkanias@ed.ac.uk"
__status__ = "Production"

import re

from incentive.tools import read_arg
from incentive.plot import ODOUR_A_CMAP, ODOUR_B_CMAP

import numpy as np
import os

# the directory of the file
__dir__ = os.path.dirname(os.path.abspath(__file__))
# the directory of the data
__data_dir__ = os.path.realpath(os.path.join(__dir__, "..", "src", "incentive", "data", "arena"))


if __name__ == '__main__':
    from incentive.plot import plot_arena_box
    from incentive.arena import load_arena_stats

    odours_visited = "A and B"

    nb_active_kcs = 5
    # rpe = read_arg(["-rpe", "--reward-prediction-error"])
    rpe = True
    directory = read_arg(["-d", "--dir"], vtype=str, default=__data_dir__)

    file_names = os.listdir(directory)

    df = load_arena_stats(file_names, nb_active_kcs=nb_active_kcs, prediction_error=rpe)

    nb_flies, _ = df[np.all([df["phase"] == "pre", df["paired odour"] == "A", df["reinforcement"] == "reward",
                             df["susceptible"], df["restrained"], df["long-term memory"],
                             df["repeat"] == 1], axis=0)].shape

    d_pi = np.full_like(df["time_A"], np.nan)

    i_learn = np.zeros(df["phase"].shape[0] // 3, dtype=bool)
    t_learn = np.array(df[df["phase"] == "learn"]["time_A"])
    i_learn[t_learn > 0] = True
    t_pre = np.array(df[df["phase"] == "pre"]["time_A"])[t_learn > 0]
    t_post = np.array(df[df["phase"] == "post"]["time_A"])[t_learn > 0]
    i_pi = np.array(df["phase"] == "pre")
    i_pi[i_pi] = i_learn
    d_pi[i_pi] = (t_post - t_pre) / (t_post + t_pre)

    i_learn = np.zeros(df["phase"].shape[0] // 3, dtype=bool)
    t_learn = np.array(df[df["phase"] == "learn"]["time_B"])
    i_learn[t_learn > 0] = True
    t_pre = np.array(df[df["phase"] == "pre"]["time_B"])[t_learn > 0]
    t_post = np.array(df[df["phase"] == "post"]["time_B"])[t_learn > 0]
    i_pi = np.array(df["phase"] == "learn")
    i_pi[i_pi] = i_learn
    d_pi[i_pi] = (t_post - t_pre) / (t_post + t_pre)

    d_pi[np.array(df["phase"] == "post")] = (d_pi[df["phase"] == "learn"] - d_pi[df["phase"] == "pre"]) / (
        (d_pi[df["phase"] == "learn"] + d_pi[df["phase"] == "pre"]))

    df["PI"] = d_pi

    codes = ["srm"]
    import matplotlib.pyplot as plt

    plt.figure(("rpe-" if rpe else "") + "arena-pi-lines-" + odours_visited.lower().replace(" ", "-"), figsize=(8, 3))

    data = {
        "A - punishment": np.zeros((10, 6, nb_flies), dtype=float),
        "B - punishment": np.zeros((10, 6, nb_flies), dtype=float),
        "AB - punishment": np.zeros((10, 6, nb_flies), dtype=float),
        "A - reward": np.zeros((10, 6, nb_flies), dtype=float),
        "B - reward": np.zeros((10, 6, nb_flies), dtype=float),
        "AB - reward": np.zeros((10, 6, nb_flies), dtype=float)
    }

    for i, title in enumerate(data):
        details = re.match(r"([\w]{1,2}) - ([\w]+)", title)
        odour = details.group(1)
        reinforcement = details.group(2)
        col = i + 1

        a_pre, a_post, a_learn = [], [], []
        b_pre, b_post, b_learn = [], [], []
        for r in range(10):
            for t, time in enumerate(["time_A", "time_B"]):
                for p, phase in enumerate(["pre", "learn", "post"]):
                    data[title][r, t * 3 + p, :] = df[np.all([df["phase"] == phase,
                                                              df["reinforcement"] == reinforcement,
                                                              df["paired odour"] == odour,
                                                              df["susceptible"] == ("s" in codes[0]),
                                                              df["restrained"] == ("r" in codes[0]),
                                                              df["long-term memory"] == ("m" in codes[0]),
                                                              df["repeat"] == r + 1], axis=0)][time]
        data[title][:, [0, 3]] /= 20
        data[title][:, [1, 4]] /= 30
        data[title][:, [2, 5]] /= 50

        data[title] = np.cumsum(data[title], axis=0)
        data[title] /= (np.array(10) + 1)  # normalise with the repeat time

        a_pr = data[title][:, 0]  # - data[title][:, 1]
        a_tr = data[title][:, 1]  # - data[title][:, 1]
        # # keep only the first time that gets in the reinforced area
        # da_c = da_c[np.argmax(da_c > 0, axis=0), np.arange(da_c.shape[1])]
        a_ps = data[title][:, 2]  # - data[title][:, 1]
        b_pr = data[title][:, 3]  # - data[title][:, 4]
        b_tr = data[title][:, 4]  # - data[title][:, 4]
        # # keep only the first time that gets in the reinforced area
        # db_c = db_c[np.argmax(db_c > 0, axis=0), np.arange(db_c.shape[1])]
        b_ps = data[title][:, 5]  # - data[title][:, 4]
        # if "AB" in odour:
        #     d_learn = np.array(np.any([data[title][:, 1] > 0, data[title][:, 4] > 0], axis=0), dtype=float)
        #     pi_c = np.max([d_learn * (a_pr - b_pr) / (a_pr + b_pr), d_learn * (b_pr - a_pr) / (a_pr + b_pr)], axis=0)
        #     pi_i = np.max([d_learn * (a_ps - b_ps) / (a_ps + b_ps), d_learn * (b_ps - a_ps) / (a_ps + b_ps)], axis=0)
        # elif "A" in odour:
        #     d_learn = np.array(data[title][:, 1] > 0, dtype=float)
        #     # time spend with odour A in higher than time spent with odour B
        #     pi_c = d_learn * (a_pr - b_pr) / (a_pr + b_pr)
        #     pi_i = d_learn * (a_ps - b_ps) / (a_ps + b_ps)
        # elif "B" in odour:
        #     d_learn = np.array(data[title][:, 4] > 0, dtype=float)
        #     # time spend with odour B in higher than time spent with odour A
        #     pi_c = d_learn * (b_pr - a_pr) / (a_pr + b_pr)
        #     pi_i = d_learn * (b_ps - a_ps) / (a_ps + b_ps)
        # else:
        #     d_learn = False
        #     pi_c = np.zeros_like(data[title][:, 0])
        #     pi_i = np.zeros_like(data[title][:, 0])
        #
        # f_c = (pi_c + 1) / 2
        # f_i = (pi_i + 1) / 2

        # d_f = (f_i - f_c) / np.sqrt(.5 * (f_i + f_c) * (1 - .5 * (f_i + f_c)))
        #
        # d_q25 = np.nanquantile(d_f, .25, axis=1)
        # d_q50 = np.nanquantile(d_f, .50, axis=1)
        # d_q75 = np.nanquantile(d_f, .75, axis=1)
        ia = np.any(a_tr > 0, axis=0)
        ib = np.any(b_tr > 0, axis=0)

        i = np.zeros_like(ia)
        operation = None
        is_not = False
        iis = {"a": ia, "b": ib}
        for s in odours_visited.lower().split(" "):
            if s in ["not"]:
                is_not = not is_not
            elif s in ["and", "or"]:
                operation = s
            elif s in ["a", "b"]:
                ii = ~iis[s] if is_not else iis[s]
                is_not = False
                if operation is None:
                    i = ii
                else:
                    if operation in ["and"]:
                        i = i & ii
                        operation = None
                    elif operation in ["or"]:
                        i = i | ii
                        operation = None

        print(f"{np.sum(i)} / {i.shape[0]}")

        ca = np.zeros((a_pr[:, i].shape[0] * 3, a_pr[:, i].shape[1]), dtype=a_pr.dtype)
        ca[0::3, :] = a_pr[:, i]
        ca[1::3, :] = a_tr[:, i]
        ca[2::3, :] = a_ps[:, i]
        ca_q25 = np.nanquantile(ca, .25, axis=1)
        ca_q50 = np.nanquantile(ca, .50, axis=1)
        ca_q75 = np.nanquantile(ca, .75, axis=1)

        cb = np.zeros((b_pr[:, i].shape[0] * 3, b_pr[:, i].shape[1]), dtype=b_pr.dtype)
        cb[0::3, :] = b_pr[:, i]
        cb[1::3, :] = b_tr[:, i]
        cb[2::3, :] = b_ps[:, i]
        cb_q25 = np.nanquantile(cb, .25, axis=1)
        cb_q50 = np.nanquantile(cb, .50, axis=1)
        cb_q75 = np.nanquantile(cb, .75, axis=1)

        c = (ca - cb) / (ca + cb + np.finfo(float).eps)
        c[np.isnan(c)] = 0.
        # c_q25 = (ca_q25 - cb_q25) / (ca_q25 + cb_q25)
        # c_q50 = (ca_q50 - cb_q50) / (ca_q50 + cb_q50)
        # c_q75 = (ca_q75 - cb_q75) / (ca_q75 + cb_q75)
        c_q25 = np.nanquantile(c, .25, axis=1)
        c_q50 = np.nanquantile(c, .50, axis=1)
        c_q75 = np.nanquantile(c, .75, axis=1)

        color_r = 'red' if reinforcement == "punishment" else "green"

        lw = .2
        alpha = .2
        # nb_samples = 10
        nb_samples = c.shape[1]
        s = np.random.permutation(np.arange(c.shape[1]))[:nb_samples]
        # y_max = np.maximum(ca.max(), cb.max()) * 1.1
        y_max = np.maximum(np.sqrt(ca.max()), np.sqrt(cb.max())) * 1.1

        x = np.arange(30) / 3 + 1
        ax = plt.subplot(2, 6, col)
        plt.plot(x, np.sqrt(ca[:, s]), '-', color=ODOUR_A_CMAP(0.99), lw=lw, alpha=alpha)
        # plt.fill_between(x, ca_q25, ca_q75, facecolor=ODOUR_A_CMAP(0.99), alpha=.2)
        plt.plot(x, np.sqrt(ca_q50), '-', color=ODOUR_A_CMAP(0.99), lw=2)

        plt.plot(x, -np.sqrt(cb[:, s]), '-', color=ODOUR_B_CMAP(0.99), lw=lw, alpha=alpha)
        # plt.fill_between(x, -cb_q25, -cb_q75, facecolor=ODOUR_B_CMAP(0.99), alpha=.2)
        plt.plot(x, -np.sqrt(cb_q50), '-', color=ODOUR_B_CMAP(0.99), lw=2)
        if "A" in odour:
            plt.plot(x[1::3], np.sqrt(ca_q50[1::3]), '.', color=color_r)
        if "B" in odour:
            plt.plot(x[1::3], -np.sqrt(cb_q50[1::3]), '.', color=color_r)
        plt.ylim(-y_max, y_max)
        plt.xlim(0.5, 11.4)
        plt.xticks([1, 5, 10])
        plt.yticks([-y_max, 0, y_max], ["B", "0", "A"])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        ax = plt.subplot(2, 6, col + 6)
        plt.plot([0, 10], [0, 0], 'k:', lw=.5)
        plt.plot(x, c[:, s], '-', color='black', lw=lw, alpha=alpha)
        # plt.fill_between(x, c_q25, c_q75, facecolor='black', alpha=.2)
        plt.plot(x, c_q50, 'k-', lw=2)
        plt.plot(x[1::3], c_q50[1::3], '.', color=color_r)
        plt.ylim(-1.1, 1.1)
        plt.xlim(0.5, 11.4)
        plt.xticks([1, 5, 10])
        plt.yticks([-1, 0, 1], ["B", "0", "A"])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    plt.tight_layout()
    plt.show()
