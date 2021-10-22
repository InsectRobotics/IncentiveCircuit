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
    from incentive.arena import load_arena_stats

    odours_visited = "A and B"

    nb_kc = 10
    nb_kc_a = 7
    nb_kc_b = 6
    nb_active_kcs = 5
    nb_repeats = 10
    rpe = read_arg(["-rpe", "--reward-prediction-error"])
    directory = read_arg(["-d", "--dir"], vtype=str, default=__data_dir__)

    file_names = os.listdir(directory)

    df = load_arena_stats(file_names, nb_active_kcs=nb_active_kcs, prediction_error=rpe)

    nb_flies, _ = df[np.all([df["phase"] == "pre", df["paired odour"] == "A", df["reinforcement"] == "reward",
                             df["susceptible"], df["restrained"], df["long-term memory"],
                             df["repeat"] == 1], axis=0)].shape

    codes = ["srm"]
    import matplotlib.pyplot as plt

    plt.figure(("rpe-" if rpe else "") + "arena-pi-weights-" + odours_visited.lower().replace(" ", "-"), figsize=(8, 8))

    data = {
        "A - punishment": np.zeros((nb_repeats, 6 * 3, nb_kc, nb_flies), dtype=float),
        "B - punishment": np.zeros((nb_repeats, 6 * 3, nb_kc, nb_flies), dtype=float),
        "AB - punishment": np.zeros((nb_repeats, 6 * 3, nb_kc, nb_flies), dtype=float),
        "A - reward": np.zeros((nb_repeats, 6 * 3, nb_kc, nb_flies), dtype=float),
        "B - reward": np.zeros((nb_repeats, 6 * 3, nb_kc, nb_flies), dtype=float),
        "AB - reward": np.zeros((nb_repeats, 6 * 3, nb_kc, nb_flies), dtype=float)
    }

    d_time = {
        "A - punishment": np.zeros((nb_repeats, 6, nb_flies), dtype=float),
        "B - punishment": np.zeros((nb_repeats, 6, nb_flies), dtype=float),
        "AB - punishment": np.zeros((nb_repeats, 6, nb_flies), dtype=float),
        "A - reward": np.zeros((nb_repeats, 6, nb_flies), dtype=float),
        "B - reward": np.zeros((nb_repeats, 6, nb_flies), dtype=float),
        "AB - reward": np.zeros((nb_repeats, 6, nb_flies), dtype=float)
    }

    x = np.arange(3 * nb_repeats) / 3 + 1

    for i, title in enumerate(data):
        details = re.match(r"([\w]{1,2}) - ([\w]+)", title)
        odour = details.group(1)
        reinforcement = details.group(2)
        col = i + 1

        a_pre, a_post, a_learn = [], [], []
        b_pre, b_post, b_learn = [], [], []
        for r in range(nb_repeats):
            for p, phase in enumerate(["pre", "learn", "post"]):
                for k in range(nb_kc):
                    for n, neuron in enumerate(["s+", "s-", "r+", "r-", "m+", "m-"]):
                        prop = "k%d2%s" % (k, neuron)
                        data[title][r, n * 3 + p, k, :] = df[np.all([df["phase"] == phase,
                                                                     df["reinforcement"] == reinforcement,
                                                                     df["paired odour"] == odour,
                                                                     df["susceptible"] == ("s" in codes[0]),
                                                                     df["restrained"] == ("r" in codes[0]),
                                                                     df["long-term memory"] == ("m" in codes[0]),
                                                                     df["repeat"] == r + 1], axis=0)][prop]

                for t, time in enumerate(["time_A", "time_B"]):
                    d_time[title][r, t * 3 + p, :] = df[np.all([df["phase"] == phase,
                                                                df["reinforcement"] == reinforcement,
                                                                df["paired odour"] == odour,
                                                                df["susceptible"] == ("s" in codes[0]),
                                                                df["restrained"] == ("r" in codes[0]),
                                                                df["long-term memory"] == ("m" in codes[0]),
                                                                df["repeat"] == r + 1], axis=0)][time]
        d_time[title][:, [0, 3]] /= 20
        d_time[title][:, [1, 4]] /= 30
        d_time[title][:, [2, 5]] /= 50

        d_time[title] = np.cumsum(d_time[title], axis=0)
        d_time[title] /= (np.array(10) + 1)  # normalise with the repeat time

        a_pr = d_time[title][:, 0]  # - data[title][:, 1]
        a_tr = d_time[title][:, 1]  # - data[title][:, 1]
        a_ps = d_time[title][:, 2]  # - data[title][:, 1]
        b_pr = d_time[title][:, 3]  # - data[title][:, 4]
        b_tr = d_time[title][:, 4]  # - data[title][:, 4]
        b_ps = d_time[title][:, 5]  # - data[title][:, 4]

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

        lw = .2
        alpha = .5
        nb_samples = 3

        color_r = 'red' if reinforcement == "punishment" else "green"

        t_in = np.argmax(c > 0, axis=0) // 3
        s, p = [], []
        for t in range(nb_repeats):
            ss = np.arange(c.shape[1])[t_in == t]
            if len(ss) > 0:
                s.append(ss[0])
                p.append(len(ss))
        s = np.array(s)[np.argsort(p)[::-1]][:nb_samples]

        for n, neuron in enumerate(["s_{at}", "s_{av}", "r_{at}", "r_{av}", "m_{at}", "m_{av}"]):
            k_pr = np.transpose(data[title][:, n*3+0, :, i], axes=(1, 2, 0))
            k_tr = np.transpose(data[title][:, n*3+1, :, i], axes=(1, 2, 0))
            k_po = np.transpose(data[title][:, n*3+2, :, i], axes=(1, 2, 0))

            ck = np.zeros((k_pr.shape[0] * 3, k_pr.shape[1], k_pr.shape[2]), dtype=k_pr.dtype)
            ck[0::3, :] = k_pr
            ck[1::3, :] = k_tr
            ck[2::3, :] = k_po
            ck[np.isnan(ck)] = 0.

            nb_a_only = np.min([nb_kc_a, nb_kc - nb_kc_b])
            nb_b_only = np.min([nb_kc - nb_kc_a, nb_kc_b])

            ca = np.nanmedian(ck[:, :nb_a_only], axis=1)
            cb = np.nanmedian(ck[:, -nb_b_only:], axis=1)
            co = np.nanmedian(ck[:, nb_a_only:-nb_b_only], axis=1)

            ca_q50 = np.nanmedian(ca, axis=1)
            cb_q50 = np.nanmedian(cb, axis=1)
            co_q50 = np.nanmedian(co, axis=1)

            ax = plt.subplot(6, 6, 6 * n + col)

            if "A" in odour:
                plt.plot(x, -co[:, s], '-', color=ODOUR_A_CMAP(0.99), lw=lw, alpha=alpha)
            if "B" in odour:
                plt.plot(x, co[:, s], '-', color=ODOUR_B_CMAP(0.99), lw=lw, alpha=alpha)
            plt.plot(x, ca[:, s], '-', color=ODOUR_A_CMAP(0.99), lw=lw, alpha=alpha)
            plt.plot(x, -cb[:, s], '-', color=ODOUR_B_CMAP(0.99), lw=lw, alpha=alpha)

            if "A" in odour:
                plt.plot(x, -co_q50, ':', color=ODOUR_A_CMAP(0.99), lw=2)
            if "B" in odour:
                plt.plot(x, co_q50, ':', color=ODOUR_B_CMAP(0.99), lw=2)
            plt.plot(x, ca_q50, '-', color=ODOUR_A_CMAP(0.99), lw=2)
            plt.plot(x, -cb_q50, '-', color=ODOUR_B_CMAP(0.99), lw=2)

            if "A" in odour:
                plt.plot(x[1::3], ca_q50[1::3], '.', color=color_r)
            if "B" in odour:
                plt.plot(x[1::3], -cb_q50[1::3], '.', color=color_r)

            plt.ylim(-1.1, 1.1)
            plt.xlim(0.5, 11.4)
            plt.xticks([1, 5, 10])
            plt.yticks([-1, 0, 1])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)

    plt.tight_layout()
    plt.show()
