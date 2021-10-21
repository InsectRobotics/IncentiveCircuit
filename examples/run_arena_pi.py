# #!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright 2021, School of Informatics, the University of Edinburgh"
__licence__ = "MIT"
__version__ = "1.1-alpha"
__maintainer__ = "Evripidis Gkanias"
__email__ = "ev.gkanias@ed.ac.uk"
__status__ = "Production"

from incentive.tools import read_arg

import numpy as np
import os

# the directory of the file
__dir__ = os.path.dirname(os.path.abspath(__file__))
# the directory of the data
__data_dir__ = os.path.realpath(os.path.join(__dir__, "..", "src", "incentive", "data", "arena"))


if __name__ == '__main__':
    from incentive.plot import plot_arena_box
    from incentive.arena import load_arena_stats

    nb_active_kcs = 5
    rpe = read_arg(["-rpe", "--reward-prediction-error"])
    directory = read_arg(["-d", "--dir"], vtype=str, default=__data_dir__)

    file_names = os.listdir(directory)

    df = load_arena_stats(file_names, nb_active_kcs=nb_active_kcs, prediction_error=rpe)

    # df["PI"] = -(df["time_A"] - df["time_B"]) / (df["time_A"] + df["time_B"])
    # df["PI"] = (df["dist_A"] - df["dist_B"]) / (df["dist_A"] + df["dist_B"])

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
    # codes = ["srm", "s", "r", "m"]
    for code in codes:
        plot_arena_box(df[np.all([df["susceptible"] == ("s" in code),
                                  df["restrained"] == ("r" in code),
                                  df["long-term memory"] == ("m" in code)], axis=0)],
                       max_repeat=10,
                       name="%sarena-box-k%d-%s" % ("rpe-" if rpe else "", nb_active_kcs, code),
                       show=code == codes[-1])
