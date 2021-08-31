# #!/usr/bin/env python
# -*- coding: utf-8 -*-

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

    rw = read_arg(["-rw", "--rescorla-wagner"])
    directory = read_arg(["-d", "--dir"], vtype=str, default=__data_dir__)
    repeats = np.arange(read_arg(["-R", "--repeat"], vtype=int, default=10))

    file_names = os.listdir(directory)

    df = load_arena_stats(file_names, prediction_error=rw)
    df["dist_A"][df["phase"] == "learn"] = df["dist_A"][df["phase"] == "learn"]

    df["PI"] = (df["dist_A"] - df["dist_B"]) / (df["dist_A"] + df["dist_B"])
    df["avoid A"] = df["dist_A"] - 1
    df["avoid B"] = df["dist_B"] - 1
    df["avoid A/B"] = np.max([df["dist_A"], df["dist_B"]], axis=0) / 0.6 - 1
    df["attract A"] = df["dist_A"] - 1
    df["attract B"] = df["dist_B"] - 1
    df["attract A/B"] = np.min([df["dist_A"], df["dist_B"]], axis=0) / 0.6 - 1

    # repeats = np.unique(df["repeat"])
    # repeats = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for repeat in repeats:
        plot_arena_box(df[df["repeat"] == repeat], "%sarena-box-%02d" % ("rw-" if rw else "", repeat),
                       show=repeat == repeats[-1])
