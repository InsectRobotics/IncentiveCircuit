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

    rw = read_arg(["-rw", "--rescorla-wagner"])
    directory = read_arg(["-d", "--dir"], vtype=str, default=__data_dir__)

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

    codes = ["srm"]
    # codes = ["srm", "s", "r", "m"]
    for code in codes:
        plot_arena_box(df[np.all([df["susceptible"] == ("s" in code),
                                  df["restrained"] == ("r" in code),
                                  df["long-term memory"] == ("m" in code)], axis=0)],
                       max_repeat=10,
                       name="%sarena-box-%s" % ("rw-" if rw else "", code),
                       show=code == codes[-1])

    # repeats = np.unique(df["repeat"])
    # repeats = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # for repeat in repeats:
    #     plot_arena_box(df[df["repeat"] == repeat], "%sarena-box-%02d" % ("rw-" if rw else "", repeat),
    #                    show=repeat == repeats[-1])
