#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Reads and visualises all the arena-paths of the freely-moving flies created before.

Examples:
In order to visualise the data for 100 flies in each condition, run
    $ python3 run_arena_paths.py --dir ../data/arena
or
    $ python3 run_arena_paths.py -d ../data/arena

In order to visualise the respective data using the prediction-error learning rule, run
    $ python3 run_arena_paths.py -d ../data/arena -rw
"""

from incentive.tools import read_arg

import re
import os

# the directory of the file
__dir__ = os.path.dirname(os.path.abspath(__file__))
# the directory of the data
__data_dir__ = os.path.realpath(os.path.join(__dir__, "..", "data", "arena"))


if __name__ == '__main__':
    from incentive.plot import plot_arena_paths

    import matplotlib.pyplot as plt
    import numpy as np

    rw = read_arg(["-rw", "--rescorla-wagner"])
    directory = read_arg(["-d", "--dir"], vtype=str, default=__data_dir__)

    file_names = os.listdir(__data_dir__)

    cases = [
        ["s", "p", "a"],
        ["s", "p", "b"],
        ["s", "p", ""],
        ["s", "r", "a"],
        ["s", "r", "b"],
        ["s", "r", ""],
        ["r", "p", "a"],
        ["r", "p", "b"],
        ["r", "p", ""],
        ["r", "r", "a"],
        ["r", "r", "b"],
        ["r", "r", ""],
        ["m", "p", "a"],
        ["m", "p", "b"],
        ["m", "p", ""],
        ["m", "r", "a"],
        ["m", "r", "b"],
        ["m", "r", ""],
        ["srm", "p", "a"],
        ["srm", "p", "b"],
        ["srm", "p", ""],
        ["srm", "r", "a"],
        ["srm", "r", "b"],
        ["srm", "r", ""],
    ]
    d_names = ["susceptible", "reciprocal", "long-term memory", "reinforcement",
               "paired odour", "phase", "angle"]
    d_raw = [[], [], [], [], [], [], []]

    plt.figure("%sarena-paths" % ("rw-" if rw else ""), figsize=(5, 4))
    for fname in file_names:
        if rw:
            pattern = r'rw-arena-([\w]+)-(s{0,1})(r{0,1})(m{0,1})(a{0,1})(b{0,1})'
        else:
            pattern = r'arena-([\w]+)-(s{0,1})(r{0,1})(m{0,1})(a{0,1})(b{0,1})'
        details = re.findall(pattern, fname)
        if len(details) < 1:
            continue
        punishment = "p" if 'quinine' in details[0] else "r"
        neurons = (
            ("s" if 's' in details[0] else "") +
            ("r" if "r" in details[0] else "") +
            ("m" if "m" in details[0] else "")
        )
        odour = (
            ("a" if "a" in details[0] else "") +
            ("b" if "b" in details[0] else "")
        )
        case = [neurons, punishment, odour]

        name = fname[:-4]

        if case in cases:
            data = np.load(os.path.join(__data_dir__, fname))["data"]

            nb_flies, nb_time_steps = data.shape

            ax = plt.subplot(4, 6, cases.index(case) + 1, polar=True)
            plot_arena_paths(data, name=name, save=False, show=False, ax=ax)
    plt.tight_layout()
    plt.show()
