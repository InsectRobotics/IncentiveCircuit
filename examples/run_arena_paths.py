##!/usr/bin/env python
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

import os

# the directory of the file
__dir__ = os.path.dirname(os.path.abspath(__file__))
# the directory of the data
__data_dir__ = os.path.realpath(os.path.join(__dir__, "..", "src", "incentive", "data", "arena"))


if __name__ == '__main__':
    from incentive.arena import load_arena_paths
    from incentive.plot import plot_arena_paths

    rw = read_arg(["-rw", "--rescorla-wagner"], default=False)
    directory = read_arg(["-d", "--dir"], vtype=str, default=__data_dir__)
    repeat = read_arg(['-r', '--repeat'], vtype=int, default=5)

    file_names = os.listdir(__data_dir__)
    d_raw, cases, d_names = load_arena_paths(file_names, repeat=repeat, prediction_error=rw)
    plot_arena_paths(d_raw, cases, d_names, figsize=(5, 4),
                     name="%sarena-paths%s" % ("rw-" if rw else "", "-%02d" % repeat if repeat is not None else ""))

