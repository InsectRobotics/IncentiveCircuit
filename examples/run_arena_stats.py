#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Reads and visualises all the arena-stats of the freely-moving flies created before.

Examples:
In order to visualise the von Mises KDE for 100 flies in each condition, run
    $ python3 run_arena_stats.py --dir ../data/arena
or
    $ python3 run_arena_stats.py -d ../data/arena

In order to visualise the respective KDEs using the prediction-error learning rule, run
    $ python3 run_arena_stats.py -d ../data/arena -rw
"""

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright 2021, School of Informatics, the University of Edinburgh"
__licence__ = "MIT"
__version__ = "1.1-alpha"
__maintainer__ = "Evripidis Gkanias"
__email__ = "ev.gkanias@ed.ac.uk"
__status__ = "Production"

from incentive.tools import read_arg

import os

# the directory of the file
__dir__ = os.path.dirname(os.path.abspath(__file__))
# the directory of the data
__data_dir__ = os.path.realpath(os.path.join(__dir__, "..", "data", "arena"))


if __name__ == '__main__':
    from incentive.plot import plot_arena_stats
    from incentive.arena import load_arena_stats

    rw = read_arg(["-rw", "--rescorla-wagner"])
    directory = read_arg(["-d", "--dir"], vtype=str, default=__data_dir__)

    file_names = os.listdir(directory)

    df = load_arena_stats(file_names, prediction_error=rw)

    plot_arena_stats(df, "%sarena-stats" % ("rw-" if rw else ""))
