##!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Reads and visualises all the arena-paths of the freely-moving flies created before.

Examples:
In order to visualise the data for 100 flies in each condition, run
    $ python3 plot_arena_paths.py --dir ../data/arena
or
    $ python3 plot_arena_paths.py -d ../data/arena

In order to visualise the respective data using the prediction-error learning rule, run
    $ python3 plot_arena_paths.py -d ../data/arena -rw
"""

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright 2021, School of Informatics, the University of Edinburgh"
__licence__ = "MIT"
__version__ = "1.1-alpha"
__maintainer__ = "Evripidis Gkanias"
__email__ = "ev.gkanias@ed.ac.uk"
__status__ = "Production"

from incentive import arena
from incentive.arena import load_arena_paths
from incentive.plot import plot_arena_paths
from incentive.tools import read_arg

import os


def main(*args):

    nb_active_kcs = 5
    rpe = read_arg(["-rpe", "--reward-prediction-error"], args=args)
    arena.__data_dir__ = directory = os.path.abspath(read_arg(["-d", "--dir"], vtype=str, default=arena.__data_dir__,
                                                     args=args))
    repeat = read_arg(['-r', '--repeat'], vtype=int, default=10, args=args)
    verbose = read_arg(["--verbose", "-v"], vtype=bool, args=args)

    file_names = os.listdir(directory)

    d_raw, cases, d_names, d_repeats = load_arena_paths(file_names, nb_active_kcs=nb_active_kcs, max_repeat=repeat,
                                                        prediction_error=rpe, verbose=verbose)
    plot_arena_paths(d_raw, cases, d_names, d_repeats, "srm", figsize=(5, repeat),
                     name="%sarena-paths-k%d%s" % ("rpe-" if rpe else "", nb_active_kcs,
                                                   "-%02d" % repeat if repeat is not None else ""))


if __name__ == '__main__':
    import sys

    main(*sys.argv)
