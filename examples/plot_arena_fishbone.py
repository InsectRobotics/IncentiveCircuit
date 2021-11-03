# #!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright 2021, School of Informatics, the University of Edinburgh"
__licence__ = "MIT"
__version__ = "1.1-alpha"
__maintainer__ = "Evripidis Gkanias"
__email__ = "ev.gkanias@ed.ac.uk"
__status__ = "Production"

from incentive import arena
from incentive.tools import read_arg
from incentive.plot import plot_arena_fishbone
from incentive.arena import load_arena_stats

import os


if __name__ == '__main__':

    nb_active_kcs = 5

    rpe = read_arg(["-rpe", "--reward-prediction-error"])
    arena.__data_dir__ = directory = os.path.abspath(read_arg(["-d", "--dir"], vtype=str, default=arena.__data_dir__))
    repeats = read_arg(["-R", "--repeats"], vtype=int, default=10)

    file_names = os.listdir(directory)

    df = load_arena_stats(file_names, nb_active_kcs=nb_active_kcs, prediction_error=rpe, verbose=True)

    plot_arena_fishbone(df, odours_visited="A and B", code="srm", rpe=rpe, verbose=True,
                        nb_repeats=repeats)
