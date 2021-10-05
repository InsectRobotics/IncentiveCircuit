# #!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualises the responses of all the recorded neurons.

Examples:
In order to visualise the data for 100 flies in each condition, run
    $ python3 run_arena_paths.py --dir ../data/arena
or
    $ python3 run_arena_paths.py -d ../data/arena

In order to visualise the respective data using the prediction-error learning rule, run
    $ python3 run_arena_paths.py -d ../data/arena -rw
"""

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright 2021, School of Informatics, the University of Edinburgh"
__licence__ = "MIT"
__version__ = "1.1-alpha"
__maintainer__ = "Evripidis Gkanias"
__email__ = "ev.gkanias@ed.ac.uk"
__status__ = "Production"

from incentive.imaging import load_data, plot_phase_overlap_mean_responses_from_data
from incentive.tools import read_arg


if __name__ == '__main__':

    # read the parameters
    experiment = read_arg(["-e", "--experiment"], vtype=str, default="B+")
    verbose = read_arg(["-v", "--verbose"])
    directory = read_arg(["-d", "--dir"], vtype=str, default=None)

    # load the data
    df = load_data(experiment, directory=directory)

    # generate statistics
    if read_arg(["-s", "--stats", "--statistics"]):

        # initialise statistics
        nb_neurons, nb_flies, nb_flies_min, nb_flies_max = 0, 0, 14, 0

        for name in df[experiment].index:
            nb_flies += df[experiment][name].shape[1]
            nb_neurons += 1
            if nb_flies_min > df[experiment][name].shape[1]:
                nb_flies_min = df[experiment][name].shape[1]
            if nb_flies_max < df[experiment][name].shape[1]:
                nb_flies_max = df[experiment][name].shape[1]
            if verbose:
                print(name, df[experiment][name].shape)

        print("#neurons:", nb_neurons)
        print("#flies:", nb_flies)
        print("min #flies/neuron:", nb_flies_min)
        print("max #flies/neuron:", nb_flies_max)
        print("mean #flies/neuron:", nb_flies / nb_neurons)

    if read_arg(["-a", "--all"]):
        # plot the data from all the available neurons
        plot_phase_overlap_mean_responses_from_data(df, experiment)
    else:
        # plot the data from the selected neurons for the TSM model
        neurons = [33, 39, 21, 41, 42, 30, 13, 16, 14, 17, 12, 2]
        plot_phase_overlap_mean_responses_from_data(df, experiment, nids=neurons, only_nids=read_arg(["--only-nids"]))
