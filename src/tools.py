from plot import *

import sys


def a_structure(model, models, only_nids=True):
    return plot_model_structure(model, only_nids=only_nids)


def a_values(model, models, only_nids=True):
    return plot_individuals(models, only_nids=only_nids)


def a_weights(model, models, only_nids=True):
    return plot_weights(models, only_nids=only_nids)


def a_population(model, models, only_nids=True):
    return plot_population(models, only_nids=only_nids)


def a_weight_matrices(model, models, only_nids=True):
    return plot_weights_matrices(models, vmin=-1.5, vmax=1.5, only_nids=only_nids)


def read_arg(flag_list, boolean=True, vtype=None, default=None):
    if vtype is not None:
        boolean = False
    for flag in flag_list:
        if flag in sys.argv:
            if boolean:
                return True
            elif vtype is not None:
                return vtype(sys.argv[sys.argv.index(flag)+1])
            else:
                return sys.argv[sys.argv.index(flag)+1]
    if default is not None:
        return default
    elif boolean:
        return False
    else:
        return None


def run_arg(model, models, only_nids):
    for arg in sys.argv:
        if arg in argf:
            argf[arg](model, models, only_nids)


argf = {
    "-s": a_structure,
    "--struct": a_structure,
    "--structure": a_structure,
    "-v": a_values,
    "--val": a_values,
    "--value": a_values,
    "--values": a_values,
    "-w": a_weights,
    "--weight": a_weights,
    "--weights": a_weights,
    "-p": a_population,
    "--pop": a_population,
    "--polulation": a_population,
    "-m": a_weight_matrices,
    "-wm": a_weight_matrices,
    "--wmatrices": a_weight_matrices,
    "--weight-matrices": a_weight_matrices
}
