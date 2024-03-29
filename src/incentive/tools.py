"""
Package that connects the input arguments and parameters into the processes that create the results of the manuscript.
"""

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2021, Insect Robotics Group," \
                "Institute of Perception, Action and Behaviour," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "GPLv3+"
__version__ = "v1.1-alpha"
__maintainer__ = "Evripidis Gkanias"

from .plot import *
from .models_base import MBModel

import sys
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def a_values(model, models, only_nids=True):
    """
    Plot function for the responses of the model as lines.

    Parameters
    ----------
    model: MBModel
        the initial model that contains the pure structure
    models: list[list[MBModel]]
        a list of the models generated by the routines
    only_nids: bool, optional
        specify if we want to plot only the targeted neuron indices. Default is True
    """
    return plot_responses_from_model(models, only_nids=only_nids)


def a_weights(model, models, only_nids=True):
    """
    Plot function for the KC-MBON weights of the model.

    Parameters
    ----------
    model: MBModel
        the initial model that contains the pure structure
    models: list[MBModel]
        a list of the models generated by the routines
    only_nids: bool, optional
        specify if we want to plot only the targeted neuron indices. Default is True
    """
    return plot_weights(models, only_nids=only_nids)


def read_arg(flag_list, args=None, boolean=True, vtype=None, default=None):
    """
    Reads the flag list and tries to identify the parameter in the arguments of the main function. It then transforms
    the argument into the parameter for the system. If multiple flags has been found in the arguments, only first one
    will be processed.

    Parameters
    ----------
    args: list[str]
        the arguments of the main.
    flag_list: list[str]
        list of flags to search for in the arguments of the main.
    boolean: bool, optional
        specifies if we just want to see if the flag exists in the arguments. Default is True.
    vtype: type, optional
        specifies the type of the value that we want to read if the flag is not boolean; if None, then no occurs.
        Default is None.
    default: optional
        the default value to return in case the flag is not found in the arguments. Default is None.

    Returns
    -------
    value
        whether any of the flags has been found in the arguments, the value found casted by the specified type or
        default value in case that none of the flags has been found.
    """
    if args is None:
        args = sys.argv

    if vtype is not None:
        boolean = False
    for flag in flag_list:
        if flag in args:
            if boolean:
                return True
            elif vtype is not None:
                return vtype(args[args.index(flag)+1])
            else:
                return args[args.index(flag)+1]
    if boolean:
        return False
    else:
        return default


def run_arg(model, models, only_nids, args=None):
    """
    Runs all the plotting functions that have been identified in the arguments set, using the specified  parameters.

    Parameters
    ----------
    model: MBModel
        the initial model that contains the pure structure
    models: list[MBModel]
        a list of the models generated by the routines
    only_nids: bool
        specify if we want to plot only the targeted neuron indices
    args: list[str]
        the arguments of the main
    """
    if args is None:
        args = sys.argv

    for arg in args:
        if arg in argf:
            argf[arg](model, models, only_nids)


argf = {
    "-v": a_values,
    "--val": a_values,
    "--value": a_values,
    "--values": a_values,
    "-w": a_weights,
    "--weight": a_weights,
    "--weights": a_weights,
}
"""map of the different flags to plotting function"""
