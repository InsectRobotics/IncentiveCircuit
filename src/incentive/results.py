"""
Package that contains methods that simplify the process of running the different routines with the given models.
"""

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2021, Insect Robotics Group," \
                "Institude of Perception, Action and Behaviour," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "GPLv3+"
__version__ = "v1.0.0-alpha"
__maintainer__ = "Evripidis Gkanias"

from .models_base import MBModel

from typing import List


def run_main_experiments(mb_model, reversal=True, unpaired=True, no_shock=True):
    """
    Runs the main experiments of the manuscript, creating a new model with the responses of the neurons for each one of
    them. The experiments are the 'no shock', 'unpaired' and 'reversal'; and it is possible to run a sub-set of them by
    specifying it in the parameters.

    Parameters
    ----------
    mb_model: MBModel
        the mushroom body model that contains the structure of the model.
    reversal: bool, optional
        whether to run the 'reversal' experiment. Default is True.
    unpaired: bool, optional
        whether to run the 'unpaired' experiment. Default is True.
    no_shock: bool, optional
        whether to run the 'no shock' experiment. Default is True.

    Returns
    -------
    models: List[MBModel]
        a list with copies of the input model that contain the history of the responses of the neurons and the synaptic
        weights during each experiment
    """
    models = []

    if reversal:
        # run acquisition and reversal phases
        rev_model = mb_model.copy()
        models.append(rev_model)
        rev_model(reversal=reversal)

    if unpaired:
        # run acquisition and  unpaired phases
        unp_model = mb_model.copy()
        models.append(unp_model)
        unp_model(unpaired=unpaired)

    if no_shock:
        # run acquisition and  no-shock phases
        nsk_model = mb_model.copy()
        models.append(nsk_model)
        nsk_model(no_shock=no_shock)

    return models


def run_custom_routine(mb_model, routine):
    """
    Creates a copy of the input model and runs a customised routine that takes as input the model and returns a stimuli
    generator.

    Parameters
    ----------
    mb_model: MBModel
        the mushroom body model that contains the structure of the model.
    routine
        the routine that specifies the experiment to be run

    Returns
    -------
    model: MBModel
        the new model that contains the history of the responses of the neurons and the synaptic weights of the model
        during the experiment
    """
    new_model = mb_model.copy()
    new_model(routine=routine(new_model))
    return new_model
