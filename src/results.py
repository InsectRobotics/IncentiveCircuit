from models_base import MBModel

from typing import List


def run_main_experiments(mb_model, reversal=True, unpaired=True, no_shock=True):
    """
    Runs the main experiments of the manuscript, creating a new model with the responses of the neurons for each one of
    them. The experiments are the 'no shock', 'unpaired' and 'reversal'; and it is possible to run a sub-set of them by
    specifying it in the parameters.

    :param mb_model: the mushroom body model that contains the structure of the model.
    :type mb_model: MBModel
    :param reversal: (optional) Whether to run the 'reversal' experiment. Default is True.
    :type reversal: bool
    :param unpaired: (optional) Whether to run the 'unpaired' experiment. Default is True.
    :type unpaired: bool
    :param no_shock: (optional) Whether to run the 'no shock' experiment. Default is True.
    :type no_shock: bool
    :return: a list with copies of the input model that contain the history of the responses of the neurons and the
    synaptic weights during each experiment
    :rtype: List[MBModel]
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

    :param mb_model: the mushroom body model that contains the structure of the model.
    :type mb_model: MBModel
    :param routine: the routine that specifies the experiment to be run
    :return: the new model that contains the history of the responses of the neurons and the synaptic weights of the
    model during the experiment
    :rtype: MBModel
    """
    new_model = mb_model.copy()
    new_model(routine=routine(new_model))
    return new_model

