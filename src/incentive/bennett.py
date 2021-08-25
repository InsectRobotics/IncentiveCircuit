"""
Examples:
---------
>>> ben = Bennett(train=['A', 'B'], test=['A vs B'], nb_train=10, nb_test=2, nb_in_trial=10)
>>> ben(excite=['MBON_av'], inhibit=[], intervention=0, noise=0.1)
"""

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2021, Insect Robotics Group," \
                "Institute of Perception, Action and Behaviour," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "GPLv3+"
__version__ = "v1.1-dev"
__maintainer__ = "Evripidis Gkanias"

from incentive.circuit import IncentiveCircuit

import numpy as np
import pandas as pd

import re

cs_ids = ["A", "B"]
"""
The allowed odour identities.
"""
us_ids = ["+", "-"]
"""
The allowed reinforcements.
"""


class Bennett(object):

    def __init__(self, train=None, test=None, nb_train=10, nb_test=2, nb_in_trial=100, target_intervention=None,
                 nb_pn=2, nb_kcs=20, nb_kc_odour=10, learning_rule="dlr", rng=np.random.RandomState()):
        """
        Simulates the behaviour of flies replicating Figure 5 from Bennett et al (2021) [1]_.

        Creates the experimental set-up in order to test predefined conditions using the incentive circuit.

        Notes
        -----
        .. [1] Bennett, J. E. M., Philippides, A. & Nowotny, T. Learning with reinforcement prediction errors in a model
               of the Drosophila mushroom body. Nat Commun 12, 2569 (2021).

        Parameters
        ----------
        train : list[str], optional
            a list of strings determining the order of the odour mixtures presented and whether they were paired with a
            US (sugar = '+', electric shock = '-'). Default is the empty list
        test : list[str], optional
            a list of strings determining the conditions (e.g. 'A vs B') that are going to be tested. Default is the
            empty list
        nb_train : int, optional
            the number of times that the training phase will be repeated before the test. Default is 10
        nb_test : int, optional
            the number of times that the testing phase will be repeated. Default is 2
        nb_in_trial : int, optional
            the number of time-steps determine the duration of each trial (training or test). Default is 100
        target_intervention : list[str], optional
            the target DANs or MBONs for the intervention. When 'None', it targets all neurons. By default all neurons
            are targeted
        nb_pn : int, optional
            the number of the projection neurons (PNs) of the insect brain. Default is 2
        nb_kcs : int, optional
            the number of the kenyon cells (KCs) of the mushroom body (MB). Default is 20
        nb_kc_odour : int, optional
            the number of KCs associated to each odour. Default is 10
        learning_rule : str, optional
            the learning rule of the mushroom body (MB). Default is 'dlr' (for dopaminergic learning rule)
        rng : np.random.RandomState, optional
            the random number generator
        """

        if train is None:
            train = []
        if test is None:
            test = []
        self._nb_train = nb_train
        self._nb_test = nb_test
        self._train = train
        self._test = test
        self.target_intervention = target_intervention

        nb_trials = nb_train * len(train) + nb_test * len(test)
        self.mb = IncentiveCircuit(
            nb_pn=nb_pn, nb_kc=nb_kcs, nb_kc_odour=nb_kc_odour,
            learning_rule=learning_rule, nb_timesteps=nb_in_trial, nb_trials=nb_trials, has_real_names=False,
            has_sm=True, has_rm=True, has_ltm=True, has_rrm=True, has_rfm=True, has_mam=True
        )
        self.mb.w_p2k *= self.mb.nb_pn

        self.rng = rng

    def __call__(self, *args, **kwargs):
        """
        Runs the routine replicating the Bennett et al (2021) batch experiment.

        Parameters
        ----------
        train : list[str]
            a list of strings determining the order of the odour mixtures presented and whether they were paired with a
            US (sugar = '+', electric shock = '-'). Default is the list of internal training trials.
        test : list[str]
            a list of strings determining the conditions (e.g. 'A vs B') that are going to be tested. Default is the
            list of internal testing trials.
        nb_train : int
            the number of times that the training phase will be repeated before the test. Default is the internal number
            of training trials.
        nb_test : int
            the number of times that the testing phase will be repeated. Default is the internal number of testing
            trials.
        """
        kwargs.setdefault("train", self._train)
        kwargs.setdefault("test", self._test)
        kwargs.setdefault("nb_trains", self._nb_train)
        kwargs.setdefault("nb_tests", self._nb_test)

        self.mb.w_k2m[0] = self.mb.w_k2m[-1].copy()

        routine = bennett_routine(self, *args, **kwargs)

        return self.mb(routine=routine)

    def get_values(self, odour_id, train=True, test=True):
        """
        Calculates the attraction/avoidance value associated with the given odour identity.

        It calculates the attraction/avoidance value using the MBON responses and returns the values associated with the
        timings when the given odour ID was present. Optionally, you can choose whether the training or testing trials
        should be included.

        Parameters
        ----------
        odour_id : str
            the odour identity
        train : bool, str
            when True, it includes the responses from the training trials. When it is a string, it gets the values
            associated to a specific training phase. Default is True.
        test : bool, str
            when True, it includes the responses from the testing trials. When it is a string, it gets the values
            associated to a specific testing phase. Default is True.

        Returns
        -------
        np.ndarray[float]
            the attraction/avoidance value associated with the odour identity as a function of time
        """

        nb_neurons = self.mb.nb_dan + self.mb.nb_mbon
        v = self.values[1:].reshape((-1, self.mb.nb_timesteps, nb_neurons))
        nb_v_train = self._nb_train * self.mb.nb_timesteps * int(train is not False)
        nb_v_test = self._nb_test * (self.mb.nb_timesteps // 2) * int(test is not False)
        v_out = np.full(nb_v_train + nb_v_test, np.nan)

        if train:
            ids = []
            if not isinstance(train, str):
                for i, case in enumerate(self._train):
                    details = re.findall(r"([\w]+)[+-]?", case)[0]
                    if odour_id == details:
                        ids.append(i)
            else:
                ids.append(self._train.index(train))
            for i in ids:
                v_i = v[i:i+self._nb_train].reshape((-1, nb_neurons))
                v_out[:nb_v_train] = np.mean((v_i[:, 6::2] - v_i[:, 7::2]) /
                                             np.maximum(v_i[:, 6::2] + v_i[:, 7::2], np.finfo(float).eps), axis=1)

        if test:
            ids = []
            jds = []
            if not isinstance(test, str):
                for i, case in enumerate(self._test):
                    details = list(re.findall(r"([\w]+) vs ([\w]+)", case)[0])
                    if odour_id in details:
                        ids.append(i)
                        jds.append(details.index(odour_id))
            else:
                ids.append(self._test.index(test))
                details = list(re.findall(r"([\w]+) vs ([\w]+)", test)[0])
                jds.append(details.index(odour_id))

            v_outs = []
            for i, j in zip(ids, jds):
                vv = v[self._nb_train * len(self._train):]
                v_ij = vv[i::len(self._test), j::2][:self._nb_test].reshape((-1, nb_neurons))
                v_out[nb_v_train:] = np.mean((v_ij[:, 6::2] - v_ij[:, 7::2]) / 2, axis=1)
                v_outs.append(v_out.copy())
            if len(v_outs) > 0:
                v_out = np.array(v_outs)

        if v_out.ndim < 2:
            v_out = v_out[np.newaxis, ...]

        return v_out

    def get_pi(self, test, train=False):
        """
        Calculates the preference index (PI) for a specific test.

        The PI is calculated by subtracting the attraction/avoidance value of the second odour from the one of the first
        odour of the test and divide this by 2.

        Parameters
        ----------
        test : str
            the test to get the results from
        train : bool, str
            when True, it includes the responses from the training trials. When it is a string, it gets the values
            associated to a specific training phase. Default is False.

        Returns
        -------
        np.ndarray[float]
            the preference index (PI) for the given test as a function of time.
        """
        details = list(re.findall(r"([\w]+) vs ([\w]+)", test)[0])
        v_l = self.get_values(details[0], train=train, test=test)
        v_r = self.get_values(details[1], train=train, test=test)
        return (v_l - v_r) / np.maximum(np.absolute(v_l) + np.absolute(v_r), np.finfo(float).eps)

    @property
    def t(self):
        """
        The current time-step based on the clock of the associated fly (mushroom body).
        """
        return self.mb._t

    @property
    def values(self):
        """
        The time-dependent responses of the DANs and MBONs in the mushroom body.
        """
        return self.mb._v


def bennett_routine(agent, train, test, excite=None, inhibit=None, intervention=None,
                    nb_trains=10, nb_tests=2, noise=0.1):
    """
    The Bennett et al (2021) Figure 5 experiment generator.

    Takes as input the agent and the experiment parameters and generates the CS and US of the experiment for each
    time-step that can be used as an input to the model.

    Parameters
    ----------
    agent : Bennett
        the agent to apply the experiment.
    train : list[str]
        a list of strings determining the order of the odour mixtures presented and whether they were paired with a
        US (sugar = '+', electric shock = '-')
    test : list[str]
        a list of strings determining the conditions (e.g. 'A vs B') that are going to be tested
    excite : list[str], optional
        a list of strings determining the neurons that will get excited by intervention. Neuron names can be: 'MBON_at',
        'MBON_av', 'DAN_at' or 'DAN_av'. Default is None
    inhibit : list[str], optional
        a list of strings determining the neurons that will get inhibited by intervention. Neuron names can be:
        'MBON_at', 'MBON_av', 'DAN_at' or 'DAN_av'. Default is None
    intervention : {1, 2, 3, 4}, optional
        integer in [1-4] that reveals the code of the intervention protocol: 1 - training (CS+ only), 2 - Training (CS+
        and CS-), 3 - Testing only, 4 - Training (CS+ and CS-) and testing. Default is None
    nb_trains : int, optional
        the number of times that the training phase will be repeated before the test. Default is 10
    nb_tests : int, optional
        the number of times that the testing phase will be repeated. Default is 2
    noise : float, optional
        the magnitude of Gaussian noise to be applied on the CS signal. Default is 0.1

    Yields
    ------
    tuple[int, int, np.ndarray[float], np.ndarray[float]]
        a tuple of (trial, time-step, CS, US) which a sequence of the inputs and times for the model, generating the
        experience of the agent.
    """
    train_cs = np.zeros((len(train), len(cs_ids)), dtype=float)
    train_us = np.zeros((len(train), len(us_ids)), dtype=float)
    test_l = np.zeros((len(test), len(cs_ids)), dtype=float)
    test_r = np.zeros((len(test), len(cs_ids)), dtype=float)
    intervention_schedule = {
        "train_CS+": bool(int("{0:03b}".format(intervention)[0])),
        "train_CS-": bool(int("{0:03b}".format(intervention)[1])),
        "test": bool(int("{0:03b}".format(intervention)[2]))
    }

    for i, pair in enumerate(train):
        details = re.findall(r"([A-Z]+)([-+0]?)", pair)
        cs, us = details[0]
        if us == '0':
            us = '+-'  # create neutral reinforcement
        css, uss = [], []
        for c in cs:
            css.append(cs_ids.index(c))
        for u in us:
            uss.append(us_ids.index(u))

        train_cs[i, css] = 2. / np.maximum(len(cs), 1.)
        train_us[i, uss] = 2. / np.maximum(len(us), 1.)

    for i, pair in enumerate(test):
        details = re.findall(r"([A-Z]+) vs ([A-Z]+)", pair)
        csl, csr = details[0]
        csls, csrs = [], []
        for c in csl:
            csls.append(cs_ids.index(c))
        for c in csr:
            csrs.append(cs_ids.index(c))

        test_l[i, csls] = 2. / np.maximum(len(csl), 1.)
        test_r[i, csrs] = 2. / np.maximum(len(csr), 1.)

    mb_model = agent.mb
    mb_model._t = 0
    mb_model.routine_name = "bennett2021"

    for i, cs, us in zip(range(len(train)), train_cs, train_us):
        for trial in range(nb_trains):
            if mb_model._t >= mb_model.nb_trials * mb_model.nb_timesteps:
                break

            for timestep in range(mb_model.nb_timesteps):
                if intervention_schedule["train_CS+"] and np.sum(us) > 0:
                    add_intervention(mb_model, excite, inhibit, target=agent.target_intervention)
                if intervention_schedule["train_CS-"] and np.sum(us) == 0:
                    add_intervention(mb_model, excite, inhibit, target=agent.target_intervention)
                yield i * nb_trains + trial, timestep, cs + agent.rng.randn(cs.shape[0]) * noise, us
                mb_model._t += 1

    for i, csl, csr in zip(range(len(train)), test_l, test_r):
        for trial in range(nb_tests):
            if mb_model._t >= mb_model.nb_trials * mb_model.nb_timesteps:
                break

            for timestep in range(mb_model.nb_timesteps // 2):
                trial_ = nb_trains * len(train) + i * nb_tests + trial
                if intervention_schedule["test"]:
                    add_intervention(mb_model, excite, inhibit, target=agent.target_intervention)
                yield trial_, timestep * 2, csl + agent.rng.randn(csl.shape[0]) * noise, np.zeros_like(train_us[0])
                mb_model._t += 1

                yield trial_, timestep * 2 + 1, csr + agent.rng.randn(csr.shape[0]) * noise, np.zeros_like(train_us[0])
                mb_model._t += 1


def add_intervention(mb_model, excite, inhibit, target=None):
    """
    Sets up the model for intervention to a target neuron.

    Given the abstract description of the target neuron in the 'excite' and 'inhibit' variables, this function adds an
    intervention to the target neuron types described by the 'target' variable. The interventions occurs on the
    intersection between the abstract (i.e. 'excite' or 'inhibit') and the more specific (i.e. 'target') description of
    the target neurons.

    Parameters
    ----------
    mb_model : IncentiveCircuit
        the model to add the intervention.
    excite : list[str]
        a list of strings determining the neurons that will get excited by intervention. Neuron names can be: 'MBON_at',
        'MBON_av', 'DAN_at' or 'DAN_av'.
    inhibit : list[str]
        a list of strings determining the neurons that will get inhibited by intervention. Neuron names can be:
        'MBON_at', 'MBON_av', 'DAN_at' or 'DAN_av'.
    target : str, optional
        the target neuron types of the model for the intervention. Default is None.
    """
    for i, i_type in enumerate([excite, inhibit]):
        for neuron in i_type:
            if neuron is not None and "DAN" in neuron:
                if target is None or "d" in target:
                    mb_model.add_intervention("d_{%s}" % neuron[-2:], excite=i < 1, inhibit=i > 0)
                if target is None or "c" in target:
                    mb_model.add_intervention("c_{%s}" % neuron[-2:], excite=i < 1, inhibit=i > 0)
                if target is None or "f" in target:
                    mb_model.add_intervention("f_{%s}" % neuron[-2:], excite=i < 1, inhibit=i > 0)
            elif neuron is not None and "MBON" in neuron:
                if target is None or "s" in target:
                    mb_model.add_intervention("s_{%s}" % neuron[-2:], excite=i < 1, inhibit=i > 0)
                if target is None or "r" in target:
                    mb_model.add_intervention("r_{%s}" % neuron[-2:], excite=i < 1, inhibit=i > 0)
                if target is None or "m" in target:
                    mb_model.add_intervention("m_{%s}" % neuron[-2:], excite=i < 1, inhibit=i > 0)


def fraction_to_cs_plus(pi):
    return (pi + 1) / 2


def pi_difference(pi_condition, pi_control):
    return pi_condition - pi_control


def pi_binomial_adjustment(pi_condition, pi_control):
    """
    The binomial adjustment of the preference index (PI) as described by Bennett et al (2021).

    Parameters
    ----------
    pi_condition : np.ndarray[float], float
        the PI of the normal condition
    pi_control : np.ndarray[float], float
        the PI of the control condition

    Returns
    -------
    np.ndarray[float]
        the binomial adjustment of the preference index.
    """
    ft_condition = fraction_to_cs_plus(pi_condition)
    ft_control = fraction_to_cs_plus(pi_control)
    d = ft_condition - ft_control
    z = np.maximum(ft_condition + ft_control, np.finfo(float).eps)
    return d / np.sqrt(z / 2 * (1 - z / 2) * 2 / 50)


def read_data(file_path):
    """
    Loads and cleans the data from the EXCEL file.

    Parameters
    ----------
    file_path : str
        the path for the EXCEL file

    Returns
    -------
    pd.DataFrame
        the pandas DataFrame that contains all the important information from the data.
    """
    excel = pd.read_excel(file_path, engine="openpyxl",
                          header=0, nrows=165,  skiprows=[1],
                          index_col=None, usecols="A,X,Y,AC:AJ")

    excel["PI difference"] = excel[" PI difference (binomial adjustment used in paper)"]
    excel.drop(" PI difference (binomial adjustment used in paper)", axis='columns', inplace=True)

    excel["Testing time"] = excel["Testing time (miutes after training)"]
    excel.drop("Testing time (miutes after training)", axis='columns', inplace=True)

    # clean up unreported data
    excel = excel[~pd.isna(excel["PI difference"])]

    return excel


def translate_condition_code(code):
    """
    Translates the condition code from the EXCEL to an experimental set-up.

    Parameters
    ----------
    code : int
        the condition code

    Returns
    -------
    dict
        a dictionary with the extracted information from the condition code.
    """
    a = code // 1000
    b = (code % 1000) // 100
    c = (code % 100) // 10
    d = code % 10

    reward = ["-", "+", ""]
    targets = ["MBON_at", "MBON_av", "DAN_at", "DAN_av"]

    return {
        "train": ["A%s" % reward[d-1], "B"],
        "test": ["A vs B"],
        "inhibit": [targets[b-1]] if c == 1 else [],
        "excite": [targets[b-1]] if c == 2 else [],
        "intervention-schedule": a
    }
