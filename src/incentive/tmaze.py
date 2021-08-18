"""
Examples:
---------
>>> maze = TMaze(train=['A-', 'B'], test=['A vs B'], nb_train=4, nb_test=2, nb_in_trial=100, nb_samples=100)
>>> maze(noise=.5)
"""

from incentive.circuit import IncentiveCircuit

import numpy as np

import re

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2021, Insect Robotics Group," \
                "Institute of Perception, Action and Behaviour," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "GPLv3+"
__version__ = "v1.1-dev"
__maintainer__ = "Evripidis Gkanias"

cs_ids = ["A", "B", "C", "D"]
"""
The allowed odour identities.
"""
us_ids = ["+", "-"]
"""
The allowed reinforcements.
"""


class TMaze(object):
    def __init__(self, train=None, test=None, nb_train=5, nb_test=2, nb_in_trial=100, nb_samples=100,
                 nb_pn=4, nb_kcs=20, nb_kc_odour=5, learning_rule="dlr", rng=np.random.RandomState()):
        """
        Simulated fly in the T-maze.

        It sets up a T-maze experiment for a number of individual flies where CS and US are presented during the
        training phases and then the flies are free to choose between the different CS during the test phase.

        Parameters
        ----------
        train : list[str], optional
            a list of strings determining the order of the odour mixtures presented and whether they were paired with a
            US (sugar = '+', electric shock = '-'). Default is the empty list
        test : list[str], optional
            a list of strings determining the conditions (e.g. 'A vs B') that are going to be tested. Default is the
            empty list
        nb_train : int, optional
            the number of times that the training phase will be repeated before the test. Default is 5
        nb_test : int, optional
            the number of times that the testing phase will be repeated. Default is 2
        nb_in_trial : int, optional
            the number of time-steps determine the duration of each trial (training or test). Default is 100
        nb_samples : int, optional
            the number of samples determine the number of the individual flies tested. Default is 100
        nb_pn : int, optional
            the number of the projection neurons (PNs) of the insect brain. Default is 4
        nb_kcs : int, optional
            the number of the kenyon cells (KCs) of the mushroom body (MB). Default is 20
        nb_kc_odour : int, optional
            the number of KCs associated to each odour. Default is 5
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
        self._nb_samples = nb_samples

        nb_trials = nb_train * len(train) + nb_test * len(test)
        self.mb = []
        for s in range(nb_samples):
            self.mb.append(IncentiveCircuit(
                nb_pn=nb_pn, nb_kc=nb_kcs, nb_kc_odour=nb_kc_odour,
                learning_rule=learning_rule, nb_timesteps=nb_in_trial, nb_trials=nb_trials, has_real_names=False,
                has_sm=True, has_rm=True, has_ltm=True, has_rrm=True, has_rfm=True, has_mam=True
            ))
            self.mb[-1].bias[:] = 0.

        self.rng = rng

    def __call__(self, *args, **kwargs):
        """
        Runs the T-Maze routine for all the flies.

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

        for mb in self.mb:
            mb.w_k2m[0] = mb.w_k2m[-1].copy()
            routine = tmaze_routine(self, mb, *args, **kwargs)
            mb(routine=routine)

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

        vs_out = []

        for mb in self.mb:
            nb_neurons = mb.nb_dan + mb.nb_mbon
            v = mb._v[1:].reshape((-1, mb.nb_timesteps, nb_neurons))
            nb_v_train = self._nb_train * mb.nb_timesteps * int(train is not False)
            nb_v_test = self._nb_test * (mb.nb_timesteps // 2) * int(test is not False)
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
                    v_i = v[i::len(self._train)][:self._nb_train].reshape((-1, nb_neurons))
                    v_out[:nb_v_train] = np.mean(v_i[:, 6::2] - v_i[:, 7::2], axis=1) / 2

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
                    v_out[nb_v_train:] = np.mean(v_ij[:, 6::2] - v_ij[:, 7::2], axis=1) / 2
                    v_outs.append(v_out.copy())
                if len(v_outs) > 0:
                    v_out = np.array(v_outs)

            if v_out.ndim < 2:
                v_out = v_out[np.newaxis, ...]

            vs_out.append(v_out)

        return np.mean(vs_out, axis=-1)

    def get_test_result(self, test, train=False):
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
        return (v_l - v_r) / 2

    @property
    def t(self):
        """
        The current time-step based on the internal clock of the first fly.
        """
        return self.mb[0]._t


def tmaze_routine(agent, mb_model, train, test, nb_trains=5, nb_tests=2, noise=0.1):
    """
    The T-Maze experiment generator.

    Takes as input the agent, the mushroom body model and the experiment parameters and generates the CS and US of the
    T-Maze experiment for each time-step that can be used as an input to the model.

    Parameters
    ----------
    agent : TMaze
        the agent to apply the experiment.
    mb_model : IncentiveCircuit
        the mushroom body model (associated with a fly) from the agent.
    train : list[str]
        a list of strings determining the order of the odour mixtures presented and whether they were paired with a
        US (sugar = '+', electric shock = '-')
    test : list[str]
        a list of strings determining the conditions (e.g. 'A vs B') that are going to be tested
    nb_trains : int, optional
        the number of times that the training phase will be repeated before the test
    nb_tests : int, optional
        the number of times that the testing phase will be repeated
    noise : float, optional
        the magnitude of Gaussian noise to be applied on the CS signal

    Yields
    ------
    tuple[int, int, np.ndarray[float], np.ndarray[float]]
        a tuple of (trial, time-step, CS, US) which a sequence of the inputs and times for the model, generating the
        experience of the agent.
    """

    assert mb_model in agent.mb, "The input 'mb_model' is not assigned in the experiment (agent)."

    train_cs = np.zeros((len(train), len(cs_ids)), dtype=float)
    train_us = np.zeros((len(train), len(us_ids)), dtype=float)
    test_l = np.zeros((len(test), len(cs_ids)), dtype=float)
    test_r = np.zeros((len(test), len(cs_ids)), dtype=float)

    for i, pair in enumerate(train):
        details = re.findall(r"([\w]+)([-+]?)", pair)
        cs, us = details[0]
        css, uss = [], []
        for c in cs:
            css.append(cs_ids.index(c))
        for u in us:
            uss.append(us_ids.index(u))

        train_cs[i, css] = 1.
        train_us[i, uss] = 2.

    for i, pair in enumerate(test):
        details = re.findall(r"([\w]+) vs ([\w]+)", pair)
        csl, csr = details[0]
        csls, csrs = [], []
        for c in csl:
            csls.append(cs_ids.index(c))
        for c in csr:
            csrs.append(cs_ids.index(c))

        test_l[i, csls] = 1.
        test_r[i, csrs] = 1.

    mb_model._t = 0
    mb_model.routine_name = "t-maze"

    for trial in range(nb_trains):
        if mb_model._t >= mb_model.nb_trials * mb_model.nb_timesteps:
            break

        for i, cs, us in zip(range(len(train)), train_cs, train_us):
            for timestep in range(mb_model.nb_timesteps):
                yield trial * len(train) + i, timestep, cs + agent.rng.randn(cs.shape[0]) * noise, us
                mb_model._t += 1

    for trial in range(nb_tests):
        if mb_model._t >= mb_model.nb_trials * mb_model.nb_timesteps:
            break

        for i, csl, csr in zip(range(len(train)), test_l, test_r):
            for timestep in range(mb_model.nb_timesteps // 2):
                trial_ = nb_trains * len(train) + trial * len(test) + i
                yield trial_, timestep * 2, csl + agent.rng.randn(csl.shape[0]) * noise, np.zeros_like(train_us[0])
                mb_model._t += 1

                yield trial_, timestep * 2 + 1, csr + agent.rng.randn(csr.shape[0]) * noise, np.zeros_like(train_us[0])
                mb_model._t += 1
