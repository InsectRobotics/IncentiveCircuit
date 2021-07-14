from incentive.circuit import IncentiveCircuit

import numpy as np
import pandas as pd

import re

cs_ids = ["A", "B"]
us_ids = ["+", "-"]


class Bennett(object):

    def __init__(self, train=None, test=None, nb_train=10, nb_test=2, nb_in_trial=100, target_intervention=None,
                 nb_pn=2, nb_kcs=20, nb_kc_odour=10, learning_rule="dlr", rng=np.random.RandomState()):

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

        self.rng = rng

    def __call__(self, *args, **kwargs):
        kwargs.setdefault("train", self._train)
        kwargs.setdefault("test", self._test)
        kwargs.setdefault("nb_trains", self._nb_train)
        kwargs.setdefault("nb_tests", self._nb_test)

        self.mb.w_k2m[0] = self.mb.w_k2m[-1].copy()

        routine = bennett_routine(self, *args, **kwargs)

        return self.mb(routine=routine)

    def get_values(self, odour_id, train=True, test=True):

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
        details = list(re.findall(r"([\w]+) vs ([\w]+)", test)[0])
        v_l = self.get_values(details[0], train=train, test=test)
        v_r = self.get_values(details[1], train=train, test=test)
        return (v_l - v_r) / np.maximum(np.absolute(v_l) + np.absolute(v_r), np.finfo(float).eps)

    @property
    def t(self):
        return self.mb._t

    @property
    def values(self):
        return self.mb._v


def bennett_routine(agent, train, test, excite=None, inhibit=None, intervention=None,
                    nb_trains=10, nb_tests=2, noise=0.1):
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

        test_l[i, csls] = 2. / np.maximum(len(cs), 1.)
        test_r[i, csrs] = 2. / np.maximum(len(us), 1.)

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
    ft_condition = fraction_to_cs_plus(pi_condition)
    ft_control = fraction_to_cs_plus(pi_control)
    d = ft_condition - ft_control
    z = np.maximum(ft_condition + ft_control, np.finfo(float).eps)
    return d / np.sqrt(z / 2 * (1 - z / 2) * 2 / 50)


def read_data(file_path):
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
