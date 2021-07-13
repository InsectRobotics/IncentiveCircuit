from incentive.circuit import IncentiveCircuit

import numpy as np

import re

cs_ids = ["A", "B", "C", "D"]
us_ids = ["+", "-"]


class TMaze(object):

    def __init__(self, train=None, test=None, nb_train=5, nb_test=2, nb_in_trial=100,
                 nb_pn=4, nb_kcs=20, nb_kc_odour=5, learning_rule="dlr", rng=np.random.RandomState()):

        if train is None:
            train = []
        if test is None:
            test = []
        self._nb_train = nb_train
        self._nb_test = nb_test
        self._train = train
        self._test = test

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

        routine = tmaze_routine(self, *args, **kwargs)

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

        return v_out

    def get_test_result(self, test, train=False):
        details = list(re.findall(r"([\w]+) vs ([\w]+)", test)[0])
        v_l = self.get_values(details[0], train=train, test=test)
        v_r = self.get_values(details[1], train=train, test=test)
        return (v_l - v_r) / 2

    @property
    def t(self):
        return self.mb._t

    @property
    def values(self):
        return self.mb._v


def tmaze_routine(agent, train, test, nb_trains=5, nb_tests=2, noise=0.1):
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

        train_cs[i, css] = 4.
        train_us[i, uss] = 4.

    for i, pair in enumerate(test):
        details = re.findall(r"([\w]+) vs ([\w]+)", pair)
        csl, csr = details[0]
        csls, csrs = [], []
        for c in csl:
            csls.append(cs_ids.index(c))
        for c in csr:
            csrs.append(cs_ids.index(c))

        test_l[i, csls] = 4.
        test_r[i, csrs] = 4.

    mb_model = agent.mb
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
