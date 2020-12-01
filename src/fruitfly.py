from invplot.brain.mbplot import *
from utils import info, debug, set_verbose, __root__
from simulation.olfactory_learning import OlfactoryLearning

import os
import re
import csv
import pandas as pd
import numpy as np
import yaml
import string

__dir__ = os.path.dirname(os.path.abspath(__file__))
__data_dir__ = os.path.join(__root__, "data", "FruitflyMB")

with open(os.path.join(__data_dir__, 'meta.yaml'), 'rb') as f:
    meta = yaml.load(f, Loader=yaml.BaseLoader)
eps = np.finfo(float).eps


class Data(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def load_data(dataset='B+S'):
        pattern = {
            # pattern for the initial data
            'B+S': r'realSCREEN_([\d\w\W]+)\.xlsx_finaldata([\w\W]+)_timepoint(\d)\.csv',
            # pattern for the control data
            'A+S': r'(realSCREEN_){0,1}([\d\w\W]+)_O\+S\.xlsx_finaldata([\w\W]+)_timepoint(\d)\.csv',
            # pattern for the no-shock data
            'B+NS': r'(realSCREEN_){0,1}([\d\w\W]+)_M\+NS\.xlsx_finaldata([\w\W]+)_timepoint(\d)\.csv',
            # pattern for the KC data
            'KC': r'realSCREEN_([\d\w\W]+)\.xlsx_finaldata([\w\W]+)_timepoint(\d)\.csv'
        }

        directory = {
            'B+S': '',
            'A+S': 'SF traces imaging controls',
            'B+NS': 'SF traces imaging controls',
            'KC': 'neural traces KC sub-compartments'
        }

        if isinstance(dataset, str):
            if dataset == 'all':
                dataset = pattern.keys()
            else:
                dataset = [dataset]

        assert np.all([ds in pattern.keys() for ds in dataset]), (
                'Accepted dataset names are: %s.' % [', '.join(pattern.keys())])

        data = {}
        for ds in dataset:
            dataset_dir = os.path.join(__data_dir__, directory[ds])
            for filename in os.listdir(dataset_dir):
                details = re.findall(pattern[ds], filename)
                if len(details) == 0:
                    continue

                temp = details[0]
                if len(temp) > 3:
                    _, genotype, odour, trial = temp[:4]
                elif len(temp) > 2:
                    genotype, odour, trial = temp
                else:
                    err('Information in the filename is not sufficient: %s' % filename, console=True)
                    err('Skipping file!', console=True)
                    continue
                # debug(genotype, odour, trial, console=True)
                trial = int(trial)

                # print genotype, odour, trial

                timepoint = None
                filename = os.path.join(dataset_dir, filename)
                with open(filename, 'r') as csvfile:
                    reader = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_NONNUMERIC)
                    for row in reader:
                        if timepoint is None:
                            timepoint = row
                        else:
                            timepoint = np.vstack([timepoint, row])  # {timepoint} x {datapoint}

                name = "%s-%s-%s" % (genotype, odour, ds)
                if name not in data.keys():
                    data[name] = [[]] * 9
                # debug(name, trial, len(data[name][trial-1]), timepoint.shape, console=True)
                data[name][trial - 1] = timepoint

            for name in data.keys():
                debug(len(data[name]), *[d.shape for d in data[name]], console=True)
                data_name_array = np.array(data[name])
                debug(name, data_name_array.shape, console=True)
                data[name] = data_name_array

        return data


class DataFrame(Data):
    comps = {
        u"\u03b1": [['p', 'c', 's']] * 3,
        u"\u03b2": [['p', 'c', 's']] * 2,
        u"\u03b1'": [['m', 'p', 'a']] * 3,
        u"\u03b2'": [['m', 'p', 'a']] * 2,
        u"\u03b3": [['m', 'd']] * 5
    }

    def __init__(self, dataset=None, deltatime=0.2, short=False, **kwargs):
        """

        Parameters
        ----------
        dataset : str, list of str
            name(s) of the data-sets to load -- a set of: 'A+S', 'B+S', 'B+NS', 'KC' or 'all'.
            If dataset is None then none of the data-sets are loaded
        deltatime : float
            the time-difference (in seconds) between the values in a trial of the data-set.
            If this time is different from the one in the data-set, the data-set will be converted in order to be
            consistent.
        short : bool
            whether the data-set should be short or extended during the internal conversions.
        kwargs
            other arguments as described in pandas DataFrame
        """
        if dataset is None and kwargs:
            super(DataFrame, self).__init__(**kwargs)
        else:
            if dataset is None:
                dataset = ['B+S']
            recs = self.load_data(dataset)
            raw_data = {}
            for name in recs.keys():
                genotype, odour, ds = re.findall(r'([\d\w\W]+)-([\w\W]+)-([\w\W]+)', name)[0]
                debug(name, genotype, odour, ds, console=True, verbose=5)
                if genotype not in meta.keys():
                    continue
                if genotype not in raw_data.keys():
                    raw_data[genotype] = meta[genotype].copy()
                if 'M+S' in odour and ds in ['B+S', 'KC']:
                    raw_data[genotype]['CS+'] = odour
                    raw_data[genotype]['cs+traces'] = recs[name]
                elif 'O+S' in odour and ds in ['A+S']:
                    raw_data[genotype]['CS+'] = odour
                    raw_data[genotype]['cs+traces'] = recs[name]
                elif 'M+NS' in odour and ds in ['B+NS']:
                    raw_data[genotype]['CS+'] = odour
                    raw_data[genotype]['cs+traces'] = recs[name]
                else:
                    raw_data[genotype]['CS-'] = odour
                    raw_data[genotype]['cs-traces'] = recs[name]

            ids, genotypes, names, types, trials, odours, conditions, traces = [], [], [], [], [], [], [], []
            for j, name in enumerate(raw_data.keys()):
                for i in range(9):
                    nb_flies = raw_data[name]['cs-traces'].shape[-1]
                    trials.append([i + 1] * nb_flies)
                    odours.append([raw_data[name]["CS-"]] * nb_flies)
                    conditions.append(['CS-'] * nb_flies)
                    types.append([raw_data[name]['type']] * nb_flies)
                    genotypes.append([name] * nb_flies)
                    names.append([raw_data[name]['name']] * nb_flies)
                    traces.append(raw_data[name]['cs-traces'][i])
                    ids.append(j * nb_flies + np.arange(nb_flies) + 1)

                for i in range(8):
                    nb_flies = raw_data[name]['cs+traces'].shape[-1]
                    trials.append([i + 1] * nb_flies)
                    odours.append([raw_data[name]["CS+"]] * nb_flies)
                    conditions.append(['CS+'] * nb_flies)
                    types.append([raw_data[name]['type']] * nb_flies)
                    genotypes.append([name] * nb_flies)
                    names.append([raw_data[name]['name']] * nb_flies)
                    traces.append(raw_data[name]['cs+traces'][i])
                    ids.append(j * nb_flies + np.arange(nb_flies) + 1)

            genotypes = np.concatenate(genotypes)[np.newaxis]
            names = np.concatenate(names)[np.newaxis]
            types = np.concatenate(types)[np.newaxis]
            trials = np.concatenate(trials)[np.newaxis]
            odours = np.concatenate(odours)[np.newaxis]
            conditions = np.concatenate(conditions)[np.newaxis]
            ids = np.concatenate(ids)[np.newaxis]
            traces = np.concatenate(traces, axis=-1)

            keys = ["type", "condition", "name", "genotype", "odour", "trial", "id"] + list(np.linspace(0.2, 20, 100))
            dat = np.concatenate([types, conditions, names, genotypes, odours, trials, ids, traces], axis=0)
            types = ['unicode'] * 5 + [int] * 2 + [float] * 100

            dat_dict = {}
            for key, d, t in zip(keys, dat, types):
                dat_dict[key] = d.astype(t)
            raw_data = pd.DataFrame(dat_dict)

            raw_data.set_index(["type", "condition", "name", "genotype", "odour", "trial", "id"], inplace=True)

            raw_data.columns = pd.MultiIndex(levels=[raw_data.columns.astype(float).values, [False, True]],
                                             codes=[range(100), [0] * 44 + [1] * 5 + [0] * 51],
                                             names=['time', 'shock'])
            super(DataFrame, self).__init__(raw_data.astype(float))
        self.deltatime = deltatime  # seconds
        self.short = short

    def __name2location(self, name):
        """
        '/' = or
        '<' = from
        '>' = to
        :param name:
        :return:
        """

        pedc = False
        calyx = False
        if 'pedc' in name:
            pedc = True
            name = string.replace(name, 'pedc', '')
        if 'calyx' in name:
            calyx = True
            name = string.replace(name, 'calyx', '')

        comps = []
        for comp in re.findall(r'(\W\'{0,1}\d{0,1}\w*)', name):

            cs = []
            for c in comp:
                if c == "'":
                    cs[-1] = cs[-1] + c
                elif c in self.comps.keys():
                    cs.append(c)
                else:
                    cs.append(c)

            if len(cs) > 3:
                for c in cs[2:]:
                    if c.isdigit():
                        cs2 = [cs[0]]
                    if 'cs2' in locals():
                        cs2.append(c)
                if 'cs2' in locals():
                    for c in cs2[1:]:
                        cs.remove(c)
                    if len(cs2) > 1 and cs2[1].isdigit():
                        cs2[1] = int(cs2[1])
            if len(cs) > 1 and cs[1].isdigit():
                cs[1] = int(cs[1])
            comps.append(cs)
            if 'cs2' in locals():
                comps.append(cs2)
        if pedc:
            comps.append(['pedc'])
        if calyx:
            comps.append(['calyx'])

        return comps

    def slice(self, types=None, loc=None, odours=None):

        gens_ret = self.copy()

        if types is not None:
            if not isinstance(types, list):
                types = [types]
            types_ = types[:]

            for type_i in types_:
                if type_i.lower() in ["dan"]:
                    types.remove(type_i)
                    types.append("PAM")
                    types.append("PPL1")
                if type_i.lower() in ["mbon"]:
                    types.remove(type_i)
                    types.append("MBON-glu")
                    types.append("MBON-ach")
                    types.append("MBON-gaba")
                elif type_i.lower() in ["glutamine", "glu"]:
                    types.remove(type_i)
                    types.append("MBON-glu")
                elif type_i.lower() in ["cholimergic", "ach"]:
                    types.remove(type_i)
                    types.append("MBON-ach")
                elif type_i.lower() in ["gaba"]:
                    types.remove(type_i)
                    types.append("MBON-gaba")

            gens = gens_ret.copy()
            gens_ret.clear()
            for type_i in types:
                for genotype in gens.keys():
                    if gens[genotype]['type'].lower() == type_i.lower():
                        gens_ret[genotype] = gens[genotype]

        if odours is not None:
            if not isinstance(odours, list):
                odours = [odours]
            odours_ = odours[:]

            for odour in odours_:
                if "ROI" not in odour:
                    odours.append(odour + "_bigROI")
                    odours.append(odour + "_bROI")
                    odours.append(odour + "_sROI")

            gens = gens_ret.copy()
            gens_ret.clear()
            for odour in odours:
                for genotype in gens.keys():
                    if odour in gens[genotype].keys():
                        gens_ret[genotype] = gens[genotype]

        if loc is not None:
            locs = self.__name2location(loc)

            for loc in locs:
                for l, _ in enumerate(loc):
                    gens = gens_ret.copy()
                    gens_ret.clear()
                    for genotype in gens.keys():
                        for comp in gens[genotype]['loc']:
                            if np.all([loc[ll] in comp for ll in range(l + 1)]):
                                gens_ret[genotype] = gens[genotype]
                                continue

        return DataFrame(**gens_ret)

    def get_dataset(self, neurons=None):
        return _frame2dataset(self.unstacked, ts=int(20. / self.deltatime), names=neurons, short=self.short)

    def get_features(self, neurons=None):
        return _frame2features(self.unstacked, ts=int(20. / self.deltatime), names=neurons, short=self.short)

    def get_hist(self, neurons=None):
        return _features2hist(
            _frame2features(self.unstacked, ts=int(20. / self.deltatime), names=neurons, short=self.short),
            names=neurons
        )

    def get_timeseries(self, neurons=None):
        feats = self.get_features(neurons=None)
        ntypes, nnames = _split_names(neurons, feats)
        print(len(ntypes), ntypes)
        print(len(nnames), nnames)
        print(len(neurons), neurons)

        ts = []
        for tp, nm in zip(ntypes, nnames):
            m = re.match(r"(.+\d)(\w*)(_\w+)*", nm)
            comp, brn, bil = m.group(1), m.group(2), m.group(3)
            try:
                ts.append(feats.T[tp, nm].to_numpy())
            except KeyError:
                try:
                    ts.append(feats.T[tp, comp].to_numpy())
                except KeyError:
                    ts.append(np.zeros(1700))
        print(len(ts))
        ts = pd.DataFrame(np.array(ts), index=neurons)

        return ts

    def plot_neurons(self, neurons=None, show=False, save=None):
        return plot_hist(show=show, **self.get_hist(neurons=neurons), save=save)

    @property
    def unstacked(self):
        return DataFrame.pivot_trials(self)

    @property
    def normalised(self):
        return DataFrame.normalise(self)

    @property
    def dataset6neuron(self):
        return self.get_dataset()

    @staticmethod
    def pivot_trials(df):
        # drop the 'odour' index
        dff = df.reset_index(level='odour', drop=True)

        # reorder the levels of indices in the 'index' axis so that the 'trial' index is last
        dff = dff.reorder_levels(['type', 'name', 'genotype', 'id', 'condition', 'trial'], axis=0)  # type: DataFrame

        # unstack the last level: the last level changes axis for 'index' to 'column'
        dff = dff.unstack([-2, -1])  # type: DataFrame

        # reorder the levels of indices in the 'columns' axis
        dff = dff.reorder_levels(['trial', 'condition', 'time', 'shock'], axis=1)  # type: DataFrame

        # sort the indices in the 'columns' axis
        dff = dff.sort_index(axis=1, level=['trial', 'condition', 'time'], ascending=[True, False, True])

        # sort the indices in the 'index' axis
        dff = dff.sort_index(axis=0, level=['type', 'name', 'genotype', 'id'])  # type: DataFrame

        # drop the last trial (CS+ trial 9) which does not exist
        dff = dff.T[:-100].T

        dff = DataFrame(data=dff, deltatime=df.deltatime, short=df.short)

        return dff

    @staticmethod
    def normalise(df):
        x_max = np.max([df.max(axis=1), -df.min(axis=1)], axis=0)
        return (df.T / (x_max + eps)).T


class DataSet(Data):

    def __init__(self, dataset=None, neuron_names=None, deltatime=0.2, short=False, **kwargs):
        """

        Parameters
        ----------
        dataset : str, list of str
            name(s) of the data-sets to load -- a set of: 'A+S', 'B+S', 'B+NS', 'KC' or 'all'.
            If dataset is None then none of the data-sets are loaded
        neuron_names: str, list of str
            name(s) of the neurons to load
            If it's None, then all the neurons are loaded
        deltatime : float
            the time-difference (in seconds) between the values in a trial of the data-set.
            If this time is different from the one in the data-set, the data-set will be converted in order to be
            consistent.
        short : bool
            whether the data-set should be short or extended during the internal conversions.
        kwargs
            other arguments as described in pandas DataFrame
        """

        if neuron_names is not None:
            neurons = list(zip(*_split_names(neuron_names)))
        else:
            neurons = None
        if dataset is None and kwargs:
            super(DataSet, self).__init__(**kwargs)
        else:
            if dataset is None:
                dataset = ['B+S']
            recs = self.load_data(dataset)
            raw_data = {}
            for name in recs.keys():
                genotype, odour, ds = re.findall(r'([\d\w\W]+)-([\w\W]+)-([\w\W]+)', name)[0]
                # print(genotype, odour, ds)
                debug(name, genotype, odour, ds, verbose=5)
                if genotype not in meta.keys():
                    continue
                if neurons is not None and (meta[genotype]['type'], meta[genotype]['name']) not in neurons:
                    debug("Not included:", meta[genotype]['type'], meta[genotype]['name'], verbose=3, console=True)
                    continue
                if genotype not in raw_data.keys():
                    raw_data[genotype] = meta[genotype].copy()
                if 'M+S' in odour and ds in ['B+S', 'KC']:
                    raw_data[genotype]['CS+'] = odour
                    raw_data[genotype]['cs+traces'] = recs[name]
                elif 'O+S' in odour and ds in ['A+S']:
                    raw_data[genotype]['CS+'] = odour
                    raw_data[genotype]['cs+traces'] = recs[name]
                elif 'M+NS' in odour and ds in ['B+NS']:
                    raw_data[genotype]['CS+'] = odour
                    raw_data[genotype]['cs+traces'] = recs[name]
                else:
                    raw_data[genotype]['CS-'] = odour
                    raw_data[genotype]['cs-traces'] = recs[name]

            nb_trials, nb_steps, nb_flies = 0, 0, 0
            ids, genotypes, names, types, csp, csm, traces = [], [], [], [], [], [], []
            for j, name in enumerate(raw_data.keys()):
                nb_trials, nb_steps, nb_flies = raw_data[name]['cs-traces'].shape
                nb_trials += raw_data[name]['cs+traces'].shape[0] - 1
                traces.append(np.zeros((17 * nb_steps, nb_flies), dtype=float))
                for i in range(9):
                    traces[-1][(2*i)*nb_steps:(2*i+1)*nb_steps] = raw_data[name]['cs-traces'][i]
                for i in range(8):
                    traces[-1][(2*i+1)*nb_steps:(2*i+2)*nb_steps] = raw_data[name]['cs+traces'][i]
                csp.append([raw_data[name]["CS+"]] * nb_flies)
                csm.append([raw_data[name]["CS-"]] * nb_flies)
                types.append([raw_data[name]['type']] * nb_flies)
                genotypes.append([name] * nb_flies)
                names.append([raw_data[name]['name']] * nb_flies)
                ids.append(j * nb_flies + np.arange(nb_flies) + 1)

            genotypes = np.concatenate(genotypes)[np.newaxis]
            names = np.concatenate(names)[np.newaxis]
            types = np.concatenate(types)[np.newaxis]
            csp = np.concatenate(csp)[np.newaxis]
            csm = np.concatenate(csm)[np.newaxis]
            ids = np.concatenate(ids)[np.newaxis]
            traces = np.concatenate(traces, axis=-1)

            keys = ["type", "name", "genotype", "csp", "csm", "id"] +\
                    list(np.linspace(0.2, 20 * nb_trials, nb_steps * nb_trials))
            dat = np.concatenate([types, names, genotypes, csp, csm, ids, traces], axis=0)
            types = ['unicode'] * 5 + [int] + [float] * nb_steps * nb_trials

            dat_dict = {}
            for key, d, t in zip(keys, dat, types):
                dat_dict[key] = d.astype(t)
            raw_data = pd.DataFrame(dat_dict)
            raw_data.set_index(["type", "name", "genotype", "csp", "csm", "id"], inplace=True)

            levels = [range(1, 10), ['CS-', 'CS+'], raw_data.columns.astype(float).values]
            codes = [
                sum([[i // 2] * nb_steps for i in range(nb_trials)], []),
                sum([[i % 2] * nb_steps for i in range(nb_trials)], []),
                range(nb_steps * nb_trials)]
            raw_data.columns = pd.MultiIndex(levels=levels, codes=codes, names=['trial', 'condition', 'time'])
            super(DataSet, self).__init__(raw_data.astype(float))

        self.deltatime = deltatime  # seconds
        self.short = short

    def timeseries(self, neurons, m=50000):
        s = OlfactoryLearning(delta_time=self.deltatime)
        s.build(cs_in=5., cs_out=10., us_in=9., us_out=9.3, trial_duration=20.,
                trial_structure=[
                    {
                        "phase": "pre-training",
                        "CS": ["A", "B"],
                        "US": [None, None]
                    }, {
                        "phase": "training",
                        "CS": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
                        "US": [None, "shock", None, "shock", None, "shock", None, "shock", None, "shock"]
                    }, {
                        "phase": "rest",
                        "CS": ["A"],
                        "US": [None]
                    }, {
                        "phase": "reversal",
                        "CS": ["B", "A", "B", "A"],
                        "US": [None, "shock", None, "shock"]
                    }])
        ntypes, nnames = _split_names(neurons, self, union=False)
        data = []
        for i, tp, nm, nn in zip(range(len(neurons)), ntypes, nnames, neurons):
            try:
                data.append(self.T[tp, nm].T.to_numpy())
            except KeyError:
                data.append(np.zeros((1, self.shape[1]), dtype='float32'))
        y = np.zeros((m, len(data), self.shape[1]))
        for i in range(m):
            y[i] = np.vstack([nn[np.random.randint(nn.shape[0])] for nn in data])

        # create input (CS and US)
        cs = s.cs_values.numpy()
        us = s.us_values.numpy()
        x = np.zeros((m, 3, self.shape[1]))
        x[:, :2] = cs
        x[:, 2:] = us

        return x, y


def _frame2features(df, ts=None, names=None, short=False):
    """

    Parameters
    ----------
    df : DataFrame
        the DataFrame to extract the features from (has to be 'unstacked')
    ts : int
        the number of timesteps to subsample the time sequence (default = 20 / dt)
    names : list of str
        names of the neurons that we want to extract the features from (default = 6-neuron neurons)
    short : bool
        whether we want to create a short or extended dataset

    Returns
    -------
        pd.DataFrame : the dataset as feature vectors

    """
    if ts is None:
        ts = int(20. / df.deltatime)  # 100
    if short is None:
        short = df.short
    ntypes, nnames = _split_names(names, df)
    cond = []
    for group in [[tp, nm] for tp, nm, in zip(ntypes, nnames)]:
        c = np.all([df.index.get_level_values(t) == g for t, g in zip(["type", "name"], group)], axis=0)
        cond.append(c)

    data_i = df.iloc[np.any(cond, axis=0)]  # type: pd.DataFrame
    names = ["time", "trial", "condition"]
    if ts == 4:
        t0 = np.concatenate([np.ones(25), np.zeros(75)] * 17).astype(bool)
        t1 = np.concatenate([np.zeros(25), np.ones(20), np.zeros(55)] * 17).astype(bool)
        t2 = np.concatenate([np.zeros(45), np.ones(5), np.zeros(50)] * 17).astype(bool)
        t3 = np.concatenate([np.zeros(50), np.ones(25), np.zeros(25)] * 17).astype(bool)

        data_t0 = data_i.T.iloc[t0].groupby(["trial", "condition"], axis=0).mean().T
        data_t1 = data_i.T.iloc[t1].groupby(["trial", "condition"], axis=0).mean().T
        data_t2 = data_i.T.iloc[t2].groupby(["trial", "condition"], axis=0).mean().T
        data_t3 = data_i.T.iloc[t3].groupby(["trial", "condition"], axis=0).mean().T
        data = [data_t0, data_t1, data_t2, data_t3]
        keys = ["pre-odour", "odour", "shock", "post-odour"]
    elif ts <= 100:
        data = []
        for t in range(ts):
            t0 = int(np.floor(t * 100. / ts))  # data before timestep
            t1 = int(np.ceil(100. / ts))  # data during timestep
            t2 = 100 - t1 - t0  # data after timestep
            tt = np.concatenate([np.zeros(t0), np.ones(t1), np.zeros(t2)] * 17).astype(bool)
            data.append(data_i.T.iloc[tt].groupby(["trial", "condition"], axis=0).mean().T)

        keys = (np.linspace(0, 20, ts, endpoint=False) + 20. / ts).tolist()
    else:
        keys = (np.linspace(0, 20, ts, endpoint=False) + 20. / ts).tolist()
        data = [data_i]

    dff = pd.concat(data, axis=1, names=names, keys=keys).reorder_levels(
        ["trial", "condition", "time"], axis=1).sort_index(
        axis=1, level=["trial", "condition"], ascending=[True, False])  # type: pd.DataFrame

    return dff.groupby(["type", "name"]).mean() if short else dff


def _frame2dataset(df, ts=None, names=None, short=None):
    if ts is None:
        ts = int(20. / df.deltatime)
    if short is None:
        short = df.short
    ntypes, nnames = _split_names(names, df)

    debug(ntypes, console=True, verbose=4)
    debug(nnames, console=True, verbose=4)

    df = _frame2features(df, ts=ts, short=short)
    debug(df, verbose=3)

    dff = None  # pd.DataFrame
    for t, n in zip(ntypes, nnames):
        try:
            dfff = df.T[t, n].T  # type: pd.DataFrame
        except KeyError as e:
            err(e)
            continue
        if short:
            dfff = dfff.to_frame().T
        dfff = dfff.T.set_index(np.repeat(t, dfff.shape[1]), append=True).T
        dfff = dfff.T.set_index(np.repeat(n, dfff.shape[1]), append=True).T
        debug(dfff, verbose=3)
        dfff = dfff.reorder_levels([3, 4, 0, 1, 2], axis=1)
        dfff.columns.set_names([u'type', u'name', u'trial', u'condition', u'time'], inplace=True)
        dfff['key'] = np.ones(dfff.shape[0])
        if dff is None:
            dff = dfff
        else:
            dff = pd.merge(dff, dfff, on=["key"])
    dff.drop(columns="key", inplace=True)
    return dff


def _features2hist(df, names=None):
    ntypes, nnames = _split_names(names, df)
    if names is None:
        names = [u"%s-%s" % (nt if '-' not in nt else 'MBON', nn) for nt, nn in zip(ntypes, nnames)]

    nnumbs = []
    ki, mi, di = 0, 0, 0
    for nt in ntypes:
        if 'MBON' in nt:
            mi += 1
            nnumbs.append('m%d' % mi)
        elif 'KC' in nt:
            ki += 1
            nnumbs.append('k%d' % ki)
        else:
            di += 1
            nnumbs.append('d%d' % di)
    df_avg = df.groupby(["type", "name"]).mean()
    ts = int(df.shape[1] / 17.)

    hist = {
        "names": {},
        "tau": 20. / ts
    }

    for n, nt, nn, name in zip(nnumbs, ntypes, nnames, names):
        try:
            print(n, nt, nn, name)
            nv = df_avg.T[nt, nn].T.to_numpy(dtype=float)
        except KeyError as e:
            err(e)
            continue
        hist["names"][n] = name
        hist[n] = np.zeros((17, 2))
        i_us = int(9. * ts / 20.)  # the time step id where the US appears
        hist[n][0::2, 0] = nv[i_us::ts][0::2]
        hist[n][1::2, 0] = (nv[i_us::ts][0:-2:2] + nv[i_us::ts][2::2]) / 2.
        hist[n][0, 1], hist[n][-1, 1] = nv[i_us::ts][1], nv[i_us::ts][-2]
        hist[n][1::2, 1] = nv[i_us::ts][1::2]
        hist[n][2:-2:2, 1] = (nv[i_us::ts][1:-2:2] + nv[i_us::ts][3::2]) / 2.

        hist[n + "s"] = np.zeros((17, ts, 2))
        hist[n + "s"][0::2, :, 0] = nv.reshape((17, ts))[0::2]
        hist[n + "s"][1::2, :, 0] = nv.reshape((17, ts))[0:-1:2][:, 0][..., np.newaxis]
        hist[n + "s"][1::2, :, 1] = nv.reshape((17, ts))[1::2]
        hist[n + "s"][0, :, 1] = nv.reshape((17, ts))[3, 0]
        hist[n + "s"][2::2, :, 1] = nv.reshape((17, ts))[1::2][:, 0][..., np.newaxis]
        hist[n + "s"] = hist[n + "s"].reshape((-1, 2))

    return hist


def _split_names(names, df=None, union=True):
    if names is None:
        names = [u"PPL1-γ2α'1", u"PPL1-γ1pedc", u"PPL1-α3", u"PPL1-α'3", u"PPL1-α'2α2",
                 u"PAM-γ5", u"PAM-γ4<γ1γ2", u"PAM-γ3", u"PAM-β2β'2a", u"PAM-β2", u"PAM-β1pedc", u"PAM-β1", u"PAM-β'2p",
                 u"PAM-β'2m", u"PAM-β'2a", u"PAM-β'1", u"PAM-α1",
                 u"MBON-γ5β'2a", u"MBON-γ4>γ1γ2", u"MBON-β2β'2a", u"MBON-β1>α", u"MBON-β'2mp", u"MBON-α1",
                 u"MBON-γ3/γ3β'1", u"MBON-γ1pedc", u"MBON-β'1", u"MBON-γ2α'1", u"MBON-α3", u"MBON-α2sc", u"MBON-α2p3p",
                 u"MBON-α'3", u"MBON-α'2", u"MBON-α'1"]
        # names = [u"PPL1-\u03b31pedc", u"PAM-\u03b2'2m", u"PAM-\u03b2'2a",
        #          u"MBON-\u03b31pedc", u"MBON-\u03b2'2mp", u"MBON-\u03b35\u03b2'2a"]
    _mbon_types = ["Glu", "GABA", "ACh", "ND"]
    ntypes, nnames = [], []
    for name in names:
        m = re.match(r"(.+)-(.+)", name)
        t = m.group(1)
        n = m.group(2)
        if 'MBON' in t and df is not None:
            is_added = False
            for mt in _mbon_types:
                tp = u"%s-%s" % (t, mt)
                b = [[g == v for v in df.index.get_level_values(t)]
                     for t, g in zip(["type", "name"], [tp, n])]
                # print(len(b), len(b[0]))
                c = np.all(b, axis=0)
                if c.sum() > 0 and not is_added:
                    ntypes.append(tp)
                    nnames.append(n)
                    is_added = True
            if not is_added:
                ntypes.append(t)
                nnames.append(n)
        elif 'MBON' in t:
            if union:
                ntypes.append('MBON-Glu')
                ntypes.append('MBON-GABA')
                ntypes.append('MBON-ACh')
                nnames.append(n)
                nnames.append(n)
            else:
                ntypes.append(t)
            nnames.append(n)
        else:
            ntypes.append(t)
            nnames.append(n)
    return ntypes, nnames


if __name__ == "__main__":
    from utils import clear_logs
    from datetime import datetime

    set_verbose(2)

    neuron_names = [
        u"PPL1-γ2α'1",
        u"PPL1-γ1pedc", u"PPL1-α3", u"PPL1-α'3", u"PPL1-α'2α2",

        u"PAM-γ5", u"PAM-γ4<γ1γ2", u"PAM-γ3", u"PAM-β2β'2a", u"PAM-β2", u"PAM-β1pedc", u"PAM-β1", u"PAM-β'2p",
        u"PAM-β'2m", u"PAM-β'2a", u"PAM-β'1", u"PAM-α1",

        u"MBON-γ5β'2a", u"MBON-γ4>γ1γ2", u"MBON-β2β'2a", u"MBON-β1>α",
        u"MBON-β'2mp", u"MBON-α1",

        u"MBON-γ3/γ3β'1",
        u"MBON-γ1pedc", u"MBON-β'1",

        u"MBON-γ2α'1", u"MBON-α3", u"MBON-α2sc", u"MBON-α2p3p", u"MBON-α'3", u"MBON-α'2", u"MBON-α'1",

        # u"MBON-calyx",

        u"KC-γm", u"KC-γd", u"KC-α/β", u"KC-α'/β'",
    ]
    print(DataSet('B+S', neuron_names))
    df = DataFrame('B+S', deltatime=0.2, short=False)
    # pd.set_option('display.max_rows', df.shape[0] + 1)
    # info(df, console=True)
    # info(df.get_features(neurons=neuron_names), console=True)
    # # info(df.dataset6neuron, console=True)
    fig = df.plot_neurons(neurons=neuron_names, show=True, save='mb-neural-traces.png')

    # df = DataSet('B+S', neuron_names=neuron_names, deltatime=0.2, short=True)
    # x, y = df.timeseries(neuron_names, m=2)
    # info(df, console=True)

    # clear_logs('20200107-0000', datetime.strftime(datetime.now(), '%Y%m%d-%H%M'))
