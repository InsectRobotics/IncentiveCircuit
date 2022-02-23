__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright 2021, School of Informatics, the University of Edinburgh"
__licence__ = "MIT"
__version__ = "1.1-alpha"
__maintainer__ = "Evripidis Gkanias"
__email__ = "ev.gkanias@ed.ac.uk"
__status__ = "Production"

import numpy as np
import pandas as pd

import os

# the directory of the file
__dir__ = os.path.dirname(os.path.abspath(__file__))
# the directory of the data
__data_dir__ = os.path.realpath(os.path.join(__dir__, "data", "handler2019"))

conditions = ['-6s ISI', '-1.2s ISI', '-0.6s ISI', '0s ISI', '0.5s ISI', '6s ISI']
stim = {
    "cAMP": {
        '-6s ISI': [4, 8],
        '-1.2s ISI': [8.8, 12.8],
        '-0.6s ISI': [9.4, 13.4],
        '0s ISI': [10, 14],
        '0.5s ISI': [10.5, 14.5],
        '6s ISI': [16, 20]
    },
    "ERGCaMP": {
        '-6s ISI': [-1, 7],
        # '-6s ISI': [6, 7],
        '-1.2s ISI': [-1, 7],
        # '-1.2s ISI': [6, 7],
        '-0.6s ISI': [-1, 7],
        # '-0.6s ISI': [6, 7],
        '0s ISI': [-1, 7],
        # '0s ISI': [6, 7],
        '0.5s ISI': [-1, 7],
        # '0.5s ISI': [6, 7],
        '6s ISI': [-1, 7]
        # '6s ISI': [6, 7]
    }
}
camp_time_adjust = -10
erca_time_adjust = -6


def run_case(us_on, tau_kc=100/3, tau_short=58, tau_long=100):

    # tau_long = 1/(1-gamma_d2s)
    # tau_short = 1/(1-gamma_d1s)

    # calculate the discount factor from the time-constants
    gamma_kc = 1 - 1 / tau_kc
    gamma_d1 = 1 - 1 / tau_short
    gamma_d2 = 1 - 1 / tau_long

    W = 1.
    time, css, uss, k, d1, d2, m, w = [], [], [], [], [], [], [], []
    for t, cs, us in handler_routine(us_on=us_on):
        time.append(t)
        css.append(cs)
        uss.append(us)
        if len(k) > 0:
            K = (1 - gamma_kc) * cs + gamma_kc * k[-1]
        else:
            K = cs
        k.append(np.clip(K, 0, 2))
        m.append(np.clip(k[-1] * np.maximum(W, 0), 0, 2))
        if len(d1) > 0 and len(d2) > 0:
            D1 = (1 - gamma_d1) * (us - 1. * m[-2]) + gamma_d1 * d1[-1]
            D2 = (1 - gamma_d2) * (us - 1. * m[-2]) + gamma_d2 * d2[-1]
        else:
            D1 = us
            D2 = us
        d1.append(np.clip(D1, 0, 2))
        d2.append(np.clip(D2, 0, 2))
        W += dopaminergic_plasticity_rule(k[-1], d1[-1], d2[-1], W, w_rest=1.)
        w.append(np.maximum(W, 0))

    # the potentiating effects
    d_up = np.maximum(np.array(d1) - np.array(d2), np.finfo(float).eps)
    # the depressing effects
    d_down = np.minimum(np.array(d1) - np.array(d2), -np.finfo(float).eps)

    dR1 = -d_up * (np.array(k) - 1) - (d_up - d_down) * np.array(w)
    dR2 = d_down * (np.array(k) - 1)

    return time, css, uss, k, d1, d2, m, w, dR1, dR2


def handler_routine(us_on=0., us_duration=.6, cs_on=0., cs_duration=.5, nb_samples=1001):
    time_range = [-7, 8]

    for t in np.linspace(time_range[0], time_range[1], nb_samples, endpoint=True):
        cs, us = 0., 0.
        if cs_on <= t < cs_on+cs_duration:
            cs = 1.
        if us_on <= t < us_on+us_duration:
            us = 1.
        yield t, cs, us


# dw/dt = D (k + w - w_rest)
# dw/dt = D_1 k + D_2 (k + w - w_rest)
# D_1 = D^-  (short trace)
# D_2 = D^+  (longer trace)
def dopaminergic_plasticity_rule(k, D_1, D_2, w, w_rest, passive_effect=1e-00):
    """
    The dopaminergic plasticity rule, modified to use the D_1 and D_2 components separately.

    Parameters
    ----------
    k : np.ndarray[float]
        the KC responses.
    D_1 : np.ndarray[float]
        the short component of dopamine.
    D_2 : np.ndarray[float]
        the long component of dopamine.
    w : np.ndarray[float]
        the current KC>MBON synapses
    w_rest : np.ndarray[float], float
        the resting value of the KC>MBON synaptic weights
    passive_effect : float, optional
        weighting of the passive effects of the learning rule. Default us 1

    Returns
    -------
    np.ndarray[float]
        the synaptic change ΔW
    """
    return (D_2 - D_1) * np.maximum(k + w - w_rest, passive_effect * (k + w - w_rest))


def read_data(file_path, timecourse='cAMP', verbose=False):
    excel = pd.read_excel(file_path, engine="openpyxl", sheet_name=f"Fig5D_{timecourse} timecourse",
                          header=[0, 1], nrows=303, skiprows=[1],
                          index_col=None)  # , usecols="B:H,K:P,S:X,AA:AF,AI:AN,AQ:AV")

    try:
        excel.set_index(np.array(excel["-6s ISI", "prep"]), inplace=True)
    except KeyError:
        try:
            excel.set_index(np.array(excel["-6s ISI", "compartment"]), inplace=True)
        except KeyError:
            if verbose:
                print("Could not find the keys: 'prep', 'compartment'")

    keys = {
        "prep": 1,
        "compartment": 1,
        "Unnamed: 0_level_0": 0,
        "Unnamed: 1_level_0": 0,
        "Unnamed: 8_level_1": 1,
        "Unnamed: 9_level_1": 1,
        "Unnamed: 16_level_1": 1,
        "Unnamed: 18_level_1": 1,
        "Unnamed: 24_level_1": 1,
        "Unnamed: 27_level_1": 1,
        "Unnamed: 32_level_1": 1,
        "Unnamed: 36_level_1": 1,
        "Unnamed: 40_level_1": 1,
        "Unnamed: 45_level_1": 1
    }

    for key, level in keys.items():
        try:
            excel.drop(columns=[key], level=level, inplace=True)
        except KeyError:
            if verbose:
                print(f"Could not find the key: '{key}', in level: '{level}'")

    return excel


def load_data(timecourse):
    file_path = os.path.join(__data_dir__, "Handler2019_Fig2Fig5_Data.xlsx")
    return read_data(file_path, timecourse)


def load_means(timecourse):
    data = load_data(timecourse)
    time = np.array(data.index)

    means = []
    for condition in conditions:
        s = stim[timecourse][condition]
        means.append(np.nanmean(data[condition], axis=1))

    return time, means


def load_statistics():
    camp_data = load_data('cAMP')
    erca_data = load_data('ERGCaMP')

    camp_means = {}
    erca_means = {}
    for condition in conditions:
        s = stim['cAMP'][condition]
        camp_means[condition] = np.nanmean(camp_data[condition][s[0]:s[1]], axis=0)
        s = stim['ERGCaMP'][condition]
        erca_means[condition] = np.nanmean(erca_data[condition][s[0]:s[1]], axis=0)

    camp_means = pd.DataFrame(camp_means).T
    erca_means = -pd.DataFrame(erca_means).T

    camp_means_norm = (camp_means - camp_means.min(axis=0)) / (camp_means.max(axis=0) - camp_means.min(axis=0))
    erca_means_norm = (erca_means - erca_means.min(axis=0)) / (erca_means.max(axis=0) - erca_means.min(axis=0))

    # average across the different samples
    camp_ave = camp_means_norm.T.mean(axis=0)
    erca_ave = erca_means_norm.T.mean(axis=0)

    # calculate the standard error
    camp_sem = camp_means_norm.T.std(axis=0) / np.sqrt(camp_means_norm.T.shape[0])
    erca_sem = erca_means_norm.T.std(axis=0) / np.sqrt(erca_means_norm.T.shape[0])

    # calculate the plasticity mean and standard error
    plas_ave = erca_ave - camp_ave
    plas_sem = np.sqrt(np.square(erca_sem) + np.square(camp_sem))

    result = {
        "cAMP": {
            "mean": np.array(camp_ave),
            "sem": np.array(camp_sem)
        },
        "ERGCaMP": {
            "mean": np.array(erca_ave),
            "sem": np.array(erca_sem)
        },
        "plasticity": {
            "mean": np.array(plas_ave),
            "sem": np.array(plas_sem)
        }
    }

    return result
