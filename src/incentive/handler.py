__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright 2021, School of Informatics, the University of Edinburgh"
__licence__ = "MIT"
__version__ = "1.1-alpha"
__maintainer__ = "Evripidis Gkanias"
__email__ = "ev.gkanias@ed.ac.uk"
__status__ = "Production"

import numpy as np


def run_case(us_on, gamma_k=.97, gamma_d1=.98, gamma_d2=.99):

    W = 1.
    time, css, uss, k, d1, d2, m, w = [], [], [], [], [], [], [], []
    for t, cs, us in handler_routine(us_on=us_on):
        time.append(t)
        css.append(cs)
        uss.append(us)
        if len(k) > 0:
            K = (1 - gamma_k) * cs + gamma_k * k[-1]
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

    dR1 = (np.maximum(np.array(d1) - np.array(d2), 0) * (np.array(k) - 1) +
           (np.array(d1) - np.array(d2)) * np.array(w))
    dR2 = np.minimum(np.array(d1) - np.array(d2), 0) * (np.array(k) - 1)

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
def dopaminergic_plasticity_rule(k, D_1, D_2, w, w_rest):
    return (D_2 - D_1) * (k + w - w_rest)
