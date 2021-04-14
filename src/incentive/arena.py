"""
Package that contains the arena simulation with two odour distributions. The arena contains flies that are moving freely
while experiencing the odours and reinforcement, learning to approach or avoid the odours depending on the condition.
"""

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2021, Insect Robotics Group," \
                "Institude of Perception, Action and Behaviour," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "GPLv3+"
__version__ = "v1.0.0-alpha"
__maintainer__ = "Evripidis Gkanias"

from .circuit import IncentiveCircuit

import numpy as np
import pandas as pd
import re
import os

__dir__ = os.path.dirname(os.path.abspath(__file__))
"""the directory of the file"""
__data_dir__ = os.path.realpath(os.path.join(__dir__, "data", "arena"))
"""the directory of the data"""


class FruitFly(object):
    a_source = .6+0j
    a_sigma = .2
    b_source = -.6+0j
    b_sigma = .3

    def __init__(self, nb_kcs=10, nb_kc_odour_a=5, nb_kc_odour_b=5, nb_steps=1000, nb_in_trial=1, learning_rule="dlr",
                 gain=.04, rng=np.random.RandomState()):
        """
        Simulation parameters and methods for the fly running in an arena with two odour distributions. The incentive
        circuit is used in order to find the most attractive or aversive direction and move towards or away from it.

        Parameters
        ----------
        nb_kcs: int, optional
            the number of KCs of the incentive circuit. Default is 10.
        nb_kc_odour_a: int, optional
            the number of KCs associated with odour A. Default is 5.
        nb_kc_odour_b: int, optional
            the number of KCs associated with odour B. Default is 5.
        nb_steps: int, optional
            the number of time-steps that the simulation will run. Default is 1000.
        nb_in_trial: int, optional
            the number of in-trial time-steps for the incentive circuit processing. Default is 1.
        learning_rule: str, optional
            the learning rule that the incentive circuit will use in order to update its weights. Defult is "dlr".
        gain: float, optional
            controls the walking speed of the fly. Default is 0.04 m.
        rng: optional
            the random generator
        """
        self.mb = IncentiveCircuit(
            learning_rule=learning_rule, nb_apl=0, pn2kc_init="default", nb_timesteps=nb_in_trial, nb_trials=nb_steps,
            nb_kc=nb_kcs, nb_kc_odour_1=nb_kc_odour_a, nb_kc_odour_2=nb_kc_odour_b, has_real_names=False,
            has_sm=True, has_rm=True, has_ltm=True, has_rrm=True, has_rfm=True, has_mam=True)

        self.xy = np.zeros(nb_steps, dtype=complex)
        self.p_a = np.zeros(nb_steps, dtype=float)
        self.p_b = np.zeros(nb_steps, dtype=float)
        self.turn = np.zeros(nb_steps, dtype=float)
        self.gain = gain
        self.rng = rng

    def __call__(self, *args, **kwargs):
        """
        Runs the arena_routine with the set parameters.
        """
        punishment = kwargs.get("punishment", True)
        reward = kwargs.get("reward", not punishment)
        r_start = kwargs.get("r_start", .2)
        r_end = kwargs.get("r_end", .5)
        noise = kwargs.get("noise", .1)
        s = kwargs.get("susceptible", 1.)
        r = kwargs.get("restrained", 1.)
        m = kwargs.get("ltm", 1.)
        a = kwargs.get("only_a", False)
        b = kwargs.get("only_b", False)
        routine = arena_routine(
            self, noise=noise, r_start=r_start, r_end=r_end, reward=reward, punishment=punishment,
            susceptible=s, restrained=r, ltm=m, only_a=a, only_b=b)
        return self.mb(routine=routine)

    @property
    def t(self):
        """
        The current time-step.
        """
        return self.mb._t


def arena_routine(agent, noise=0.1, r_start=.2, r_end=.5, reward=False, punishment=True,
                  susceptible=1., restrained=1., ltm=1., only_a=False, only_b=False):
    """
    The arena routine creates a routine for the FruitFly simulation for a fixed number of time-steps. Returns a
    generator that computes the trial, time-step, CS and US that will be used as input to the MB.

    Parameters
    ----------
    agent: FruitFly
        the fly simulation.
    noise: float, optional
        the added noise parameter. Default is 0.1.
    r_start: float, optional
        the percentage of the time-steps that will pass before the reinforcement will be introduced. Default is 0.2.
    r_end: float, optional
        the percentage of the time-steps that will pass before the reinforcement stops being presented. Default is 0.5.
    reward: bool, optional
        if the reinforcement is rewarding or not. Default is False.
    punishment: bool, optional
        if the reinforcement is punishing or not. Default is True.
    susceptible: float, optional
        the percentage of influence from the susceptible MBONs. Default is 1.
    restrained: float, optional
        the percentage of influence from the restrained MBONs. Default is 1.
    ltm: float, optional
        the percentage of influence from the long-term memory MBONs. Default is 1.
    only_a: bool, optional
        if the reinforcement is applied only when odour A is more intense.
    only_b: bool, optional
        if the reinforcement is applied only when odour B is more intense.

    Yields
    -------
    trial: int
        the trial number.
    timestep: int
        the time-step number.
    cs: np.ndarray
        the odour mixture (conditional stimulus, CS) based on the position of the agent.
    us: np.ndarray
        the reinforcement mixture (unconditional stimulus, US) based on the position of the agent and the time.
    """
    mb_model = agent.mb
    mb_model._t = 0
    mb_model.routine_name = "arena"
    a_odour_source = agent.a_source
    b_odour_source = agent.b_source
    a_sigma = agent.a_sigma
    b_sigma = agent.b_sigma

    for trial in range(1, mb_model.nb_trials):
        if mb_model._t >= mb_model.nb_trials * mb_model.nb_timesteps:
            break
        p_a = gaussian_p(agent.xy[mb_model._t], a_odour_source, a_sigma)
        p_b = gaussian_p(agent.xy[mb_model._t], b_odour_source, b_sigma)
        p_a = np.clip(p_a / (p_a + p_b) + noise * agent.rng.randn(), 0, 1)
        p_b = 1 - p_a
        agent.p_a[agent.t] = p_a
        agent.p_b[agent.t] = p_b

        trial_ = mb_model._t // mb_model.nb_timesteps

        # create mixture of odours
        cs = p_a * mb_model.csa + p_b * mb_model.csb
        # create reinforcement
        us = np.zeros(mb_model.us_dims, dtype=float)
        if r_start * mb_model.nb_trials < trial_ <= r_end * mb_model.nb_trials:
            w = float((only_b and (p_b >= p_a)) or (only_a and (p_a >= p_b)) or ((not only_a) and (not only_b)))
            us[1] = float(punishment) * w
            us[0] = float(reward) * w

        for timestep in range(mb_model.nb_timesteps):
            yield trial, timestep, cs, us
            mb_model._t += 1

        t = mb_model._t
        s_at, s_av, r_at, r_av, m_at, m_av = mb_model._v[t, 6:]

        s = s_at - s_av
        r = r_at - r_av
        m = m_at - m_av

        sw, rw, mw = float(susceptible), float(restrained), float(ltm)
        attraction = (sw * s + rw * r + mw * m) / (sw + rw + mw)
        if p_a > p_b:
            direction = a_odour_source - agent.xy[t-1]
        else:
            direction = b_odour_source - agent.xy[t-1]
        direction /= np.maximum(np.absolute(direction), np.finfo(float).eps)

        if t < 2:
            vel = 0+0j
        else:
            vel = agent.xy[t-1] - agent.xy[t-2]
        rho = attraction * direction

        vel += rho + agent.rng.randn() * .1 + agent.rng.randn() * .1j
        z = np.maximum(np.absolute(vel), np.finfo(float).eps)
        vel = agent.gain * vel / z

        agent.xy[t] = agent.xy[t-1] + vel
        agent.xy[t] = agent.xy[t] / np.maximum(np.absolute(agent.xy[t]), 1)


def load_arena_stats(file_names, prediction_error=False):
    """
    Creates a DataFrame that contains the stats of experiments in the arena.

    - susceptible: bool
    - restrained: bool
    - long-term memory: bool
    - reinforcement: {"punishment", "reward"}
    - paired odour: {"A", "B", "A+B", "AB"}
    - phase: {"pre", "learn", "post"}
    - angle: float - global orientation in degrees
    - dist_A: float - distance from odour A in meters
    - dist_B: float - distance from odour B in meters
    - ang_A: float - angle from odour A in degrees
    - ang_B: float - angle from odour B in degrees
    
    Parameters
    ----------
    file_names: str
        the names of the files used to calculate the statistics.
    prediction_error: bool
        if the prediction error was used as the learning rule when creating the files.

    Returns
    -------
    data: pd.DataFrame
        a DataFrame of size N x C, where N is the number of files x 3 and C is the number of features calculated
    """

    d_names = ["susceptible", "restrained", "long-term memory", "reinforcement",
               "paired odour", "phase", "angle", "dist_A", "dist_B", "ang_A", "ang_B"]
    d_raw = [[], [], [], [], [], [], [], [], [], [], []]

    for fname in file_names:
        if prediction_error:
            pattern = r'rw-arena-([\w]+)-(s{0,1})(r{0,1})(m{0,1})(a{0,1})(b{0,1})'
        else:
            pattern = r'arena-([\w]+)-(s{0,1})(r{0,1})(m{0,1})(a{0,1})(b{0,1})'
        details = re.findall(pattern, fname)
        if len(details) < 1:
            continue
        punishment = 'quinine' in details[0]
        susceptible = 's' in details[0]
        restrained = 'r' in details[0]
        ltm = 'm' in details[0]
        only_a = 'a' in details[0]
        only_b = 'b' in details[0]

        data = np.load(os.path.join(__data_dir__, fname))["data"]

        nb_flies, nb_time_steps = data.shape

        e_pre, s_post = int(.2 * nb_time_steps), int(.5 * nb_time_steps)

        d_raw[0].extend([susceptible] * 3 * nb_flies)
        d_raw[1].extend([restrained] * 3 * nb_flies)
        d_raw[2].extend([ltm] * 3 * nb_flies)
        d_raw[3].extend(["punishment" if punishment else "reward"] * 3 * nb_flies)
        d_raw[4].extend([("A+B" if only_b else "A") if only_a else ("B" if only_b else "AB")] * 3 * nb_flies)
        d_raw[5].extend(["pre"] * nb_flies)
        d_raw[5].extend(["learn"] * nb_flies)
        d_raw[5].extend(["post"] * nb_flies)
        d_raw[6].extend(np.angle(data[:, e_pre-1]))
        d_raw[6].extend(np.angle(data[:, s_post-1]))
        d_raw[6].extend(np.angle(data[:, -1]))
        d_raw[7].extend(np.absolute(data[:, e_pre-1] - FruitFly.a_source))
        d_raw[7].extend(np.absolute(data[:, s_post-1] - FruitFly.a_source))
        d_raw[7].extend(np.absolute(data[:, -1] - FruitFly.a_source))
        d_raw[8].extend(np.absolute(data[:, e_pre-1] - FruitFly.b_source))
        d_raw[8].extend(np.absolute(data[:, s_post-1] - FruitFly.b_source))
        d_raw[8].extend(np.absolute(data[:, -1] - FruitFly.b_source))
        d_raw[9].extend(np.angle(data[:, e_pre-1] - FruitFly.a_source))
        d_raw[9].extend(np.angle(data[:, s_post-1] - FruitFly.a_source))
        d_raw[9].extend(np.angle(data[:, -1] - FruitFly.a_source))
        d_raw[10].extend(np.angle(data[:, e_pre-1] - FruitFly.b_source))
        d_raw[10].extend(np.angle(data[:, s_post-1] - FruitFly.b_source))
        d_raw[10].extend(np.angle(data[:, -1] - FruitFly.b_source))
    d_raw = np.array(d_raw)
    df = pd.DataFrame(d_raw, index=d_names).T
    df["angle"] = np.rad2deg(np.array(df["angle"], dtype=float))
    df["dist_A"] = np.array(df["dist_A"], dtype=float)
    df["dist_B"] = np.array(df["dist_B"], dtype=float)
    df["ang_A"] = np.rad2deg(np.array(df["ang_A"], dtype=float))
    df["ang_B"] = np.rad2deg(np.array(df["ang_B"], dtype=float))
    df["susceptible"] = np.array(df["susceptible"] == "True", dtype=bool)
    df["restrained"] = np.array(df["restrained"] == "True", dtype=bool)
    df["long-term memory"] = np.array(df["long-term memory"] == "True", dtype=bool)

    return df


def gaussian_p(pos, mean, sigma):
    """
    Calculates the probability of the given 2D positions, based on the mean and variance of a Gaussian distribution.

    Parameters
    ----------
    pos: np.ndarray[complex] | np.ndarray[float]
        the 2D positions of the samples
    mean: complex | np.ndarray[complex] | np.ndarray[float]
        the 2D mean of the Gaussian distribution
    sigma: float
        the 1D variance of the Gaussian distribution

    Returns
    -------
    p: np.ndarray[float]
        the probability density for each sample
    """
    if pos.dtype == complex:
        pos = np.array([pos.real, pos.imag]).T
    if isinstance(mean, complex):
        mean = np.array([mean.real, mean.imag]).T
    pos, mean = np.array(pos), np.array(mean)

    return 1. / (np.sqrt(2 * np.pi) * sigma) * np.exp(-np.sum(np.square(pos - mean), axis=-1) / (2 * np.square(sigma)))
