"""
Package that contains the arena simulation with two odour distributions. The arena contains flies that are moving freely
while experiencing the odours and reinforcement, learning to approach or avoid the odours depending on the condition.
"""

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2021, Insect Robotics Group," \
                "Institute of Perception, Action and Behaviour," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "GPLv3+"
__version__ = "v1.1-alpha"
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
    a_sigma = .15
    b_source = -.6+0j
    b_sigma = .2
    r_radius = .4
    i_threshold = .2

    def __init__(self, nb_kcs=10, nb_kc_odour_a=None, nb_kc_odour_b=None, nb_steps=1000, nb_in_trial=1,
                 learning_rule="dlr", nb_active_kcs=2, gain=.02, rng=np.random.RandomState(2021)):
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
            the learning rule that the incentive circuit will use in order to update its weights. Default is "dlr".
        nb_active_kcs : int
            the number of active KCs at all times. Default is 2.
        gain: float, optional
            controls the walking speed of the fly. Default is 0.02 m.
        rng: optional
            the random generator
        """
        self.mb = IncentiveCircuit(
            learning_rule=learning_rule, nb_apl=0, nb_timesteps=nb_in_trial, nb_trials=nb_steps,
            nb_active_kcs=nb_active_kcs, nb_kc=nb_kcs, nb_kc_odour_1=nb_kc_odour_a, nb_kc_odour_2=nb_kc_odour_b,
            has_real_names=False, has_sm=True, has_rm=True, has_ltm=True, has_rrm=True, has_rfm=True, has_mam=True)
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

        self.mb.w_k2m[0] = self.mb.w_k2m[-2].copy()
        routine = arena_routine(
            self, noise=noise, r_start=r_start, r_end=r_end, reward=reward, punishment=punishment,
            susceptible=s, restrained=r, ltm=m, only_a=a, only_b=b, reset_position=False)
        return self.mb(routine=routine)

    @property
    def t(self):
        """
        The current time-step.
        """
        return self.mb._t


def arena_routine(agent, noise=0.1, r_start=.2, r_end=.5, reward=False, punishment=True,
                  susceptible=1., restrained=1., ltm=1., only_a=False, only_b=False,
                  reset_position=False):
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
    reset_position : bool, optional
        replaces the agent to the centre of teh arena at the beginning of each phase. Default is False

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
    rein_rad = agent.r_radius

    for trial in range(1, mb_model.nb_trials):
        if mb_model._t >= mb_model.nb_trials * mb_model.nb_timesteps:
            break
        i_a = gaussian_p(agent.xy[agent.t], a_odour_source, a_sigma)  # the odour A intensity
        i_b = gaussian_p(agent.xy[agent.t], b_odour_source, b_sigma)  # the odour B intensity
        p_a = np.clip(i_a / (i_a + i_b), 0, 1)  # the probability of detecting odour A
        p_b = 1 - p_a  # the probability of detecting odour B
        agent.p_a[agent.t] = p_a
        agent.p_b[agent.t] = p_b

        trial_ = mb_model._t // mb_model.nb_timesteps  # the trial number

        # create odour identity
        # csa = float(agent.rng.rand() <= p_a) * mb_model.csa  # detect odour A
        # csb = float(agent.rng.rand() <= p_b) * mb_model.csb  # detect odour B
        csa = float(i_a > FruitFly.i_threshold) * mb_model.csa  # detect odour A
        csb = float(i_b > FruitFly.i_threshold) * mb_model.csb  # detect odour B
        cs = csa + csb

        # create reinforcement
        us = np.zeros(mb_model.us_dims, dtype=float)
        if r_start * mb_model.nb_trials < trial_ <= r_end * mb_model.nb_trials:
            # w = float((only_b and (p_b >= p_a)) or (only_a and (p_a >= p_b)) or (not only_a) and (not only_b))
            d_a = np.absolute(a_odour_source - agent.xy[mb_model._t])
            d_b = np.absolute(b_odour_source - agent.xy[mb_model._t])
            w = float((only_a and (d_a < rein_rad)) or  # reinforcement is close to odour A
                      (only_b and (d_b < rein_rad)) or  # reinforcement is close to odour B
                      (not only_a) and (not only_b) and (d_a < rein_rad or d_b < rein_rad))  # close to both odours
            us[1] = float(punishment) * w
            us[0] = float(reward) * w

        for timestep in range(mb_model.nb_timesteps):
            yield trial, timestep, cs, us
            mb_model._t += 1

        t = mb_model._t
        s_at, s_av, r_at, r_av, m_at, m_av = mb_model._v[t, 6:]

        # attraction force = attraction - avoidance
        s = s_at - s_av
        r = r_at - r_av
        m = m_at - m_av

        # choose which MBONs to follow
        sw, rw, mw = float(susceptible), float(restrained), float(ltm)

        attraction = (sw * s + rw * r + mw * m) / (sw + rw + mw)  # magnitude of the overall attraction force
        d_a = (a_odour_source - agent.xy[t-1]) / non_zero_distance(a_odour_source, agent.xy[t-1])  # vector to odour A
        d_b = (b_odour_source - agent.xy[t-1]) / non_zero_distance(b_odour_source, agent.xy[t-1])  # vector to odour B
        direction = p_a * d_a + p_b * d_b  # direction of the overall attraction force (based on the intensity)
        direction /= np.maximum(np.absolute(direction), np.finfo(float).eps)  # normalise

        # calculate the attraction vector
        rho = attraction * direction

        # create a random vector
        epsilon = noise * (2 * agent.rng.rand() - 1 + (2 * agent.rng.rand() - 1) * 1j)

        # calculate the fly momentum
        if t < 2:
            momentum = 0+0j
        else:
            momentum = agent.xy[t-1] - agent.xy[t-2]
        if reset_position and (np.absolute(momentum) > 2 * agent.gain):
            momentum *= 0

        # calculate velocity based on the attraction force, the momentum and the random vector
        vel = rho + momentum + epsilon
        z = np.maximum(np.absolute(vel), np.finfo(float).eps)  # normalisation factor
        vel = agent.gain * vel / z  # ensure that the velocity has a standard magnitude

        new_xy = agent.xy[t-1] + vel  # update position
        agent.xy[t] = new_xy / np.maximum(np.absolute(new_xy), 1)  # make sure that we are still in the arena

        if reset_position and (trial_ == r_start * mb_model.nb_trials or trial_ == r_end * mb_model.nb_trials):
            agent.xy[t] *= 0.


def load_arena_stats(file_names, nb_active_kcs=2, prediction_error=False):
    """
    Creates a DataFrame that contains the stats of experiments in the arena.

    - susceptible: bool
    - restrained: bool
    - long-term memory: bool
    - repeat: int
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
    file_names: list[str]
        the names of the files used to calculate the statistics.
    prediction_error: bool, optional
        if the prediction error was used as the learning rule when creating the files. Default is False.

    Returns
    -------
    data: pd.DataFrame
        a DataFrame of size N x C, where N is the number of files x 3 and C is the number of features calculated
    """

    d_names = ["susceptible", "restrained", "long-term memory", "reinforcement",
               "paired odour", "phase", "angle", "absolute", "dist_A", "dist_B", "ang_A", "ang_B", "repeat"]
    d_raw = [[], [], [], [], [], [], [], [], [], [], [], [], []]

    for fname in file_names:
        if prediction_error:
            pattern = r'rw-arena-([\w]+)-kc([0-9])-(s{0,1})(r{0,1})(m{0,1})(a{0,1})(b{0,1})-?([0-9]*)'
        else:
            pattern = r'arena-([\w]+)-kc([0-9])-(s{0,1})(r{0,1})(m{0,1})(a{0,1})(b{0,1})-?([0-9]*)'
        details = re.findall(pattern, fname)
        if len(details) < 1:
            continue
        punishment = 'quinine' in details[0]
        susceptible = 's' in details[0]
        restrained = 'r' in details[0]
        ltm = 'm' in details[0]
        only_a = 'a' in details[0]
        only_b = 'b' in details[0]

        if int(details[0][1]) != nb_active_kcs:
            continue
        print(details[0])

        repeat = 0 if details[0][7] == '' else int(details[0][7])

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
        d_raw[7].extend(np.real(data[:, e_pre-1]))
        d_raw[7].extend(np.real(data[:, s_post-1]))
        d_raw[7].extend(np.real(data[:, -1]))
        d_raw[8].extend(np.absolute(data[:, e_pre-1] - FruitFly.a_source))
        d_raw[8].extend(np.absolute(data[:, s_post-1] - FruitFly.a_source))
        d_raw[8].extend(np.absolute(data[:, -1] - FruitFly.a_source))
        d_raw[9].extend(np.absolute(data[:, e_pre-1] - FruitFly.b_source))
        d_raw[9].extend(np.absolute(data[:, s_post-1] - FruitFly.b_source))
        d_raw[9].extend(np.absolute(data[:, -1] - FruitFly.b_source))
        d_raw[10].extend(np.angle(data[:, e_pre-1] - FruitFly.a_source))
        d_raw[10].extend(np.angle(data[:, s_post-1] - FruitFly.a_source))
        d_raw[10].extend(np.angle(data[:, -1] - FruitFly.a_source))
        d_raw[11].extend(np.angle(data[:, e_pre-1] - FruitFly.b_source))
        d_raw[11].extend(np.angle(data[:, s_post-1] - FruitFly.b_source))
        d_raw[11].extend(np.angle(data[:, -1] - FruitFly.b_source))
        d_raw[12].extend([repeat] * 3 * nb_flies)
    d_raw = np.array(d_raw)
    df = pd.DataFrame(d_raw, index=d_names).T
    df["angle"] = np.rad2deg(np.array(df["angle"], dtype=float))
    df["absolute"] = np.array(df["absolute"], dtype=float)
    df["dist_A"] = np.array(df["dist_A"], dtype=float)
    df["dist_B"] = np.array(df["dist_B"], dtype=float)
    df["ang_A"] = np.rad2deg(np.array(df["ang_A"], dtype=float))
    df["ang_B"] = np.rad2deg(np.array(df["ang_B"], dtype=float))
    df["susceptible"] = np.array(df["susceptible"] == "True", dtype=bool)
    df["restrained"] = np.array(df["restrained"] == "True", dtype=bool)
    df["long-term memory"] = np.array(df["long-term memory"] == "True", dtype=bool)
    df["repeat"] = np.array(df["repeat"], dtype=int)

    return df


def load_arena_paths(file_names, nb_active_kcs=2, max_repeat=None, prediction_error=False):
    """
    Loads the raw paths from the given files and returns their trace, case and name in 3
    separate lists.

    Parameters
    ----------
    file_names: list[str]
        list of filenames in the arena data directory.
    max_repeat: int, optional
        which repeat of the experiment to load. Default is the first.
    prediction_error: bool, optional
        if the prediction error was used as the learning rule when creating the files. Default is False.

    Returns
    -------
    d_raw: list[np.ndarray]
        the raw position for every case
    cases: list[list]
        the different cases associated with the data
    d_names: list[str]
        the name of each case
    """

    cases = [
        ["s", "p", "a"],
        ["s", "p", "b"],
        ["s", "p", ""],
        ["s", "r", "a"],
        ["s", "r", "b"],
        ["s", "r", ""],
        ["r", "p", "a"],
        ["r", "p", "b"],
        ["r", "p", ""],
        ["r", "r", "a"],
        ["r", "r", "b"],
        ["r", "r", ""],
        ["m", "p", "a"],
        ["m", "p", "b"],
        ["m", "p", ""],
        ["m", "r", "a"],
        ["m", "r", "b"],
        ["m", "r", ""],
        ["srm", "p", "a"],
        ["srm", "p", "b"],
        ["srm", "p", ""],
        ["srm", "r", "a"],
        ["srm", "r", "b"],
        ["srm", "r", ""],
    ]

    d_raw = [[]] * len(cases) * max_repeat
    d_names = [[]] * len(cases) * max_repeat
    d_repeats = [0] * len(cases) * max_repeat
    # d_names = ["susceptible", "reciprocal", "long-term memory", "reinforcement",
    #            "paired odour", "phase", "angle"]

    for fname in file_names:
        if prediction_error:
            pattern = r'^rw-arena-([\w]+)-kc([0-9])-(s{0,1})(r{0,1})(m{0,1})(a{0,1})(b{0,1})-?([0-9]*)'
        else:
            pattern = r'^arena-([\w]+)-kc([0-9])-(s{0,1})(r{0,1})(m{0,1})(a{0,1})(b{0,1})-?([0-9]*)'
        details = re.findall(pattern, fname)
        if len(details) < 1:
            continue

        if nb_active_kcs != int(details[0][1]):
            continue

        if max_repeat < (0 if details[0][7] == '' else int(details[0][7])):
            continue

        print(fname, details[0])
        punishment = "p" if 'quinine' in details[0] else "r"
        neurons = (
            ("s" if 's' in details[0] else "") +
            ("r" if "r" in details[0] else "") +
            ("m" if "m" in details[0] else "")
        )
        odour = (
            ("a" if "a" in details[0] else "") +
            ("b" if "b" in details[0] else "")
        )
        case = [neurons, punishment, odour]

        name = fname[:-4]

        if case in cases:
            i = cases.index(case) + len(cases) * (int(details[0][7]) - 1)
            d_raw[i] = np.load(os.path.join(__data_dir__, fname))["data"]
            d_names[i] = name
            d_repeats[i] = int(details[0][7])

    return d_raw, cases * max_repeat, d_names, d_repeats


def load_arena_traces(file_names, nb_active_kcs=2, repeat=None, prediction_error=False):
    """
    Loads the neural responses from the given files and returns their trace, case and name in 3
    separate lists.

    Parameters
    ----------
    file_names: list[str]
        list of filenames in the arena data directory.
    repeat: int, optional
        which repeat of the experiment to load. Default is the first.
    prediction_error: bool, optional
        if the prediction error was used as the learning rule when creating the files. Default is False.

    Returns
    -------
    d_raw: list[np.ndarray]
        the raw position for every case
    cases: list[list]
        the different cases associated with the data
    d_names: list[str]
        the name of each case
    """

    cases = [
        # ["s", "p", "a"],
        # ["s", "p", "b"],
        # ["s", "p", ""],
        # ["s", "r", "a"],
        # ["s", "r", "b"],
        # ["s", "r", ""],
        # ["r", "p", "a"],
        # ["r", "p", "b"],
        # ["r", "p", ""],
        # ["r", "r", "a"],
        # ["r", "r", "b"],
        # ["r", "r", ""],
        # ["m", "p", "a"],
        # ["m", "p", "b"],
        # ["m", "p", ""],
        # ["m", "r", "a"],
        # ["m", "r", "b"],
        # ["m", "r", ""],
        ["srm", "p", "a"],
        ["srm", "p", "b"],
        ["srm", "p", ""],
        ["srm", "r", "a"],
        ["srm", "r", "b"],
        ["srm", "r", ""],
    ]

    d_res = [[]] * len(cases)
    d_wei = [[]] * len(cases)
    d_nam = [[]] * len(cases)
    d_names = [[]] * len(cases)
    # d_names = ["susceptible", "reciprocal", "long-term memory", "reinforcement",
    #            "paired odour", "phase", "angle"]

    for fname in file_names:
        if prediction_error:
            pattern = r'^rw-arena-([\w]+)-kc([0-9])-(s{0,1})(r{0,1})(m{0,1})(a{0,1})(b{0,1})-?([0-9]*)'
        else:
            pattern = r'^arena-([\w]+)-kc([0-9])-(s{0,1})(r{0,1})(m{0,1})(a{0,1})(b{0,1})-?([0-9]*)'
        details = re.findall(pattern, fname)
        if len(details) < 1:
            continue

        if repeat != (None if details[0][7] == '' else int(details[0][7])):
            continue
        if nb_active_kcs != int(details[0][1]):
            continue

        print(fname, details[0])
        punishment = "p" if 'quinine' in details[0] else "r"
        neurons = (
            ("s" if 's' in details[0] else "") +
            ("r" if "r" in details[0] else "") +
            ("m" if "m" in details[0] else "")
        )
        odour = (
            ("a" if "a" in details[0] else "") +
            ("b" if "b" in details[0] else "")
        )
        case = [neurons, punishment, odour]

        name = fname[:-4]

        if case in cases:
            i = cases.index(case)
            d = np.load(os.path.join(__data_dir__, fname))
            d_res[i] = d["response"]
            d_wei[i] = d["weights"]
            d_nam[i] = d["names"]
            d_names[i] = name

    return d_res, d_wei, d_nam, cases, d_names


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


def non_zero_distance(a, b):
    return np.maximum(np.linalg.norm(a - b), np.finfo(float).eps)
