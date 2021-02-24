from twinspoke import IncentiveComplex

import numpy as np


class FruitFly(object):
    a_source = .6+0j
    a_sigma = .2
    b_source = -.6+0j
    b_sigma = .3

    def __init__(self, nb_kcs=10, nb_kc_odour_a=5, nb_kc_odour_b=5, nb_steps=1000, nb_in_trial=1, learning_rule="dlr",
                 rng=np.random.RandomState()):
        self.mb = IncentiveComplex(
            learning_rule=learning_rule, nb_apl=0, pn2kc_init="default", nb_timesteps=nb_in_trial, nb_trials=nb_steps,
            nb_kc=nb_kcs, nb_kc_odour_1=nb_kc_odour_a, nb_kc_odour_2=nb_kc_odour_b, has_real_names=False,
            has_sm=True, has_rm=True, has_ltm=True, has_rrm=True, has_rfm=True, has_mam=True)

        self.xy = np.zeros(nb_steps, dtype=np.complex)
        self.p_a = np.zeros(nb_steps, dtype=float)
        self.p_b = np.zeros(nb_steps, dtype=float)
        self.turn = np.zeros(nb_steps, dtype=float)
        self.rng = rng

    def __call__(self, *args, **kwargs):
        punishment = kwargs.get("punishment", True)
        reward = kwargs.get("reward", not punishment)
        r_start = kwargs.get("r_start", .2)
        r_end = kwargs.get("r_end", .5)
        noise = kwargs.get("noise", .1)
        s = kwargs.get("susceptible", True)
        r = kwargs.get("reciprocal", True)
        m = kwargs.get("ltm", True)
        a = kwargs.get("only_a", False)
        b = kwargs.get("only_b", False)
        routine = arena_routine(
            self, noise=noise, r_start=r_start, r_end=r_end, reward=reward, punishment=punishment,
            susceptible=s, reciprocal=r, ltm=m, only_a=a, only_b=b)
        self.mb(routine=routine)
        pass

    @property
    def t(self):
        return self.mb._t


def arena_routine(agent, noise=0.1, r_start=.2, r_end=.5, reward=False, punishment=True,
                  susceptible=True, reciprocal=True, ltm=True, only_a=False, only_b=False):
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

        sw, rw, mw = float(susceptible), float(reciprocal), float(ltm)
        attraction = (sw * s + rw * r + mw * m) / (sw + rw + mw)
        if t < 2:
            vel = 0+0j
        else:
            vel = agent.xy[t-1] - agent.xy[t-2]
        if p_a > p_b:
            rho = attraction * (a_odour_source - agent.xy[t-1])
        else:
            rho = attraction * (b_odour_source - agent.xy[t-1])

        vel += rho + agent.rng.randn() * .1 + agent.rng.randn() * .1j
        z = np.maximum(np.absolute(vel), np.finfo(float).eps)
        vel = 0.01 * vel / z

        agent.xy[t] = agent.xy[t-1] + vel
        agent.xy[t] = agent.xy[t] / np.maximum(np.absolute(agent.xy[t]), 1)


def gaussian_p(pos, mean, sigma):
    if pos.dtype == np.complex:
        pos = np.array([pos.real, pos.imag]).T
    if isinstance(mean, np.complex):
        mean = np.array([mean.real, mean.imag]).T
    pos, mean = np.array(pos), np.array(mean)

    return 1. / (np.sqrt(2 * np.pi) * sigma) * np.exp(-np.sum(np.square(pos - mean), axis=-1) / (2 * np.square(sigma)))
