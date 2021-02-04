import numpy as np


def reversal_routine(mb_model):
    cs_on = np.arange(mb_model.nb_trials * 2)
    us_on = np.array([3, 5, 7, 9, 11, 14, 16, 18, 20, 22, 24])
    mb_model.routine_name = "reversal"
    return _routine_base(mb_model, odour=cs_on, shock=us_on)


def unpaired_routine(mb_model):
    cs_on = np.arange(mb_model.nb_trials * 2)
    us_on = np.array([3, 5, 7, 9, 11, 14, 16, 18, 20, 22, 24])
    mb_model.routine_name = "unpaired"
    return _routine_base(mb_model, odour=cs_on, shock=us_on, paired=[3, 5, 7, 9, 11])


def no_shock_routine(mb_model):
    cs_on = np.arange(mb_model.nb_trials * 2)
    us_on = np.array([3, 5, 7, 9, 11])
    mb_model.routine_name = "no shock"
    return _routine_base(mb_model, odour=cs_on, shock=us_on)


def shock_routine(mb_model, timesteps=100):
    mb_model._t = 0
    mb_model.nb_trials = 1
    mb_model.nb_timesteps = timesteps
    mb_model.w_k2m = np.array([mb_model.w_k2m[0]] * (timesteps + 1))
    mb_model._v = np.array([mb_model._v[0]] * (timesteps + 1))
    mb_model._v_apl = np.array([mb_model._v_apl[0]] * (timesteps + 1))
    for timestep in range(timesteps):
        cs_ = mb_model.csa
        us_ = np.zeros(mb_model.us_dims, dtype=float)
        if mb_model.us_dims > 2:
            us_[4] = 2.
        else:
            us_[1] = 2.

        if timestep < .1 * timesteps:
            cs = cs_ * 0.
            us = us_ * 0.
        elif timestep < .3 * timesteps:
            cs = cs_ * 1.
            us = us_ * 0.
        elif timestep < .6 * timesteps:
            cs = cs_ * 1.
            us = us_ * 1.
        elif timestep < .8 * timesteps:
            cs = cs_ * 1.
            us = us_ * 0.
        else:
            cs = cs_ * 0.
            us = us_ * 0.

        yield 1, timestep, cs, us

        mb_model._t += 1


def sugar_routine(mb_model, timesteps=100):
    mb_model._t = 0
    mb_model.nb_trials = 1
    mb_model.nb_timesteps = timesteps
    mb_model.w_k2m = np.array([mb_model.w_k2m[0]] * (timesteps + 1))
    mb_model._v = np.array([mb_model._v[0]] * (timesteps + 1))
    mb_model._v_apl = np.array([mb_model._v_apl[0]] * (timesteps + 1))
    for timestep in range(timesteps):
        cs_ = mb_model.csa
        us_ = np.zeros(mb_model.us_dims, dtype=float)
        if mb_model.us_dims > 0:
            us_[0] = 2.

        if timestep < .1 * timesteps:
            cs = cs_ * 0.
            us = us_ * 0.
        elif timestep < .3 * timesteps:
            cs = cs_ * 1.
            us = us_ * 0.
        elif timestep < .6 * timesteps:
            cs = cs_ * 1.
            us = us_ * 1.
        elif timestep < .8 * timesteps:
            cs = cs_ * 1.
            us = us_ * 0.
        else:
            cs = cs_ * 0.
            us = us_ * 0.

        yield 1, timestep, cs, us

        mb_model._t += 1


def extended_shock_routine(mb_model, timesteps=100):
    mb_model._t = 0
    mb_model.nb_trials = 1
    mb_model.nb_timesteps = timesteps
    mb_model.w_k2m = np.array([mb_model.w_k2m[0]] * (timesteps + 1))
    mb_model._v = np.array([mb_model._v[0]] * (timesteps + 1))
    mb_model._v_apl = np.array([mb_model._v_apl[0]] * (timesteps + 1))
    for timestep in range(timesteps):
        cs_ = mb_model.csa
        us_ = np.zeros(mb_model.us_dims, dtype=float)
        if mb_model.us_dims > 2:
            us_[4] = 2.
        else:
            us_[1] = 2.

        if timestep < .1 * timesteps:
            cs = cs_ * 0.
            us = us_ * 0.
        elif timestep < .2 * timesteps:
            cs = cs_ * 1.
            us = us_ * 0.
        elif timestep < .3 * timesteps:
            cs = cs_ * 1.
            us = us_ * 1.
        elif timestep < .5 * timesteps:
            cs = cs_ * 1.
            us = us_ * 0.
        elif timestep < .6 * timesteps:
            cs = cs_ * 0.
            us = us_ * 0.
        elif timestep < .65 * timesteps:
            cs = cs_ * 0.
            us = us_ * 1.
        elif timestep < .75 * timesteps:
            cs = cs_ * 0.
            us = us_ * 0.
        elif timestep < .85 * timesteps:
            cs = cs_ * 1.
            us = us_ * 0.
        else:
            cs = cs_ * 0.
            us = us_ * 0.

        yield 1, timestep, cs, us

        mb_model._t += 1


def extended_sugar_routine(mb_model, timesteps=100):
    mb_model._t = 0
    mb_model.nb_trials = 1
    mb_model.nb_timesteps = timesteps
    mb_model.w_k2m = np.array([mb_model.w_k2m[0]] * (timesteps + 1))
    mb_model._v = np.array([mb_model._v[0]] * (timesteps + 1))
    mb_model._v_apl = np.array([mb_model._v_apl[0]] * (timesteps + 1))
    for timestep in range(timesteps):
        cs_ = mb_model.csa
        us_ = np.zeros(mb_model.us_dims, dtype=float)
        if mb_model.us_dims > 0:
            us_[0] = 2.

        if timestep < .1 * timesteps:
            cs = cs_ * 0.
            us = us_ * 0.
        elif timestep < .2 * timesteps:
            cs = cs_ * 1.
            us = us_ * 0.
        elif timestep < .3 * timesteps:
            cs = cs_ * 1.
            us = us_ * 1.
        elif timestep < .5 * timesteps:
            cs = cs_ * 1.
            us = us_ * 0.
        elif timestep < .6 * timesteps:
            cs = cs_ * 0.
            us = us_ * 0.
        elif timestep < .65 * timesteps:
            cs = cs_ * 0.
            us = us_ * 1.
        elif timestep < .75 * timesteps:
            cs = cs_ * 0.
            us = us_ * 0.
        elif timestep < .85 * timesteps:
            cs = cs_ * 1.
            us = us_ * 0.
        else:
            cs = cs_ * 0.
            us = us_ * 0.

        yield 1, timestep, cs, us

        mb_model._t += 1


def _routine_base(mb_model, odour=None, shock=None, paired=None):
    mb_model._t = 0
    if odour is None:
        odour = np.arange(mb_model.nb_trials * 2)
    if shock is None:
        shock = np.arange(mb_model.nb_trials * 2)
    if paired is None:
        paired = np.arange(mb_model.nb_trials * 2)

    for trial in range(1, mb_model.nb_trials // 2 + 2):
        for cs_ in [mb_model.csa, mb_model.csb]:
            if mb_model._t >= mb_model.nb_trials * mb_model.nb_timesteps:
                break

            trial_ = mb_model._t // mb_model.nb_timesteps

            # odour is presented only in specific trials
            cs__ = cs_ * float(trial_ in odour)

            # shock is presented only in specific trials
            us__ = np.zeros(mb_model.us_dims, dtype=float)
            if mb_model.us_dims > 2:
                us__[4] = float(trial_ in shock)
            else:
                us__[1] = float(trial_ in shock)

            for timestep in range(mb_model.nb_timesteps):

                # we skip odour in the first timestep of the trial
                cs = cs__ * float(timestep > 0)
                if trial_ in paired:
                    # shock is presented only after the 4th sec of the trial
                    us = us__ * float(4 <= 5 * (timestep + 1) / mb_model.nb_timesteps)
                else:
                    us = us__ * float(timestep < 1)
                # print(self.__routine_name, trial, timestep, cs, us)
                if mb_model.verbose:
                    print("Trial: %d / %d, %s" % (trial_ + 1, mb_model.nb_trials, ["CS-", "CS+"][mb_model._t % 2]))
                yield trial, timestep, cs, us

                mb_model._t += 1
