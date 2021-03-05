from incentive.tools import read_arg

import numpy as np
import os


# the directory of the file
__dir__ = os.path.dirname(os.path.abspath(__file__))
# the directory of the data
__data_dir__ = os.path.realpath(os.path.join(__dir__, "..", "data", "arena"))

if __name__ == '__main__':
    from incentive.arena import FruitFly

    sv = 1.
    rv = 1.
    mv = 10.
    nb_flies = read_arg(["-f", "--nb-flies"], vtype=int, default=100)
    nb_timesteps = read_arg(["-t", "--nb-time-steps"], vtype=int, default=100)

    if read_arg(["-p", "--punishment"]):
        punishment = [True]
    elif read_arg(["-np", "--not-punishment", "--reward"]):
        punishment = [False]
    else:
        punishment = [True, False]
    if read_arg(["-s", "--susceptible"]):
        susceptible = [sv]
    elif read_arg(["-ns", "--not-susceptible"]):
        susceptible = [0.]
    else:
        susceptible = [sv, 0.]
    if read_arg(["-r", "--restrained"]):
        reciprocal = [rv]
    elif read_arg(["-nr", "--not-restrained"]):
        reciprocal = [0.]
    else:
        reciprocal = [rv, 0.]
    if read_arg(["-m", "--ltm", "--long-term-memory"]):
        ltm = [mv]
    elif read_arg(["-nm", "--not-ltm", "--not-long-term-memory"]):
        ltm = [0.]
    else:
        ltm = [mv, 0.]
    if read_arg(["-a", "--only-a"]):
        only_a = [True]
    elif read_arg(["-na", "--not-a", "--not-only-a"]):
        only_a = [False]
    else:
        only_a = [True, False]
    if read_arg(["-b", "--only-b"]):
        only_b = [True]
    elif read_arg(["-nb", "--not-b", "--not-only-b"]):
        only_b = [False]
    else:
        only_b = [True, False]
    rng = np.random.RandomState(read_arg(["--rng", "--random-state"], vtype=int, default=2021))
    ps, ss, rs, ms, a_s, bs = np.meshgrid(
        punishment, susceptible, reciprocal, ltm, only_a, only_b
    )

    for punishment, susceptible, reciprocal, ltm, only_a, only_b in zip(
            ps.reshape(-1), ss.reshape(-1), rs.reshape(-1), ms.reshape(-1), a_s.reshape(-1), bs.reshape(-1)):
        if (((not susceptible) and reciprocal and ltm) or
                (susceptible and (not reciprocal) and ltm) or
                (susceptible and reciprocal and (not ltm)) or
                ((not susceptible) and (not reciprocal) and (not ltm)) or
                (only_a and only_b)):
            continue

        name = "rw-arena-quinine-" if punishment else "rw-arena-sugar-"
        if susceptible:
            name += "s"
        if reciprocal:
            name += "r"
        if ltm:
            name += "m"
        if only_a:
            name += "a"
        if only_b:
            name += "b"

        print(name)

        data = np.zeros((nb_flies, nb_timesteps), dtype=np.complex)
        for i in range(nb_flies):
            fly = FruitFly(rng=rng, nb_steps=nb_timesteps, learning_rule="rw")
            fly(punishment=punishment, noise=.5, susceptible=susceptible, reciprocal=reciprocal, ltm=ltm,
                only_a=only_a, only_b=only_b)
            data[i] = fly.xy

        np.savez(os.path.join(__data_dir__, "%s.npz" % name), data=data)
