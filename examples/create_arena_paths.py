# #!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Creates the paths of the freely-moving flies using the incentive circuit.

Examples:
In order to create all the possible combinations for 100 flies each and for 500 seconds, run
    $ python3 create_arena_paths.py --nb-flies 100 --nb-time-steps 500
or
    $ python3 create_arena_paths.py -f 100 -t 500

In order to generate the data for 100 flies and for 500 seconds, for the punishment delivery case where the motivation
is being set by the restrained MBONs only, run
    $ python3 create_arena_paths.py -f 100 -t 500 -p -ns -r -nm

"""

from incentive.tools import read_arg, ROOT_DIR

import numpy as np
import yaml
import os

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright 2021, School of Informatics, the University of Edinburgh"
__licence__ = "MIT"
__version__ = "1.1-alpha"
__maintainer__ = "Evripidis Gkanias"
__email__ = "ev.gkanias@ed.ac.uk"
__status__ = "Production"


with open(os.path.join(os.path.join(ROOT_DIR, "data"), 'model-parameters.yml'), 'rb') as f:
    model_params = yaml.load(f, Loader=yaml.BaseLoader)
    """load the default parameters of the model"""


if __name__ == '__main__':
    from incentive.arena import FruitFly
    from incentive import arena

    nb_kcs = int(model_params["number-kc"])
    nb_kc1 = int(model_params["number-kc-odour-a"])
    nb_kc2 = int(model_params["number-kc-odour-b"])
    nb_active_kcs = int(model_params["number-kc-active"])
    sv = float(model_params["susceptible-weight"])
    rv = float(model_params["stm-weight"])
    mv = float(model_params["ltm-weight"])
    nb_flies = read_arg(["-f", "--nb-flies"], vtype=int, default=100)
    nb_timesteps = read_arg(["-t", "--nb-time-steps"], vtype=int, default=100)  # seconds
    arena.__data_dir__ = directory = os.path.abspath(read_arg(["-d", "--dir"], vtype=str, default=arena.__data_dir__))
    repeats = read_arg(["-R", "--repeat"], vtype=int, default=10)

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
    ss, rs, ms, a_s, bs, ps = np.meshgrid(
        susceptible, reciprocal, ltm, only_a, only_b, punishment
    )

    conditions = ["srm", "s", "r", "m"]
    regions = ["", "a", "b"]

    for punishment, susceptible, reciprocal, ltm, only_a, only_b in zip(
            ps.reshape(-1), ss.reshape(-1), rs.reshape(-1), ms.reshape(-1), a_s.reshape(-1), bs.reshape(-1)):

        name = "arena-quinine-" if punishment else "arena-sugar-"
        name += "kc%d-" % nb_active_kcs
        name_ext = ""
        if susceptible:
            name_ext += "s"
        if reciprocal:
            name_ext += "r"
        if ltm:
            name_ext += "m"

        name_reg = ""
        if only_a:
            name_reg += "a"
        if only_b:
            name_reg += "b"

        if name_ext not in conditions or name_reg not in regions:
            continue

        name += name_ext + name_reg

        data = np.zeros((nb_flies, nb_timesteps), dtype=complex)
        responses = np.zeros((nb_flies, nb_timesteps, 12), dtype=float)
        weights = np.zeros((nb_flies, nb_timesteps, nb_kcs, 12), dtype=float)
        names = None
        flies = []

        for repeat in range(repeats):
            print(name, end=' ')
            for i in range(nb_flies):
                if len(flies) <= i:
                    fly = FruitFly(rng=rng, nb_steps=nb_timesteps, gain=0.05, nb_kcs=nb_kcs,
                                   nb_kc_odour_a=nb_kc1, nb_kc_odour_b=nb_kc2, nb_active_kcs=nb_active_kcs,
                                   ltm_speed=float(model_params["ltm-speed"]))
                    flies.append(fly)
                else:
                    fly = flies[i]
                fly(punishment=punishment, noise=.5, susceptible=susceptible, reciprocal=reciprocal, ltm=ltm,
                    only_a=only_a, only_b=only_b)
                data[i] = fly.xy.copy()
                responses[i] = fly.mb._v[1:].copy()
                weights[i] = fly.mb.w_k2m[1:].copy()
                if names is None:
                    names = fly.mb.names

            if repeats > 1:
                print("R:", repeat + 1)
                name_r = name + "-%02d" % (repeat+1)
            else:
                print()
                name_r = name
            np.savez(os.path.join(directory, "%s.npz" % name_r),
                     data=data, response=responses, weights=weights, names=names)
