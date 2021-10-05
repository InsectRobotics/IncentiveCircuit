from incentive.circuit import IncentiveCircuit
from incentive.results import run_main_experiments
from incentive.tools import read_arg, run_arg

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright 2021, School of Informatics, the University of Edinburgh"
__licence__ = "MIT"
__version__ = "1.1-alpha"
__maintainer__ = "Evripidis Gkanias"
__email__ = "ev.gkanias@ed.ac.uk"
__status__ = "Production"

if __name__ == '__main__':

    # read the parameters
    only_nids = read_arg(["--only-nids"])
    nb_kcs = read_arg(["-k", "--nb-kc", "--nb-kcs"], vtype=int, default=10)
    kc1 = read_arg(["-k1", "--nb-kc1", "--odour1"], vtype=int, default=nb_kcs // 2)
    kc2 = read_arg(["-k2", "--nb-kc2", "--odour2"], vtype=int, default=nb_kcs // 2)
    # kc1 = read_arg(["-k1", "--nb-kc1", "--odour1"], vtype=int, default=nb_kcs // 2 + 1)
    # kc2 = read_arg(["-k2", "--nb-kc2", "--odour2"], vtype=int, default=nb_kcs // 2 + 2)

    # create the Incentive Circuit
    model = IncentiveCircuit(
        learning_rule="dlr", nb_apl=0, nb_timesteps=3, nb_trials=26,
        nb_kc=nb_kcs, nb_kc_odour_1=kc1, nb_kc_odour_2=kc2, has_real_names=False,
        has_sm=True, has_rm=True, has_rrm=True, has_ltm=True, has_rfm=True, has_mam=True)

    # run all the experiments and get a copy of the model with the history of their responses and parameters for each
    # one of them
    models = run_main_experiments(model, reversal=True, unpaired=True, extinction=True)

    # plot the results based on the input flags
    run_arg(model, models, only_nids)
