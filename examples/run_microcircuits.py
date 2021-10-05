from incentive.circuit import IncentiveCircuit
from incentive.results import run_custom_routine
from incentive.routines import rewarding_routine, shock_routine
from incentive.plot import plot_sm, plot_ltm, plot_rm, plot_rrm, plot_rfm, plot_mam
from incentive.tools import read_arg

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright 2021, School of Informatics, the University of Edinburgh"
__licence__ = "MIT"
__version__ = "1.1-alpha"
__maintainer__ = "Evripidis Gkanias"
__email__ = "ev.gkanias@ed.ac.uk"
__status__ = "Production"


if __name__ == '__main__':

    # read the parameters
    nb_kcs = read_arg(["-k", "--nb-kc", "--nb-kcs"], vtype=int, default=10)
    kc1 = read_arg(["-k1", "--nb-kc1", "--odour1"], vtype=int, default=nb_kcs // 2)
    kc2 = read_arg(["-k2", "--nb-kc2", "--odour2"], vtype=int, default=kc1)

    # Susceptible Memory Sub-circuit
    if read_arg(["--sm", "--susceptible-memory"]):
        model = IncentiveCircuit(
            learning_rule="dlr", nb_apl=0, nb_timesteps=3, nb_trials=24,
            nb_kc=nb_kcs, nb_kc_odour=kc2, has_real_names=False,
            has_sm=True, has_rm=False, has_ltm=False, has_rrm=False, has_rfm=False, has_mam=False,
            as_microcircuits=True)
        plot_sm(run_custom_routine(model, routine=shock_routine))

    # Restrained Memory Sub-circuit
    if read_arg(["--rm", "--restrained-memory"]):
        model = IncentiveCircuit(
            learning_rule="dlr", nb_apl=0, nb_timesteps=3, nb_trials=24,
            nb_kc=nb_kcs, nb_kc_odour=kc2, has_real_names=False,
            has_sm=True, has_rm=True, has_ltm=False, has_rrm=False, has_rfm=False, has_mam=False,
            as_microcircuits=True)
        plot_rm(run_custom_routine(model, routine=shock_routine))

    # Long-Term Memory Sub-circuit
    if read_arg(["--ltm", "--long-term-memory"]):
        model = IncentiveCircuit(
            learning_rule="dlr", nb_apl=0, nb_timesteps=3, nb_trials=24,
            nb_kc=nb_kcs, nb_kc_odour=kc2, has_real_names=False,
            has_sm=True, has_rm=True, has_ltm=True, has_rrm=False, has_rfm=False, has_mam=False,
            as_microcircuits=True)
        plot_ltm(run_custom_routine(model, routine=rewarding_routine))

    # Reciprocal Restrained Memory Sub-circuit
    if read_arg(["--rrm", "--reciprocal-restrained-memory"]):
        model = IncentiveCircuit(
            learning_rule="dlr", nb_apl=0, nb_timesteps=3, nb_trials=24,
            nb_kc=nb_kcs, nb_kc_odour=kc2, has_real_names=False,
            has_sm=True, has_rm=True, has_ltm=True, has_rrm=True, has_rfm=False, has_mam=False,
            as_microcircuits=True)
        plot_rrm(run_custom_routine(model, routine=shock_routine))

    # Reciprocal Forgetting Memory Sub-circuit
    if read_arg(["--rfm", "--reciprocal-forgetting-memory"]):
        model = IncentiveCircuit(
            learning_rule="dlr", nb_apl=0, nb_timesteps=3, nb_trials=24,
            nb_kc=nb_kcs, nb_kc_odour=kc2, has_real_names=False,
            has_sm=True, has_rm=True, has_ltm=True, has_rrm=True, has_rfm=True, has_mam=True,
            as_microcircuits=True)
        plot_rfm(run_custom_routine(model, routine=shock_routine))

    # Memory Assimilation Mechanism
    if read_arg(["--mam", "--memory-assimilation-mechanism"]):
        model = IncentiveCircuit(
            learning_rule="dlr", nb_apl=0, nb_timesteps=3, nb_trials=24,
            nb_kc=nb_kcs, nb_kc_odour=kc2, has_real_names=False,
            has_sm=True, has_rm=True, has_ltm=True, has_rrm=True, has_rfm=True, has_mam=True,
            as_microcircuits=True)
        plot_mam(run_custom_routine(model, routine=rewarding_routine))

