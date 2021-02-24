from incentivecomplex import IncentiveComplex
from results import run_custom_routine
from routines import rewarding_routine, shock_routine
from plot import plot_fom, plot_ltm, plot_bm, plot_rsom, plot_rfm, plot_mdm
from tools import read_arg


if __name__ == '__main__':

    # read the parameters
    nb_kcs = read_arg(["-k", "--nb-kc", "--nb-kcs"], vtype=int, default=10)
    kc1 = read_arg(["-k1", "--nb-kc1", "--odour1"], vtype=int, default=nb_kcs // 2)
    kc2 = read_arg(["-k2", "--nb-kc2", "--odour2"], vtype=int, default=nb_kcs // 2)

    # Susceptible Memory Sub-circuit
    if read_arg(["--sm", "--susceptible-memory"]):
        model = IncentiveComplex(
            learning_rule="dlr", nb_apl=0, pn2kc_init="default", nb_timesteps=3, nb_trials=24,
            nb_kc=nb_kcs, nb_kc_odour_1=kc1, nb_kc_odour_2=kc2, has_real_names=False,
            has_sm=True, has_rm=False, has_ltm=False, has_rrm=False, has_rfm=False, has_mam=False,
            as_subcircuits=True)
        plot_fom(run_custom_routine(model, routine=shock_routine))

    # Restrained Memory Sub-circuit
    if read_arg(["--rm", "--restrained-memory"]):
        model = IncentiveComplex(
            learning_rule="dlr", nb_apl=0, pn2kc_init="default", nb_timesteps=3, nb_trials=24,
            nb_kc=nb_kcs, nb_kc_odour_1=kc1, nb_kc_odour_2=kc2, has_real_names=False,
            has_sm=True, has_rm=True, has_ltm=False, has_rrm=False, has_rfm=False, has_mam=False,
            as_subcircuits=True)
        plot_bm(run_custom_routine(model, routine=shock_routine))

    # Long-Term Memory Sub-circuit
    if read_arg(["--ltm", "--long-term-memory"]):
        model = IncentiveComplex(
            learning_rule="dlr", nb_apl=0, pn2kc_init="default", nb_timesteps=3, nb_trials=24,
            nb_kc=nb_kcs, nb_kc_odour_1=kc1, nb_kc_odour_2=kc2, has_real_names=False,
            has_sm=True, has_rm=True, has_ltm=True, has_rrm=False, has_rfm=False, has_mam=False,
            as_subcircuits=True)
        plot_ltm(run_custom_routine(model, routine=rewarding_routine))

    # Reciprocal Restrained Memory Sub-circuit
    if read_arg(["--rrm", "--reciprocal-restrained-memory"]):
        model = IncentiveComplex(
            learning_rule="dlr", nb_apl=0, pn2kc_init="default", nb_timesteps=3, nb_trials=24,
            nb_kc=nb_kcs, nb_kc_odour_1=kc1, nb_kc_odour_2=kc2, has_real_names=False,
            has_sm=True, has_rm=True, has_ltm=True, has_rrm=True, has_rfm=False, has_mam=False,
            as_subcircuits=True)
        plot_rsom(run_custom_routine(model, routine=shock_routine))

    # Reciprocal Forgetting Memory Sub-circuit
    if read_arg(["--rfm", "--reciprocal-forgetting-memory"]):
        model = IncentiveComplex(
            learning_rule="dlr", nb_apl=0, pn2kc_init="default", nb_timesteps=3, nb_trials=24,
            nb_kc=nb_kcs, nb_kc_odour_1=kc1, nb_kc_odour_2=kc2, has_real_names=False,
            has_sm=True, has_rm=True, has_ltm=True, has_rrm=True, has_rfm=True, has_mam=True,
            as_subcircuits=True)
        plot_rfm(run_custom_routine(model, routine=shock_routine))

    # Memory Assimilation Mechanism
    if read_arg(["--mam", "--memory-assimilation-mechanism"]):
        model = IncentiveComplex(
            learning_rule="dlr", nb_apl=0, pn2kc_init="default", nb_timesteps=3, nb_trials=24,
            nb_kc=nb_kcs, nb_kc_odour_1=kc1, nb_kc_odour_2=kc2, has_real_names=False,
            has_sm=True, has_rm=True, has_ltm=True, has_rrm=True, has_rfm=True, has_mam=True,
            as_subcircuits=True)
        plot_mdm(run_custom_routine(model, routine=rewarding_routine))

