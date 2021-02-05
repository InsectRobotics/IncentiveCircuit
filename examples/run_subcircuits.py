from twinspoke import TwinSpokeModel
from results import run_custom_routine
from routines import rewarding_routine, shock_routine
from plot import plot_fom, plot_ltm, plot_bm, plot_rsom, plot_rfm, plot_mdm
from tools import read_arg


if __name__ == '__main__':
    nb_kcs = read_arg(["-k", "--nb-kc", "--nb-kcs"], vtype=int, default=10)
    kc1 = read_arg(["-k1", "--nb-kc1", "--odour1"], vtype=int, default=nb_kcs // 2)
    kc2 = read_arg(["-k2", "--nb-kc2", "--odour2"], vtype=int, default=nb_kcs // 2)
    # must set h-mbons bias to -2 and m-mbons bias to -4 in order for this to produce the desired results

    # First Order Memory Sub-circuit
    if read_arg(["--fom", "--first-order-memory"]):
        model = TwinSpokeModel(
            learning_rule="dlr", nb_apl=0, pn2kc_init="default", nb_timesteps=3, nb_trials=24,
            nb_kc=nb_kcs, nb_kc_odour_1=kc1, nb_kc_odour_2=kc2, has_real_names=False,
            has_fom=True, has_bm=False, has_ltm=False, has_rsom=False, has_rfm=False, has_mdm=False,
            as_subcircuits=True)
        plot_fom(run_custom_routine(model, routine=shock_routine))

    # Long-Term Memory Sub-circuit
    if read_arg(["--ltm", "--long-term-memory"]):
        model = TwinSpokeModel(
            learning_rule="dlr", nb_apl=0, pn2kc_init="default", nb_timesteps=3, nb_trials=24,
            nb_kc=nb_kcs, nb_kc_odour_1=kc1, nb_kc_odour_2=kc2, has_real_names=False,
            has_fom=False, has_bm=False, has_ltm=True, has_rsom=False, has_rfm=False, has_mdm=False,
            as_subcircuits=True)
        plot_ltm(run_custom_routine(model, routine=rewarding_routine))

    # Blocking Memory Sub-circuit
    if read_arg(["--bm", "--blocking-memory"]):
        model = TwinSpokeModel(
            learning_rule="dlr", nb_apl=0, pn2kc_init="default", nb_timesteps=3, nb_trials=24,
            nb_kc=nb_kcs, nb_kc_odour_1=kc1, nb_kc_odour_2=kc2, has_real_names=False,
            has_fom=True, has_bm=True, has_ltm=False, has_rsom=False, has_rfm=False, has_mdm=False,
            as_subcircuits=True)
        plot_bm(run_custom_routine(model, routine=shock_routine))

    # Reciprocal Second Order Memory Sub-circuit
    if read_arg(["--rsom", "--reciprocal-second-order-memory"]):
        model = TwinSpokeModel(
            learning_rule="dlr", nb_apl=0, pn2kc_init="default", nb_timesteps=3, nb_trials=24,
            nb_kc=nb_kcs, nb_kc_odour_1=kc1, nb_kc_odour_2=kc2, has_real_names=False,
            has_fom=False, has_bm=False, has_ltm=False, has_rsom=True, has_rfm=False, has_mdm=False,
            as_subcircuits=True)
        plot_rsom(run_custom_routine(model, routine=shock_routine))

    # Reciprocal LTM Forgetting Memory Sub-circuit
    if read_arg(["--rfm", "--reciprocal-forgetting-memory"]):
        model = TwinSpokeModel(
            learning_rule="dlr", nb_apl=0, pn2kc_init="default", nb_timesteps=3, nb_trials=24,
            nb_kc=nb_kcs, nb_kc_odour_1=kc1, nb_kc_odour_2=kc2, has_real_names=False,
            has_fom=False, has_bm=False, has_ltm=True, has_rsom=True, has_rfm=True, has_mdm=True,
            as_subcircuits=True)
        plot_rfm(run_custom_routine(model, routine=shock_routine))

    # Memory digestion Mechanism
    if read_arg(["--mdm", "--memory-digestion-mechanism"]):
        model = TwinSpokeModel(
            learning_rule="dlr", nb_apl=0, pn2kc_init="default", nb_timesteps=3, nb_trials=24,
            nb_kc=nb_kcs, nb_kc_odour_1=kc1, nb_kc_odour_2=kc2, has_real_names=False,
            has_fom=False, has_bm=False, has_ltm=True, has_rsom=True, has_rfm=True, has_mdm=True,
            as_subcircuits=True)
        plot_mdm(run_custom_routine(model, routine=rewarding_routine))

