__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright 2021, School of Informatics, the University of Edinburgh"
__licence__ = "MIT"
__version__ = "1.1-alpha"
__maintainer__ = "Evripidis Gkanias"
__email__ = "ev.gkanias@ed.ac.uk"
__status__ = "Production"

from incentive.handler import run_case, load_means, load_statistics
from incentive.handler import erca_time_adjust, camp_time_adjust, stim, conditions

from scipy.stats import pearsonr

import numpy as np
import matplotlib.pyplot as plt

import os

__dir__ = os.path.dirname(os.path.abspath(__file__))
"""the directory of the file"""
__data_dir__ = os.path.realpath(os.path.join(__dir__, "..", "src", "incentive", "data", "handler2019"))

data_file = os.path.join(__data_dir__, "handler_2019_parameters_exploration_200.npz")


def main(*args):

    recompute_correlation = True
    erca_w = 0
    camp_w = 0
    plas_w = 10

    tau_short = np.linspace(1, 200, 200, endpoint=True)  # 60
    tau_long = np.linspace(1, 200, 200, endpoint=True)  # 106

    camp_time, camp_data = load_means('cAMP')
    camp_time += camp_time_adjust
    erca_time, erca_data = load_means('ERGCaMP')
    erca_time += erca_time_adjust

    stats = load_statistics()
    plasticity_ave = stats["plasticity"]["mean"]
    plasticity_sem = stats["plasticity"]["sem"]
    erca_ave = stats["ERGCaMP"]["mean"]
    erca_sem = stats["ERGCaMP"]["sem"]
    camp_ave = stats["cAMP"]["mean"]
    camp_sem = stats["cAMP"]["sem"]

    uss_on = [-6., -1.2, -.6, 0., .5, 6.]

    str_long = r"$\tau_{long}$"
    str_short = r"$\tau_{short}$"

    time = np.linspace(-7, 8, 1001)
    for i, us_on in enumerate(uss_on):
        erca_d = 0.8 * np.interp(time, erca_time, erca_data[i], left=np.nan)
        camp_d = 0.4 * np.interp(time, camp_time, camp_data[i], left=np.nan)

        ts_camp = stim['cAMP'][conditions[i]][0] + camp_time_adjust
        te_camp = stim['cAMP'][conditions[i]][1] + camp_time_adjust
        ts_erca = stim['ERGCaMP'][conditions[i]][0] + erca_time_adjust
        te_erca = stim['ERGCaMP'][conditions[i]][1] + erca_time_adjust
        camp_time_window = np.all([ts_camp <= time, time < te_camp], axis=0)
        erca_time_window = np.all([ts_erca <= time, time < te_erca], axis=0)

        erca_ave[i] = np.nanmean(-erca_d[erca_time_window])
        camp_ave[i] = np.nanmean(camp_d[camp_time_window])

    if os.path.exists(data_file):
        data = np.load(data_file)
        plasticity = data["plasticity"]
        camp = data["erca"]
        erca = data["camp"]
        correlation_r = data["r"]
        correlation_p = data["p"]

        if not recompute_correlation:
            for d1, tau_s in enumerate(tau_short[::2]):
                for d2, tau_l in enumerate(tau_long[::2]):
                    print(f"tau_short = {tau_s:.0f}, tau_long: {tau_l:.0f} -- ", end="")
                    print(f"Correlation-coefficient: r={correlation_r[d1, d2]:.2f}, p={correlation_p[d1, d2]:E}")

        print(f"Data loaded from file: '{data_file}'.")
    else:
        print(f"File does not exist: '{data_file}'.")

        camp = np.zeros((tau_short.shape[0], tau_long.shape[0], len(uss_on)), dtype=float)
        erca = np.zeros((tau_short.shape[0], tau_long.shape[0], len(uss_on)), dtype=float)
        plasticity = np.zeros((tau_short.shape[0], tau_long.shape[0], len(uss_on)), dtype=float)
        correlation_r = np.zeros((tau_short.shape[0], tau_long.shape[0]), dtype=float)
        correlation_p = np.zeros((tau_short.shape[0], tau_long.shape[0]), dtype=float)

        for d1, tau_s in enumerate(tau_short):
            for d2, tau_l in enumerate(tau_long):
                for i, us_on in enumerate(uss_on):
                    _, _, _, _, _, _, m, w, dR1, dR2 = run_case(us_on, tau_short=tau_s, tau_long=tau_l)
                    dR1 = np.array(dR1)
                    dR2 = np.array(dR2)

                    ts_camp = stim['cAMP'][conditions[i]][0] + camp_time_adjust
                    te_camp = stim['cAMP'][conditions[i]][1] + camp_time_adjust
                    ts_erca = stim['ERGCaMP'][conditions[i]][0] + erca_time_adjust
                    te_erca = stim['ERGCaMP'][conditions[i]][1] + erca_time_adjust
                    camp_time_window = np.all([ts_camp <= time, time < te_camp], axis=0)
                    erca_time_window = np.all([ts_erca <= time, time < te_erca], axis=0)

                    erca[d1, d2, i] = np.nanmean(-dR1[erca_time_window])
                    camp[d1, d2, i] = np.nanmean(dR2[camp_time_window])
                z_erca = np.maximum(np.nanmax(erca[d1, d2]) - np.nanmin(erca[d1, d2]), np.finfo(float).eps)
                z_camp = np.maximum(np.nanmax(camp[d1, d2]) - np.nanmin(camp[d1, d2]), np.finfo(float).eps)
                erca[d1, d2] = (erca[d1, d2] - np.nanmin(erca[d1, d2])) / z_erca
                camp[d1, d2] = (camp[d1, d2] - np.nanmin(camp[d1, d2])) / z_camp
                plasticity[d1, d2] = erca[d1, d2] - camp[d1, d2]

                # # this would work if we try to model the absolute values
                # feats = np.r_[erca[d1, d2], camp[d1, d2], plasticity[d1, d2]]
                # correlation_r[d1, d2], correlation_p[d1, d2] = pearsonr(feats, target)

                # we try to model the differences between the different scenarios, but not the absolute correlation
                # amongst the ER-Ca, cAMP and plasticity effects
                erca_r, erca_p = pearsonr(erca[d1, d2], erca_ave)
                camp_r, camp_p = pearsonr(camp[d1, d2], camp_ave)
                plas_r, plas_p = pearsonr(plasticity[d1, d2], plasticity_ave)
                correlation_r[d1, d2] = (erca_w * erca_r + camp_w * camp_r + plas_w * plas_r) / (erca_w + camp_w + plas_w)
                correlation_p[d1, d2] = (erca_w * erca_p + camp_w * camp_p + plas_w * plas_p) / (erca_w + camp_w + plas_w)

                print(f"tau_short = {tau_s:.0f}, tau_long: {tau_l:.0f} -- ", end="")
                print(f"Correlation-coefficient: r={correlation_r[d1, d2]:.2f}, p={correlation_p[d1, d2]:E}")

    # max_plus = np.absolute(plasticity.max(axis=-1))[..., None]
    # max_minus = np.absolute(plasticity.min(axis=-1))[..., None]
    # max_plus = np.concatenate([max_plus] * plasticity.shape[-1], axis=-1)
    # max_minus = np.concatenate([max_minus] * plasticity.shape[-1], axis=-1)
    # plasticity *= 0.8
    #
    # plasticity[plasticity > 0] *= 0.6 / max_plus[plasticity > 0]
    # plasticity[plasticity < 0] *= 0.8 / max_minus[plasticity < 0]

    if recompute_correlation and os.path.exists(data_file):
        for d1, tau_s in enumerate(tau_short):
            for d2, tau_l in enumerate(tau_long):
                if np.any(~np.isnan(plasticity[d1, d2])):
                    # feats = np.r_[erca[d1, d2], camp[d1, d2]]
                    # correlation_r[d1, d2], correlation_p[d1, d2] = pearsonr(feats, target)

                    erca_r, erca_p = pearsonr(erca[d1, d2], erca_ave)
                    camp_r, camp_p = pearsonr(camp[d1, d2], camp_ave)
                    plas_r, plas_p = pearsonr(plasticity[d1, d2], plasticity_ave)
                    correlation_r[d1, d2] = (erca_w * erca_r + camp_w * camp_r + plas_w * plas_r) / (erca_w + camp_w + plas_w)
                    correlation_p[d1, d2] = (erca_w * erca_p + camp_w * camp_p + plas_w * plas_p) / (erca_w + camp_w + plas_w)

                print(f"tau_short = {tau_s:.0f}, tau_long: {tau_l:.0f} -- ", end="")
                print(f"Correlation-coefficient: r={correlation_r[d1, d2]:.2f}, p={correlation_p[d1, d2]:E}")

    i_max = np.nanargmax(correlation_r)
    d1_max = i_max // correlation_r.shape[0]
    d2_max = i_max % correlation_r.shape[0]
    # d1_max = 75
    # d2_max = 146
    print(f"Best tau_long = {d2_max + 1}, best tau_short = {d1_max + 1}, r = {correlation_r[d1_max, d2_max]:.2f}, "
          f"p = {correlation_p[d1_max, d2_max]:E}")

    if not os.path.exists(data_file):
        print(f"Saving to file: '{data_file}'.")

        np.savez_compressed(data_file, plasticity=plasticity, erca=erca, camp=camp, r=correlation_r, p=correlation_p)

    off1 = d1_max % 2
    off2 = d2_max % 2
    freq = 2
    # plt.figure(f'tau_short exploration - tau_long={d2_max + 1}', figsize=(10, 15))
    # for d1, tau_s in enumerate(tau_short[off1::freq]):
    #     if d1 >= 100:
    #         break
    #     plt.subplot(10, 10, d1+1)
    #     plt.plot([-3, -1.2, -.6, 0, .5, 3], plasticity[off1 + d1 * freq, d2_max], 'ks-', markerfacecolor="white")
    #     plt.errorbar([-3, -1.2, -.6, 0, .5, 3], plasticity_ave, yerr=plasticity_sem, fmt='grey', capsize=3)
    #     if (d1 + 1) // 10 >= 9:
    #         plt.xticks([-3, -2, -1, 0, 1, 2, 3], [-6, -2, -1, 0, 1, 2, 6])
    #     else:
    #         plt.xticks([-3, -2, -1, 0, 1, 2, 3], ["", "", "", "", "", "", ""])
    #     if d1 % 10 == 0:
    #         plt.yticks([-1, 0, 1])
    #     else:
    #         plt.yticks([-1, 0, 1], ["", "", ""])
    #     plt.ylim(-1, 1)
    #     plt.xlabel(f"{str_short}={tau_s:.0f}")
    # plt.tight_layout()
    #
    # plt.figure(f'tau_long exploration - tau_short = {d1_max + 1}', figsize=(10, 15))
    # for d2, tau_l in enumerate(tau_long[off2::freq]):
    #     if d2 >= 100:
    #         break
    #     plt.subplot(10, 10, d2+1)
    #     plt.plot([-3, -1.2, -.6, 0, .5, 3], plasticity[d1_max, off2 + d2 * freq], 'ks-', markerfacecolor="white")
    #     plt.errorbar([-3, -1.2, -.6, 0, .5, 3], plasticity_ave, yerr=plasticity_sem, fmt='grey', capsize=3)
    #     if (d2 + 1) // 10 >= 9:
    #         plt.xticks([-3, -2, -1, 0, 1, 2, 3], [-6, -2, -1, 0, 1, 2, 6])
    #     else:
    #         plt.xticks([-3, -2, -1, 0, 1, 2, 3], ["", "", "", "", "", "", ""])
    #     if d2 % 10 == 0:
    #         plt.yticks([-1, 0, 1])
    #     else:
    #         plt.yticks([-1, 0, 1], ["", "", ""])
    #     plt.ylim(-1, 1)
    #     plt.xlabel(f"{str_long}={tau_l:.0f}")
    # plt.tight_layout()

    plt.figure('Correlation matrix', figsize=(5, 2.5))

    lim_show = 150

    plt.subplot(1, 2, 1)
    plt.imshow(correlation_r[:lim_show, :lim_show], vmin=-1, vmax=1, cmap="coolwarm", origin="lower")
    plt.scatter(d2_max, d1_max, marker='x', color='k')
    plt.ylabel(r"$\tau_{short}$")
    plt.xlabel(r"$\tau_{long}$")
    plt.yticks([0, d1_max-1, lim_show-1],
               [int(tau_short[0]), int(tau_short[d1_max]), int(tau_short[lim_show-1])])
    plt.xticks([0, d2_max-1, lim_show-1],
               [int(tau_long[0]), int(tau_long[d2_max]), int(tau_long[lim_show-1])])
    plt.ylim(-0.5, 149.5)
    plt.xlim(-0.5, 149.5)
    plt.title("Pearson r")

    plt.subplot(1, 2, 2)
    plt.imshow(correlation_p[:150, :150], vmin=-2e-01, vmax=2e-01, cmap="coolwarm", origin="lower")
    plt.scatter(d2_max, d1_max, marker='x', color='k')
    plt.xlabel(r"$\tau_{long}$")
    plt.yticks([0, d1_max-1, lim_show-1], ["", "", ""])
    plt.xticks([0, d2_max-1, lim_show-1],
               [int(tau_long[0]), int(tau_long[d2_max]), int(tau_long[lim_show-1])])
    plt.ylim(-0.5, 149.5)
    plt.xlim(-0.5, 149.5)
    plt.title("Pearson p")

    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    import sys

    main(*sys.argv)
