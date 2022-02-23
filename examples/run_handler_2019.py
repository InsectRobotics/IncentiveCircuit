__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright 2021, School of Informatics, the University of Edinburgh"
__licence__ = "MIT"
__version__ = "1.1-alpha"
__maintainer__ = "Evripidis Gkanias"
__email__ = "ev.gkanias@ed.ac.uk"
__status__ = "Production"

from incentive.handler import run_case, load_means, load_statistics
from incentive.handler import camp_time_adjust, erca_time_adjust, conditions, stim

from scipy.stats import pearsonr

import numpy as np
import matplotlib.pyplot as plt


def main(*args):

    ercas = []
    camps = []

    uss_on = [-6., -1.2, -.6, 0., .5, 6.]

    camp_time, camp_data = load_means('cAMP')
    camp_time += camp_time_adjust
    erca_time, erca_data = load_means('ERGCaMP')
    erca_time += erca_time_adjust

    for i, us_on in enumerate(uss_on):
        time, cs, us, k, d1, d2, m, w, dR1, dR2 = run_case(us_on, tau_short=60, tau_long=104)
        time = np.array(time)

        plt.figure('learning-rule', figsize=(7, 5))

        plt.subplot(6, 6, i + 1)
        plt.title('%.1f s' % us_on, fontsize=8)
        plt.plot([0, 0], [-.05, 1.1], 'k:')
        plt.plot([0], [-.05], 'k', marker=6)
        plt.plot([us_on, us_on], [-.05, 1.1], 'r:')
        plt.plot([us_on], [-.05], 'r', marker=6)
        plt.plot(time, cs, 'k')
        if i == 0:
            plt.text(-4, .5, 'CS', fontsize=8, rotation=0)
        plt.ylim(-.1, 1.1)
        plt.xlim(-7, 8)
        plt.axis('off')

        plt.subplot(6, 6, i + 7)
        plt.plot([0, 0], [-.05, 1.1], 'k:')
        plt.plot([0], [-.05], 'k', marker=6)
        plt.plot([us_on, us_on], [-.05, 1.1], 'r:')
        plt.plot([us_on], [-.05], 'r', marker=6)
        plt.plot(time, us, 'r')
        if i == 0:
            plt.text(-4, .5, 'US', fontsize=8, rotation=0)
        plt.ylim(-.1, 1.1)
        plt.xlim(-7, 8)
        plt.axis('off')

        plt.subplot(6, 6, i + 13)
        plt.plot([0, 0], [-.05, 1.1], 'k:')
        plt.plot([0], [-.05], 'k', marker=6)
        plt.plot([us_on, us_on], [-.05, 1.1], 'r:')
        plt.plot([us_on], [-.05], 'r', marker=6)
        plt.plot(time, k, 'k')
        if i == 0:
            plt.text(-4, .5, 'KC', fontsize=8, rotation=0)
        plt.ylim(-.1, 1.1)
        plt.xlim(-7, 8)
        plt.axis('off')

        plt.subplot(6, 6, i + 19)
        plt.plot([0, 0], [-.35, .8], 'k:')
        plt.plot([0], [-.35], 'k', marker=6)
        plt.plot([us_on, us_on], [-.35, .8], 'r:')
        plt.plot([us_on], [-.35], 'r', marker=6)
        plt.plot(time, d1, 'm', label='D1')
        plt.plot(time, d2, 'orange', label='D2')
        plt.plot(time, 10 * (np.array(d2) - np.array(d1)), 'grey', alpha=.5, label='D2-D1')
        if i == 0:
            plt.text(-4, .8, 'dopamine', fontsize=8, rotation=0)
        plt.ylim(-.4, .8)
        plt.xlim(-7, 8)
        plt.axis('off')

        plt.subplot(6, 6, i + 25)
        plt.plot([0, 0], [-.1, 2.2], 'k:')
        plt.plot([0], [-.1], 'k', marker=6)
        plt.plot([us_on, us_on], [-.1, 2.2], 'r:')
        plt.plot([us_on], [-.1], 'r', marker=6)
        plt.plot(time, m, 'b')
        if i == 0:
            plt.text(-4, 1.5, 'MBON', fontsize=8, rotation=0)
        plt.ylim(-.2, 2.2)
        plt.xlim(-7, 8)
        plt.axis('off')

        plt.subplot(6, 6, i + 31)
        plt.plot([0, 0], [-.1, 2.2], 'k:')
        plt.plot([0], [-.1], 'k', marker=6)
        plt.plot([us_on, us_on], [-.1, 2.2], 'r:')
        plt.plot([us_on], [-.1], 'r', marker=6)
        plt.plot(time, w, 'g')
        if i == 0:
            plt.text(-4, 1.5, r'$W_{k2m}$', fontsize=8, rotation=0)
        plt.ylim(-.2, 2.2)
        plt.xlim(-7, 8)
        plt.axis('off')

        plt.tight_layout()

        plt.figure("dopamine-function", figsize=(7, 3))

        plt.subplot(3, 6, i + 1)
        plt.plot([0, 0], [-.09, .1], 'k:')
        plt.plot([0], [-.09], 'k', marker=6)
        plt.plot([us_on, us_on], [-.09, .1], 'r:')
        plt.plot([us_on], [-.09], 'r', marker=6)
        # camp = np.interp(time, camp_time, np.array(camp_data[i]) / np.max(np.absolute(camp_data)))
        camp = 0.4 * np.interp(time, camp_time, camp_data[i], left=np.nan)
        dR2 = np.array(dR2)  # / np.max(np.absolute(dR2))
        plt.plot(time, camp, 'grey')
        plt.plot(time, dR2, 'k')
        if i == 0:
            plt.text(-4, .05, 'cAMP\nDopR2', fontsize=8, rotation=0)
        plt.ylim(-.1, .1)
        plt.xlim(-7, 8)
        plt.axis('off')

        plt.subplot(3, 6, i + 7)
        plt.title('%.1f s' % us_on, fontsize=8)
        plt.plot([0, 0], [-.09, .1], 'k:')
        plt.plot([0], [-.09], 'k', marker=6)
        plt.plot([us_on, us_on], [-.09, .1], 'r:')
        plt.plot([us_on], [-.09], 'r', marker=6)
        # erca = np.interp(time, x_erca, np.array(erca_data[i]) / np.max(np.absolute(erca_data)))
        erca = 0.8 * np.interp(time, erca_time, erca_data[i], left=np.nan)
        dR1 = np.array(dR1)  # / np.max(np.absolute(dR1))
        plt.plot(time, erca, 'grey')
        plt.plot(time, dR1, 'k')
        if i == 0:
            plt.text(-4, .05, 'ER-GCaMP\nDopR1', fontsize=8, rotation=0)
        plt.ylim(-.1, .1)
        plt.xlim(-7, 8)
        plt.axis('off')

        plt.subplot(3, 6, i + 13)
        plt.plot([0, 0], [-.09, .1], 'k:')
        plt.plot([0], [-.09], 'k', marker=6)
        plt.plot([us_on, us_on], [-.09, .1], 'r:')
        plt.plot([us_on], [-.09], 'r', marker=6)
        plt.plot(time, -erca - camp, 'grey')
        plt.plot(time, -dR1 - dR2, 'k')
        if i == 0:
            plt.text(-4, .05, 'effect', fontsize=8, rotation=0)
        plt.ylim(-.1, .1)
        plt.xlim(-7, 8)
        plt.axis('off')
        plt.tight_layout()

        ts_camp = stim['cAMP'][conditions[i]][0] + camp_time_adjust
        te_camp = stim['cAMP'][conditions[i]][1] + camp_time_adjust
        ts_erca = stim['ERGCaMP'][conditions[i]][0] + erca_time_adjust
        te_erca = stim['ERGCaMP'][conditions[i]][1] + erca_time_adjust
        camp_time_window = np.all([ts_camp <= time, time < te_camp], axis=0)
        erca_time_window = np.all([ts_erca <= time, time < te_erca], axis=0)

        ercas.append(np.nanmean(-dR1[erca_time_window]))
        camps.append(np.nanmean(dR2[camp_time_window]))

    z_erca = np.nanmax(ercas) - np.nanmin(ercas)
    z_camp = np.nanmax(camps) - np.nanmin(camps)
    erca_norm = (np.array(ercas) - np.nanmin(ercas)) / z_erca
    camp_norm = (np.array(camps) - np.nanmin(camps)) / z_camp

    plasticity = erca_norm - camp_norm

    # load data statistics from EXCEL
    stats = load_statistics()
    plasticity_ave, plasticity_sem = stats["plasticity"]["mean"], stats["plasticity"]["sem"]

    plt.figure('plasticity', figsize=(2, 3))
    plt.plot([-3, -1.2, -.6, 0, .5, 3], plasticity, 'ks-', markerfacecolor="white")
    plt.errorbar([-3, -1.2, -.6, 0, .5, 3], plasticity_ave, yerr=plasticity_sem, fmt='grey', capsize=3)
    plt.xticks([-3, -2, -1, 0, 1, 2, 3], [-6, -2, -1, 0, 1, 2, 6])
    plt.yticks([-1, 0, 1])
    plt.ylim(-1, 1)

    r, p = pearsonr(plasticity, plasticity_ave)
    print(r, p)
    print(f"Correlation-coefficient: r={r:.2f}, p={p:E}")

    plt.show()


if __name__ == '__main__':
    import sys

    main(*sys.argv)
