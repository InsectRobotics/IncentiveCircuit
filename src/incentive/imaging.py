from typing import List

import yaml
import pandas as pd
import numpy as np
import os
import re
import csv

# the directory of the file
__dir__ = os.path.dirname(os.path.abspath(__file__))
# the directory of the data
__data_dir__ = os.path.realpath(os.path.join(__dir__, "../..", "data", "FruitflyMB"))
# sub-directories of each of the experiments
__dirs = {
    'B+': ''
}
# pattern of the files for each of the experiments
_patterns_ = {
    # pattern for the initial data
    'B+': r'realSCREEN_([\d\w\W]+)\.xlsx_finaldata([\w\W]+)_timepoint(\d)\.csv'
}
# load the meta-data of the genotypes and neurons from the file
with open(os.path.join(__data_dir__, 'meta.yaml'), 'rb') as f:
    _meta_ = yaml.load(f, Loader=yaml.BaseLoader)


def load_data(experiments='B+', directory=None):
    """
    Creates a DataFrame containing all the data from the specified experiments with keys in this order:
    1. {experiment}
    2. {name}_{genotype}
    3. {#trial}

    :param experiments: (optional) list of names (or single name) of the experiments to load. Default is 'B+'.
    :type experiments: List[str] | str
    :return: a DataFrame with the data from the experiments requested
    """

    if directory is None:
        directory = __data_dir__

    # Convert the argument of the function to a list of names
    if isinstance(experiments, str):
        if experiments == 'all':
            experiments = _patterns_.keys()
        else:
            experiments = [experiments]

    data = {}
    # Load the data for every experiment requested
    for experiment in experiments:
        experiment_dir = os.path.join(directory, __dirs[experiment])
        # for each file in the directory of the experiment
        for fname in os.listdir(experiment_dir):
            # get the details from the name of the file
            details = re.findall(_patterns_[experiment], fname)
            if len(details) == 0:
                # if the name of the file does not follow the correct pattern then skip it
                continue

            temp = details[0]
            # assign the details extracted from the file-name to the appropriate variables
            if len(temp) > 3:
                _, genotype, odour, trial = temp[:4]
            elif len(temp) > 2:
                genotype, odour, trial = temp
            else:
                print('Information in the filename is not sufficient: %s' % fname)
                print('Skipping file!')
                continue

            trial = int(trial)  # transform the trial number from string to integer

            timepoint = None
            fname = os.path.join(experiment_dir, fname)  # get the absolute path
            with open(fname, 'r') as csvfile:
                # read the CSV file
                reader = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_NONNUMERIC)
                # create a matrix where each row is a time-point and each column is a different fly
                for row in reader:
                    if timepoint is None:
                        timepoint = row
                    else:
                        timepoint = np.vstack([timepoint, row])  # {time-point} x {datapoint}

            # specify if odour is odour A (OCT) or B (MCH)
            a, b = "O" in odour or "A" in odour or "B" in odour, "M" in odour
            # identify if the odour is CS+ depending on the experiment
            csp = "B" in experiment and b or "A" in experiment and a

            if experiment not in data:
                # initialise the data-dict for the experiment
                data[experiment] = {}
            if genotype not in data[experiment]:
                # initialise the genotype for the experiment
                data[experiment][genotype] = [[]] * 18
            # assign the collected data to the appropriate genotype and trial number
            data[experiment][genotype][2 * (trial - 1) + int(csp)] = timepoint

    # transform the data from [experiment, genotype, trial, time-step, fly]
    # to [experiment, name_genotype, trial, time-step, fly]
    for experiment in data:
        # get all genotypes
        genotypes = list(data[experiment].keys())
        for genotype in genotypes:
            temp = []
            # get and remove from the data-set the data for the specific genotype
            gdata = data[experiment].pop(genotype)
            # get the prefix and compartment of the specific genotype from the meta-data
            if genotype in _meta_ and "MBON" in _meta_[genotype]['type']:
                name = "MBON-%s" % _meta_[genotype]['name']
            elif genotype in _meta_ and _meta_[genotype]['type'] in ["PAM", "PPL1"]:
                name = "%s-%s" % (_meta_[genotype]['type'], _meta_[genotype]['name'])
            else:
                # if the genotype does not exist in the meta-data then skip it
                continue

            if name in data[experiment]:
                # resolve conflicting data with the same name (compartment)
                name += "_1"
            # find a trial with with data
            for t in range(len(gdata)):
                if len(gdata[t]) > 0:
                    temp = gdata[t]
                    break
            # fill the data of every empty trial with zeros
            for t in range(len(gdata)):
                if len(gdata[t]) == 0:
                    gdata[t] = np.zeros_like(temp)

            # add a new entry in the data-set with the cleaned data and the {name}_{genotype} as key
            data[experiment][name+"_"+genotype] = pd.DataFrame(np.concatenate(gdata))

    # transform the dictionary into a pandas DaraFrame
    return pd.DataFrame(data)


def plot_phase_overlap_mean_responses_from_data(data, experiment="B+", nids=None, only_nids=True, figsize=None):
    """
    Plots the average responses of the neurons per phase/trial for a specific experiment with overlapping phases.

    :param data: The DataFrame with the responses
    :type data: pd.DataFrame
    :param experiment: (optional) The experiment whose data we want to plot. Default is 'B+'.
    :type experiment: str
    :param nids: (optional) List of neuron IDs that we want to show their name. If None, it shows all the names.
    :type nids: List[int]
    :param only_nids: If to plot only the responses of the specified neurons. Default is True.
    :type only_nids: bool
    """
    import matplotlib.pyplot as plt

    title = "individuals-from-data"

    data_exp = data[experiment]
    genotypes = np.sort(data_exp.index)
    odour_a_xs = np.array([np.arange(28, 43) + i * 200 for i in range(9)])
    shock_a_xs = np.array([np.arange(44, 49) + i * 200 for i in range(9)])
    odour_b_xs = np.array([np.arange(28, 43) + i * 200 + 100 for i in range(8)])
    shock_b_xs = np.array([np.arange(44, 49) + i * 200 + 100 for i in range(8)])
    xa = np.arange(18)
    xb = xa + 1

    if nids is None:
        nids = np.arange(len(genotypes))
    if only_nids:
        genotypes = genotypes[nids]

    ymin, ymax = 0, 2
    y_lim = [ymin - .1, ymax + .1]

    nb_genotypes = len(genotypes)
    nb_plots = nb_genotypes * 2
    subs = []

    nb_rows = 4
    nb_cols = nb_plots // nb_rows
    while nb_cols > 7:
        nb_rows += 4
        nb_cols = nb_plots // nb_rows + 1

    if figsize is None:
        figsize = (8 - 2 * int(not only_nids), nb_rows + 1)
    plt.figure(title, figsize=figsize)

    xticks_b = 2 * np.arange(10) + 4
    xticks_a = xticks_b.copy() - 1
    xticks_a[5:] += 2

    for j, genotype in enumerate(genotypes):

        odour_a_mean = np.nanmean(np.array(data_exp[genotype])[odour_a_xs], axis=(1, 2))
        odour_a_std = np.nanstd(np.array(data_exp[genotype])[odour_a_xs], axis=(1, 2)) / 2
        shock_a_mean = np.nanmean(np.array(data_exp[genotype])[shock_a_xs], axis=(1, 2))
        shock_a_std = np.nanstd(np.array(data_exp[genotype])[shock_a_xs], axis=(1, 2)) / 2

        odour_b_mean = np.nanmean(np.array(data_exp[genotype])[odour_b_xs], axis=(1, 2))
        odour_b_std = np.nanstd(np.array(data_exp[genotype])[odour_b_xs], axis=(1, 2)) / 2
        shock_b_mean = np.nanmean(np.array(data_exp[genotype])[shock_b_xs], axis=(1, 2))
        shock_b_std = np.nanstd(np.array(data_exp[genotype])[shock_b_xs], axis=(1, 2)) / 2

        data_a_mean = np.array([odour_a_mean, shock_a_mean]).T.reshape((-1,))
        data_a_std = np.array([odour_a_std, shock_a_std]).T.reshape((-1,))
        data_b_mean = np.array([odour_b_mean, shock_b_mean]).T.reshape((-1,))
        data_b_std = np.array([odour_b_std, shock_b_std]).T.reshape((-1,))

        z = np.maximum(np.max(data_a_mean + data_a_std), np.max(data_b_mean + data_b_std)) / 2

        data_a_mean /= z
        data_a_std /= z
        a_col = np.array([.5 * 205, .5 * 222, 238]) / 255.

        if len(subs) <= j:
            axa = plt.subplot(nb_rows, nb_cols, 2 * (j // nb_cols) * nb_cols + j % nb_cols + 1)
            axa.set_xticks(xticks_a)
            axa.set_yticks([0, ymax/2, ymax])
            axa.set_ylim(y_lim)
            axa.set_xlim([2, 24])
            axa.tick_params(labelsize=8)
            axa.set_xticklabels(["" for _ in xticks_a])
            axa.set_title(r"$%s$" % genotype, fontsize=8)
            if j % nb_cols == 0:
                axa.set_ylabel("Odour A", fontsize=8)
            else:
                axa.set_yticklabels([""] * 3)
                axa.spines['left'].set_visible(False)
                axa.set_yticks([])
            axa.spines['top'].set_visible(False)
            axa.spines['right'].set_visible(False)

            a_acol = np.array([205, 222, 238]) / 255.
            axa.fill_between(xa[2:12], data_a_mean[2:12] - data_a_std[2:12], data_a_mean[2:12] + data_a_std[2:12],
                             color=a_acol, alpha=0.2)
            axa.plot(xa[:3], data_a_mean[:3], color=(.8, .8, .8), lw=2)
            axa.plot(xa[2:12], data_a_mean[2:12], color=a_acol, lw=2, label="acquisition")
            subs.append(axa)
        subs[-1].fill_between(xa[14:], data_a_mean[14:] - data_a_std[14:], data_a_mean[14:] + data_a_std[14:],
                              color=a_col, alpha=0.2)
        subs[-1].plot(xa[11:15], data_a_mean[11:15], color=(.8, .8, .8), lw=2)
        subs[-1].plot(xa[14:], data_a_mean[14:], color=a_col, lw=2, label="reversal")
        subs[-1].plot([15, 17], data_a_mean[[15, 17]], 'r.')

    for j, genotype in enumerate(genotypes):

        odour_a_mean = np.nanmean(np.array(data_exp[genotype])[odour_a_xs], axis=(1, 2))
        odour_a_std = np.nanstd(np.array(data_exp[genotype])[odour_a_xs], axis=(1, 2)) / 2
        shock_a_mean = np.nanmean(np.array(data_exp[genotype])[shock_a_xs], axis=(1, 2))
        shock_a_std = np.nanstd(np.array(data_exp[genotype])[shock_a_xs], axis=(1, 2)) / 2

        odour_b_mean = np.nanmean(np.array(data_exp[genotype])[odour_b_xs], axis=(1, 2))
        odour_b_std = np.nanstd(np.array(data_exp[genotype])[odour_b_xs], axis=(1, 2)) / 2
        shock_b_mean = np.nanmean(np.array(data_exp[genotype])[shock_b_xs], axis=(1, 2))
        shock_b_std = np.nanstd(np.array(data_exp[genotype])[shock_b_xs], axis=(1, 2)) / 2

        data_a_mean = np.array([odour_a_mean, shock_a_mean]).T.reshape((-1,))
        data_a_std = np.array([odour_a_std, shock_a_std]).T.reshape((-1,))
        data_b_mean = np.array([odour_b_mean, shock_b_mean]).T.reshape((-1,))
        data_b_std = np.array([odour_b_std, shock_b_std]).T.reshape((-1,))

        z = np.maximum(np.max(data_a_mean + data_a_std), np.max(data_b_mean + data_b_std)) / 2

        data_b_mean /= z
        data_b_std /= z
        b_col = np.array([255, .5 * 197, .5 * 200]) / 255.

        jn = j + (nb_rows * nb_cols) // 2

        if len(subs) <= jn:
            axb = plt.subplot(nb_rows, nb_cols, (2 * (j // nb_cols) + 1) * nb_cols + j % nb_cols + 1)
            axb.set_xticks(xticks_b)
            axb.set_yticks([0, ymax/2, ymax])
            axb.set_ylim(y_lim)
            axb.set_xlim([2, 24])
            axb.tick_params(labelsize=8)
            axb.set_xticklabels(["%s" % (i + 1) for i in range(5)] * 2)
            if jn % nb_cols == 0:
                axb.set_ylabel("Odour B", fontsize=8)
                axb.text(-6, -.8, "Trial #", fontsize=8)
            else:
                axb.set_yticklabels([""] * 3)
                axb.spines['left'].set_visible(False)
                axb.set_yticks([])

            axb.spines['top'].set_visible(False)
            axb.spines['right'].set_visible(False)

            b_acol = np.array([255, 197, 200]) / 255.

            axb.fill_between(xb[2:12], data_b_mean[2:12] - data_b_std[2:12], data_b_mean[2:12] + data_b_std[2:12],
                             color=b_acol, alpha=0.2)
            axb.plot(xb[:3], data_b_mean[:3], color=(.8, .8, .8), lw=2)
            axb.plot(xb[2:12], data_b_mean[2:12], color=b_acol, lw=2, label="acquisition")

            subs.append(axb)

        subs[-1].fill_between(xb[12:16], data_b_mean[12:] - data_b_std[12:], data_b_mean[12:] + data_b_std[12:],
                              color=b_col, alpha=0.2)
        subs[-1].plot(xb[11:13], data_b_mean[11:13], color=(.8, .8, .8), lw=2)
        subs[-1].plot(xb[12:16], data_b_mean[12:], color=b_col, lw=2, label="reversal")
        subs[-1].plot(xb[[3, 5, 7, 9, 11]], data_b_mean[[3, 5, 7, 9, 11]], 'r.')

    subs[len(subs)//2 - 1].legend(fontsize=8, bbox_to_anchor=(1.05, 1.35), loc='upper left',
                                  framealpha=0., labelspacing=1.)
    subs[-1].legend(fontsize=8, bbox_to_anchor=(1.05, 1.35), loc='upper left', framealpha=0., labelspacing=1)

    # subs[len(subs)//2 - 1].legend(fontsize=8, bbox_to_anchor=(1.05, 1.), loc='upper left',
    #                               framealpha=0., labelspacing=1.)
    # subs[-1].legend(fontsize=8, bbox_to_anchor=(1.05, 1.), loc='upper left', framealpha=0., labelspacing=1)
    plt.tight_layout()
    plt.show()
