"""
Package that handles the data collected from the real neurons in the fruit fly mushroom body.
"""

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2021, Insect Robotics Group," \
                "Institute of Perception, Action and Behaviour," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "GPLv3+"
__version__ = "v1.1-alpha"
__maintainer__ = "Evripidis Gkanias"

import yaml
import pandas as pd
import numpy as np
import os
import re
import csv

__dir__ = os.path.dirname(os.path.abspath(__file__))
"""the directory of the file"""
__data_dir__ = os.path.realpath(os.path.join(__dir__, "data", "fruitfly"))
"""the directory of the data"""
__dirs = {
    'B+': ''
}
"""sub-directories of each of the experiments"""
_patterns_ = {
    # pattern for the initial data
    'B+': r'realSCREEN_([\d\w\W]+)\.xlsx_finaldata([\w\W]+)_timepoint(\d)\.csv'
}
"""pattern of the files for each of the experiments"""

with open(os.path.join(__data_dir__, 'meta.yaml'), 'rb') as f:
    _meta_ = yaml.load(f, Loader=yaml.BaseLoader)
    """load the meta-data of the genotypes and neurons from the file"""


def load_data(experiments='B+', directory=None):
    """
    Creates a DataFrame containing all the data from the specified experiments with keys in this order:
    - {experiment}
    - {name}_{genotype}
    - {#trial}

    Parameters
    ----------
    experiments: str, list
        list of names (or single name) of the experiments to load. Default is 'B+'.
    directory: str
        the directory where the experiments will be found. Default is the data directory.

    Returns
    -------
    data: pd.DataFrame
        a DataFrame with the data from the experiments requested
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


def get_summarised_responses(data, experiment="B+", nids=None, only_nids=True):

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

    xticks_b = 2 * np.arange(10) + 4
    xticks_a = xticks_b.copy() - 1
    xticks_a[5:] += 2

    responses = {}

    for genotype in genotypes:

        qa, qb = {}, {}
        for quantile in [.25, .50, .75]:
            for q, xs in zip([qa, qb], [[odour_a_xs, shock_a_xs], [odour_b_xs, shock_b_xs]]):
                q_odour = np.nanquantile(np.array(data_exp[genotype])[xs[0]], quantile, axis=(1, 2))
                q_shock = np.nanquantile(np.array(data_exp[genotype])[xs[1]], quantile, axis=(1, 2))
                q[quantile] = np.array([q_odour, q_shock]).T.reshape((-1,))

        # normalise responses
        z = np.maximum(np.max(qa[.75]), np.max(qb[.75])) / 2
        for q in [qa, qb]:
            for key in q:
                q[key] /= z

        responses[genotype] = {
            "xa": xa,
            "xb": xb,
            "qa25": qa[.25],
            "qa50": qa[.50],
            "qa75": qa[.75],
            "qb25": qb[.25],
            "qb50": qb[.50],
            "qb75": qb[.75]
        }

    return responses


def plot_phase_overlap_mean_responses_from_data(data, experiment="B+", nids=None, only_nids=True, figsize=None,
                                                show_legend=True):
    """
    Plots the average responses of the neurons per phase/trial for a specific experiment with overlapping phases.

    Parameters
    ----------
    data: pd.DataFrame
        the DataFrame with the responses
    experiment: str, optional
        the experiment whose data we want to plot. Default is 'B+'
    nids: list, optional
        list of neuron IDs that we want to show their name. If None, it shows all the names
    only_nids: bool, optional
        whether to plot only the responses of the specified neurons. Default is True
    figsize: tuple, optional
        the size of the figure
    show_legend: bool, optional
        whether to also plot the legend
    """
    import matplotlib.pyplot as plt

    title = "individuals-from-data"

    sum_res = get_summarised_responses(data, experiment=experiment, nids=nids, only_nids=only_nids)
    genotypes = [key for key in sum_res.keys()]

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

        xa = sum_res[genotype]["xa"]
        data_a_q25 = sum_res[genotype]["qa25"]
        data_a_q50 = sum_res[genotype]["qa50"]
        data_a_q75 = sum_res[genotype]["qa75"]

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
            axa.fill_between(xa[2:12], data_a_q25[2:12], data_a_q75[2:12], color=a_acol, alpha=0.2)
            axa.plot(xa[:3], data_a_q50[:3], color=(.8, .8, .8), lw=2)
            axa.plot(xa[2:12], data_a_q50[2:12], color=a_acol, lw=2, label="acquisition")
            subs.append(axa)
        subs[-1].fill_between(xa[14:], data_a_q25[14:], data_a_q75[14:], color=a_col, alpha=0.2)
        subs[-1].plot(xa[11:15], data_a_q50[11:15], color=(.8, .8, .8), lw=2)
        subs[-1].plot(xa[14:], data_a_q50[14:], color=a_col, lw=2, label="reversal")
        subs[-1].plot([15, 17], data_a_q50[[15, 17]], 'r.')

    for j, genotype in enumerate(genotypes):

        xb = sum_res[genotype]["xb"]
        data_b_q25 = sum_res[genotype]["qb25"]
        data_b_q50 = sum_res[genotype]["qb50"]
        data_b_q75 = sum_res[genotype]["qb75"]

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

            axb.fill_between(xb[2:12], data_b_q25[2:12], data_b_q75[2:12], color=b_acol, alpha=0.2)
            axb.plot(xb[:3], data_b_q50[:3], color=(.8, .8, .8), lw=2)
            axb.plot(xb[2:12], data_b_q50[2:12], color=b_acol, lw=2, label="acquisition")

            subs.append(axb)

        subs[-1].fill_between(xb[12:16], data_b_q25[12:], data_b_q75[12:], color=b_col, alpha=0.2)
        subs[-1].plot(xb[11:13], data_b_q50[11:13], color=(.8, .8, .8), lw=2)
        subs[-1].plot(xb[12:16], data_b_q50[12:], color=b_col, lw=2, label="reversal")
        subs[-1].plot(xb[[3, 5, 7, 9, 11]], data_b_q50[[3, 5, 7, 9, 11]], 'r.')

    if show_legend:
        subs[len(subs)//2 - 1].legend(fontsize=8, bbox_to_anchor=(1.05, 1.35), loc='upper left',
                                      framealpha=0., labelspacing=1.)
        subs[-1].legend(fontsize=8, bbox_to_anchor=(1.05, 1.35), loc='upper left', framealpha=0., labelspacing=1)

    # subs[len(subs)//2 - 1].legend(fontsize=8, bbox_to_anchor=(1.05, 1.), loc='upper left',
    #                               framealpha=0., labelspacing=1.)
    # subs[-1].legend(fontsize=8, bbox_to_anchor=(1.05, 1.), loc='upper left', framealpha=0., labelspacing=1)
    plt.tight_layout()
    plt.show()
