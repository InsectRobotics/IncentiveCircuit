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
        q_min = np.minimum(np.min(qa[.25]), np.min(qb[.25])) / 2
        q_max = np.maximum(np.max(qa[.75]), np.max(qb[.75])) / 2
        for q in [qa, qb]:
            for key in q:
                q[key] = (q[key] - q_min) / (q_max - q_min)

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
