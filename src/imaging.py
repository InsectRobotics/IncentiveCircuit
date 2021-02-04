import os
import re
import csv
import yaml
import pandas as pd
import numpy as np

__dir__ = os.path.dirname(os.path.abspath(__file__))
__data_dir__ = os.path.realpath(os.path.join(__dir__, "..", "data", "FruitflyMB"))
__draft_data_dir__ = os.path.realpath(os.path.join(__dir__, "..", "data", "FruitflyMB", "draft"))

__dir = {
    "MBON-γ1ped": {
        "A+": ["MBON-g1ped", "OCT+shock"],
        "B-": ["MBON-g1ped", "MCH+noshock"],
        "B+": ["MBON-g1ped", "MCH+shock"]
    },
    "MBON-γ2α'1": {
        "A+": ["MBON-g2a'1", "OCT+shock"],
        "B-": ["MBON-g2a'1", "MCH+noshock"],
        "B+": ["MBON-g2a'1", "MCH+shock"]
    },
    "MBON-γ5β'2a": {
        # "A": "",
        "B-": ["MBON-g5b'2a", "MCH+noshock"],
        "B+": ["MBON-g5b'2a", "MCH+shock"]
    },
    "PPL1-γ1ped": {
        "A+": ["PPL1-g1ped", "OCT+shock"],
        "B-": ["PPL1-g1ped", "MCH+noshock"],
        "B+": ["PPL1-g1ped", "MCH+shock"],
    },
    "PAM-β'2a": {
        "A-": ["PAM-b'2a", "OCT+noshock"],
        "A+": ["PAM-b'2a", "OCT+shock"],
        "B-": ["PAM-b'2a", "MCH+noshock"],
        "B+": ["PAM-b'2a", "MCH+shock"]
    },
    "PPL1-γ2α'1": {
        "B+": ["PPL1-g2a'1", "MCH+shock (1-9,1-8)"]
    }
}
__dirs = {
    'B+': '',
    'A+': 'SF traces imaging controls',
    'B-': 'SF traces imaging controls',
    'KC': 'neural traces KC sub-compartments'
}
_pattern_ = r'realSCREEN_([\d\w\W]+)\.xlsx_finaldata([\w\W]+)_timepoint(\d)\.csv'
_patterns_ = {
    # pattern for the initial data
    'B+': r'realSCREEN_([\d\w\W]+)\.xlsx_finaldata([\w\W]+)_timepoint(\d)\.csv',
    # pattern for the control data
    'A+': r'(realSCREEN_){0,1}([\d\w\W]+)_O\+S\.xlsx_finaldata([\w\W]+)_timepoint(\d)\.csv',
    # pattern for the no-shock data
    'B-': r'(realSCREEN_){0,1}([\d\w\W]+)_M\+NS\.xlsx_finaldata([\w\W]+)_timepoint(\d)\.csv',
    # pattern for the KC data
    'KC': r'realSCREEN_([\d\w\W]+)\.xlsx_finaldata([\w\W]+)_timepoint(\d)\.csv'
}
with open(os.path.join(__data_dir__, 'meta.yaml'), 'rb') as f:
    _meta_ = yaml.load(f, Loader=yaml.BaseLoader)


def load_data(experiments='B+'):
    if isinstance(experiments, str):
        if experiments == 'all':
            experiments = _patterns_.keys()
        elif experiments == 'draft':
            return load_draft_data()
        else:
            experiments = [experiments]

    data = {}
    for experiment in experiments:
        experiment_dir = os.path.join(__data_dir__, __dirs[experiment])
        for fname in os.listdir(experiment_dir):
            details = re.findall(_patterns_[experiment], fname)
            if len(details) == 0:
                continue

            temp = details[0]
            if len(temp) > 3:
                _, genotype, odour, trial = temp[:4]
            elif len(temp) > 2:
                genotype, odour, trial = temp
            else:
                print('Information in the filename is not sufficient: %s' % fname)
                print('Skipping file!')
                continue

            trial = int(trial)

            timepoint = None
            fname = os.path.join(experiment_dir, fname)
            with open(fname, 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_NONNUMERIC)
                for row in reader:
                    if timepoint is None:
                        timepoint = row
                    else:
                        timepoint = np.vstack([timepoint, row])  # {timepoint} x {datapoint}

            a, b = "O" in odour or "A" in odour or "B" in odour, "M" in odour
            csp = "B" in experiment and b or "A" in experiment and a

            if experiment not in data:
                data[experiment] = {}
            if genotype not in data[experiment]:
                data[experiment][genotype] = [[]] * 18
            data[experiment][genotype][2 * (trial - 1) + int(csp)] = timepoint

    for experiment in data:
        genotypes = list(data[experiment].keys())
        for genotype in genotypes:
            temp = []
            gdata = data[experiment].pop(genotype)
            if genotype in _meta_ and "MBON" in _meta_[genotype]['type']:
                name = "MBON-%s" % _meta_[genotype]['name']
            elif genotype in _meta_ and _meta_[genotype]['type'] in ["PAM", "PPL1"]:
                name = "%s-%s" % (_meta_[genotype]['type'], _meta_[genotype]['name'])
            else:
                continue
            if name in data[experiment]:
                name += "_1"
            for t in range(len(gdata)):
                if len(gdata[t]) > 0:
                    temp = gdata[t]
                    break
            for t in range(len(gdata)):
                if len(gdata[t]) == 0:
                    gdata[t] = np.zeros_like(temp)

            data[experiment][name+"_"+genotype] = pd.DataFrame(np.concatenate(gdata))

    return pd.DataFrame(data)


def load_draft_data():
    data = {}
    for genotype in __dir:
        for experiment in __dir[genotype]:
            if experiment not in data:
                data[experiment] = {genotype: []}
            data[experiment][genotype] = [[]] * 18
            for r, _, flist in os.walk(__draft_data_dir__):
                match = True
                for d in __dir[genotype][experiment]:
                    if d not in r:
                        match = False
                if not match:
                    continue

                labels = re.findall(r'.*\(\d\-(\d),\d\-(\d)\).*', r)
                if len(labels) < 1:
                    print("Unknown directory pattern:", r)
                    continue
                nb_csm, nb_csp = labels[0]
                nb_csm, nb_csp = int(nb_csm), int(nb_csp)

                for filename in flist:
                    details = re.findall(_pattern_, filename)
                    if len(details) < 1:
                        print("Unknown file pattern:", os.path.join(r, filename))
                        continue
                    _, cs, trial = details[0]
                    trial = int(trial)

                    timepoint = None
                    with open(os.path.join(r, filename), 'r') as csvfile:
                        reader = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_NONNUMERIC)
                        for row in reader:
                            if timepoint is None:
                                timepoint = row
                            else:
                                timepoint = np.vstack([timepoint, row])  # {timepoint} x {datapoint}
                    a, b = "O" in cs, "M" in cs
                    csm = "B" in experiment and a or "A" in experiment and b
                    csp = "B" in experiment and b or "A" in experiment and a

                    if csp and nb_csp < 8:
                        trial += 1
                    if csm and nb_csp < 8:
                        trial += 1
                    if csm and nb_csp < 8 and 6 < trial < 9:
                        trial += 1

                    data[experiment][genotype][2 * (trial - 1) + int(csp)] = timepoint

            temp = []
            for t in range(len(data[experiment][genotype])):
                if len(data[experiment][genotype][t]) > 0:
                    temp = data[experiment][genotype][t]
                    break
            for t in range(len(data[experiment][genotype])):
                if len(data[experiment][genotype][t]) == 0:
                    data[experiment][genotype][t] = np.zeros_like(temp)
            data[experiment][genotype] = pd.DataFrame(np.concatenate(data[experiment][genotype]))

    return pd.DataFrame(data)


def plot_individuals(data, experiment="B+", nids=None, only_nids=True, maxy=30):
    import matplotlib.pyplot as plt

    title = "individuals-from-data"

    data_exp = data[experiment]
    genotypes = np.sort(data_exp.index)
    odour_a_xs = np.array([np.arange(28, 43) + i * 200 for i in range(9)])
    shock_a_xs = np.array([np.arange(44, 49) + i * 200 for i in range(9)])
    odour_b_xs = np.array([np.arange(28, 43) + i * 200 + 100 for i in range(8)])
    shock_b_xs = np.array([np.arange(44, 49) + i * 200 + 100 for i in range(8)])
    xs = np.arange(14)

    if nids is None:
        nids = np.arange(len(genotypes))
    if only_nids:
        genotypes = genotypes[nids]

    ymin, ymax = 0, 2
    y_lim = [-0.1, 2.1]


    nb_genotypes = len(genotypes)
    nb_plots = nb_genotypes * 2
    subs = []

    nb_rows = 2
    nb_cols = nb_plots // nb_rows
    while nb_cols > 12:
        nb_rows += 2
        nb_cols = nb_plots // nb_rows + 1

    plt.figure(title, figsize=(8 - 2 * int(not only_nids), nb_rows))
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
        colour = np.array([.5 * 205, .5 * 222, 238]) / 255.

        if len(subs) <= j:
            axa = plt.subplot(nb_rows, nb_cols, j+1)
            axa.set_xticks(2 * np.arange(5) + 2)
            axa.set_yticks([0, ymax/2, ymax])
            axa.set_ylim(y_lim)
            axa.set_xlim([0, 12])
            axa.tick_params(labelsize=8)
            axa.set_xticklabels("%s" % (i + 1) for i in range(5))
            if nb_rows > 2:
                axa.set_title(r"$%s$" % genotype, fontsize=8)
                # axa.text(.5, 1.8, r"$%s$" % genotype, fontsize=8)
            else:
                axa.set_title(r"$%s$" % genotype, fontsize=8)
            if j % nb_cols == 0:
                axa.set_ylabel("Odour A", fontsize=8)
            else:
                axa.set_yticklabels([""] * 3)
            # if j // nb_cols < nb_rows - 1:
            #     axa.set_xticklabels([""] * 5)
            # elif j % nb_cols == 0:
            if j % nb_cols == 0:
                axa.text(-8, -.65, "Trial #", fontsize=8)
            axa.spines['top'].set_visible(False)
            axa.spines['right'].set_visible(False)

            acolour = np.array([205, 222, 238]) / 255.

            axa.fill_between(xs, data_a_mean[:14] - data_a_std[:14], data_a_mean[:14] + data_a_std[:14],
                             color=acolour, alpha=0.2)
            axa.plot(xs, data_a_mean[:14], color=acolour, lw=2, label="acquisition")
            subs.append(axa)
        subs[-1].fill_between(xs[:6], data_a_mean[12:] - data_a_std[12:], data_a_mean[12:] + data_a_std[12:],
                              color=colour, alpha=0.2)
        subs[-1].plot(xs[:6], data_a_mean[12:], color=colour, lw=2, label="reversal")
        subs[-1].plot([3, 5], data_a_mean[[15, 17]], 'r.')

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
        colour = np.array([255, .5 * 197, .5 * 200]) / 255.

        jn = j + (nb_rows * nb_cols) // 2

        if len(subs) <= jn:
            axb = plt.subplot(nb_rows, nb_cols, jn+1)
            axb.set_xticks(2 * np.arange(5) + 2)
            axb.set_yticks([0, ymax/2, ymax])
            axb.set_ylim(y_lim)
            axb.set_xlim([0, 12])
            axb.tick_params(labelsize=8)
            axb.set_xticklabels("%s" % (i + 1) for i in range(5))
            if nb_rows > 2:
                axb.set_title(r"$%s$" % genotype, fontsize=8)
                # axb.text(.5, 1.8, r"$%s$" % genotype, fontsize=8)
            if jn % nb_cols == 0:
                axb.set_ylabel("Odour B", fontsize=8)
            else:
                axb.set_yticklabels([""] * 3)
            # if jn // nb_cols < nb_rows - 1:
            #     axb.set_xticklabels([""] * 5)
            # elif jn % nb_cols == 0:
            if jn % nb_cols == 0:
                axb.text(-8, -.65, "Trial #", fontsize=8)
            axb.spines['top'].set_visible(False)
            axb.spines['right'].set_visible(False)

            acolour = np.array([255, 197, 200]) / 255.

            axb.fill_between(xs, data_b_mean[:14] - data_b_std[:14], data_b_mean[:14] + data_b_std[:14],
                             color=acolour, alpha=0.2)
            axb.plot(xs, data_b_mean[:14], color=acolour, lw=2, label="acquisition")
            axb.plot([3, 5, 7, 9, 11], data_b_mean[[3, 5, 7, 9, 11]], 'r.')

            subs.append(axb)
        subs[-1].fill_between(xs[:6], data_b_mean[10:] - data_b_std[10:], data_b_mean[10:] + data_b_std[10:],
                              color=colour, alpha=0.2)
        subs[-1].plot(xs[:6], data_b_mean[10:], color=colour, lw=2, label="reversal")

    subs[len(subs)//2 - 1].legend(fontsize=8, bbox_to_anchor=(1.05, 1.35), loc='upper left',
                                  framealpha=0., labelspacing=1.)
    subs[-1].legend(fontsize=8, bbox_to_anchor=(1.05, 1.35), loc='upper left', framealpha=0., labelspacing=1)

    # subs[len(subs)//2 - 1].legend(fontsize=8, bbox_to_anchor=(1.05, 1.), loc='upper left',
    #                               framealpha=0., labelspacing=1.)
    # subs[-1].legend(fontsize=8, bbox_to_anchor=(1.05, 1.), loc='upper left', framealpha=0., labelspacing=1)
    plt.tight_layout()
    plt.show()
