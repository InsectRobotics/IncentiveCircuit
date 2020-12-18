# from utils import err, debug, __root__

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

            data[experiment][name] = pd.DataFrame(np.concatenate(gdata))

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


def plot_traces(data, experiment="B+", diff=None, maxy=30):
    import matplotlib.pyplot as plt

    data_exp = data[experiment]
    data_dif = None if diff is None else data[diff]
    genotypes = np.sort(data_exp.index)
    nb_subplots = len(genotypes)
    if nb_subplots > 20:
        nb_rows = (nb_subplots + 1) // 4
        nb_cols = nb_subplots % 4 + 1
    elif nb_subplots > 15:
        nb_rows = (nb_subplots + 1) // 3
        nb_cols = nb_subplots % 3 + 1
    elif nb_subplots > 10:
        nb_rows = (nb_subplots + 1) // 2
        nb_cols = nb_subplots % 2 + 1
    else:
        nb_rows = nb_subplots
        nb_cols = 1
    plt.figure("traces-" + experiment, figsize=(10, 10))

    xticks = []
    for t in range(17):
        if t < 12:
            tick = "%d" % (t // 2 + 1) if t % 2 == 0 else ""
        else:
            tick = "%d" % ((t - 1) // 2 + 1) if t % 2 == 1 else ""
        xticks.append(tick)

    ymin, ymax = 0, 0
    for genotype in genotypes:
        if diff is not None:
            data_gen = data_exp[genotype] - data_dif[genotype]
        else:
            data_gen = data_exp[genotype]

        if np.all(np.isnan(data_gen)) or len(data_gen) < 1:
            continue
        yminn, ymaxx = np.nanmin(data_gen), np.minimum(np.nanmax(data_gen), maxy)
        if yminn < ymin:
            ymin = yminn
        if ymaxx > ymax:
            ymax = ymaxx
    y_lim = [ymin * 1.1, ymax * 1.1]
    x_fill = np.vstack([np.array([25, 25, 50, 50]) + i * 100 for i in range(17)])
    xa_fill = x_fill[0::2].flatten()
    xb_fill = x_fill[1::2].flatten()
    ya_fill = np.array([y_lim[0], y_lim[1], y_lim[1], y_lim[0]] * 9)
    yb_fill = np.array([y_lim[0], y_lim[1], y_lim[1], y_lim[0]] * 8)
    if "+" in experiment:
        s_mark = "-"
    else:
        s_mark = ":"
    for i, genotype in enumerate(genotypes):
        if diff is not None:
            data_gen = data_exp[genotype] - data_dif[genotype]
        else:
            data_gen = data_exp[genotype]
        plt.subplot(nb_rows, nb_cols, i+1)
        plt.fill_between(xa_fill, np.full_like(ya_fill, y_lim[0]), ya_fill, color='C0', alpha=0.1)
        plt.fill_between(xb_fill, np.full_like(yb_fill, y_lim[0]), yb_fill, color='C1', alpha=0.1)
        for s in [2, 3, 4, 5, 6]:
            plt.plot([(s - 1) * 200 + 145] * 2, y_lim, 'r%s' % s_mark, lw=1)
        for s in [8, 9]:
            plt.plot([(s - 1) * 200 + 45] * 2, y_lim, 'r%s' % s_mark, lw=1)
        plt.plot(data_gen, 'k-', alpha=.2, lw=.5)
        if np.any(~np.isnan(data_gen)):
            plt.plot(data_gen.mean(axis=1), 'k-', lw=2)

        odour_a_xs = np.array([np.arange(28, 43) + i * 200 for i in range(9)])
        odour_b_xs = np.array([np.arange(28, 43) + i * 200 + 100 for i in range(8)])
        data_a_mean = np.nanmean(np.array(data_gen)[odour_a_xs], axis=(1, 2))
        data_a_std = np.nanstd(np.array(data_gen)[odour_a_xs], axis=(1, 2)) / 2
        data_b_mean = np.nanmean(np.array(data_gen)[odour_b_xs], axis=(1, 2))
        data_b_std = np.nanstd(np.array(data_gen)[odour_b_xs], axis=(1, 2)) / 2
        xs_a = np.arange(0, 17)[::2] * 100 + 30
        xs_b = np.arange(1, 17)[::2] * 100 + 30

        plt.fill_between(xs_a, data_a_mean - data_a_std, data_a_mean + data_a_std, color='C0', alpha=0.2)
        plt.fill_between(xs_b, data_b_mean - data_b_std, data_b_mean + data_b_std, color='C1', alpha=0.2)
        plt.plot(xs_a, data_a_mean, "C0-", lw=2)
        plt.plot(xs_b, data_b_mean, "C1-", lw=2)

        plt.xticks(np.arange(50, 1700, 100), xticks)
        plt.xlim([0, 1700])
        plt.ylim(y_lim)
        plt.ylabel(genotype)

    plt.tight_layout()
    plt.show()


def plot_overlap(data, experiment="B+", phase2="reversal", title=None, score=None, zeros=False, individuals=False):
    import matplotlib.pyplot as plt

    sort_titles = {
        "acquisition": "Aq",
        "reversal": "Re",
        "extinction": "Ex"
    }
    if not isinstance(data, list):
        data = [data]
    if not isinstance(phase2, list):
        phase2 = [phase2]

    if title is None:
        title = "overlap-" + experiment + ("" if individuals else "avg") + ("_descr" if score is not None else "")

    plt.figure(title, figsize=(7 + 3 * len(data), 10))
    genotypes = np.sort(data[0][experiment].index)
    if len(genotypes) == 6:
        genotypes = genotypes[[0, 1, 2, 4, 5, 3]]
    nb_rows = len(genotypes)
    nb_cols = 2 + len(data) * 2
    if nb_rows > 20:
        nb_cols *= 4
        nb_rows = (nb_rows + 1) // 4
    elif nb_rows > 15:
        nb_cols *= 3
        nb_rows = (nb_rows + 1) // 3
    elif nb_rows > 10:
        nb_cols *= 2
        nb_rows = (nb_rows + 1) // 2
    xticks = np.linspace(0, 20, 5)
    ymin, ymax = 0, 0

    for genotype in genotypes:
        if np.all(np.isnan(data[0][experiment][genotype])) or len(data[0][experiment][genotype]) < 1:
            continue
        if data[0][experiment][genotype].mean(axis=1).min() < ymin:
            ymin = data[0][experiment][genotype].mean(axis=1).min()
        if data[0][experiment][genotype].mean(axis=1).max() > ymax:
            ymax = data[0][experiment][genotype].mean(axis=1).max()

    y_lim = [np.minimum(ymin * 1.1, -.1), np.maximum(ymax * 1.1, +1.)]
    x_fill = np.array([25, 25, 50, 50])
    y_fill = np.array([y_lim[0], y_lim[1], y_lim[1], y_lim[0]])

    if "+" in experiment:
        s_mark = "-"
    else:
        s_mark = ":"
    i = 1

    acq = "acquisition" if nb_cols < 5 else sort_titles["acquisition"]
    rev = "reversal" if nb_cols < 5 else sort_titles["reversal"]
    ext = "extinction" if nb_cols < 5 else sort_titles["extinction"]

    for genotype in genotypes:
        for k, dat_ in enumerate(data):
            data_exp = dat_[experiment]
            phase_k = (phase2[k] if nb_cols < 5 else sort_titles[phase2[k]])

            titles = []
            if k < 1:
                titles += ["%s (A)" % acq, "%s (B)" % acq]
            titles += ["%s (A)" % phase_k, "%s (B)" % phase_k]
            for title in titles:
                plt.subplot(nb_rows, nb_cols, i)
                plt.fill_between(x_fill, np.full_like(y_fill, y_lim[0]), y_fill,
                                 color='C%d' % (int("B" in title)), alpha=0.1)
                plt.plot([45] * 2, y_lim, 'r%s' % (s_mark if title in ["%s (B)" % acq, "%s (A)" % rev] else ":"), lw=1)
                if np.any(~np.isnan(data_exp[genotype])):
                    trials = []
                    if title == "%s (A)" % acq:
                        trials.extend([3, 5, 7, 9, 11])
                    elif title == "%s (B)" % acq:
                        trials.extend([4, 6, 8, 10, 12])
                    elif title == "%s (A)" % phase_k:
                        trials.extend([15, 17, 19, 21, 23, 25])
                    elif title == "%s (B)" % phase_k:
                        trials.extend([14, 16, 18, 20, 22, 24])
                    for tr in trials[::-1]:
                        if tr * 100 >= data_exp[genotype].shape[0]:
                            trials.remove(tr)
                    for j, tr in enumerate(trials):  # A-
                        if individuals:
                            ddt = data_exp[genotype][(tr-1)*100:tr*100]
                        else:
                            ddt = data_exp[genotype][(tr-1)*100:tr*100].mean(axis=1)
                        c = 1. / len(trials) * (j + 1)
                        if score is not None and genotype in list(score.columns) and title in list(score.index):
                            if score[genotype][title] > 0:
                                c = [1-c, 1, 1-c]
                            elif score[genotype][title] < 0 or (not zeros and score[genotype][title] == 0):
                                c = [1, 1-c, 1-c]
                        if not isinstance(c, list):
                            c = .9 - .9 / len(trials) * (j + 1)
                            c = [c, c, c]
                        # print(c, ddt.shape)
                        if individuals:
                            plt.plot(np.array(ddt), '-', c=c, lw=.2)
                        else:
                            plt.plot(np.array(ddt), '-', c=c, lw=2)

                plt.xticks([0, 25, 50, 75, 99], xticks)
                plt.xlim([0, 100])
                plt.ylim(y_lim)
                if i % 4 == 1:
                    plt.ylabel(genotype)
                plt.xlabel(title)
                i += 1

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # df = load_draft_data()
    df = load_data("B+")
    # plot_traces(df, "A+", diff="A-")
    plot_traces(df, "B+")
    # print(df)
    # plot_overlap(df, "B+")
