# from utils import err, debug, __root__

import os
import re
import csv
import pandas as pd
import numpy as np

__dir__ = os.path.dirname(os.path.abspath(__file__))
__data_dir__ = os.path.realpath(os.path.join(__dir__, "..", "data", "FruitflyMB", "draft"))

__dirs = {
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
_pattern_ = r'realSCREEN_([\d\w\W]+)\.xlsx_finaldata([\w\W]+)_timepoint(\d)\.csv'


def load_draft_data():
    data = {}
    for genotype in __dirs:
        for experiment in __dirs[genotype]:
            if experiment not in data:
                data[experiment] = {genotype: []}
            data[experiment][genotype] = [[]] * 18
            for r, _, flist in os.walk(__data_dir__):
                match = True
                for d in __dirs[genotype][experiment]:
                    if d not in r:
                        match = False
                if not match:
                    continue

                labels = re.findall(r'.*\(\d\-(\d),\d\-(\d)\).*', r)
                if len(labels) < 1:
                    err("Unknown directory pattern:", r)
                    continue
                nb_csm, nb_csp = labels[0]
                nb_csm, nb_csp = int(nb_csm), int(nb_csp)

                for filename in flist:
                    details = re.findall(_pattern_, filename)
                    if len(details) < 1:
                        err("Unknown file pattern:", os.path.join(r, filename))
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


def plot_traces(data, experiment="B+", diff=None):
    import matplotlib.pyplot as plt

    data_exp = data[experiment]
    data_dif = None if diff is None else data[diff]
    genotypes = np.sort(data_exp.index)
    nb_subplots = len(genotypes)
    plt.figure("traces-" + experiment, figsize=(10, 10))

    xticks = []
    for t in range(1, 10):
        for cs in ["A", "B"]:
            if ("+" in experiment) and ((t in [2, 3, 4, 5, 6] and cs == "B") or (t in [8, 9] and cs == "A")):
                shock = "+"
            else:
                shock = "-"
            if len(xticks) < 17:
                xticks.append(cs + shock)
            else:
                break

    ymin, ymax = 0, 0
    for genotype in genotypes:
        if diff is not None:
            data_gen = data_exp[genotype] - data_dif[genotype]
        else:
            data_gen = data_exp[genotype]

        if np.all(np.isnan(data_gen)) or len(data_gen) < 1:
            continue
        if data_gen.min().min() < ymin:
            ymin = data_gen.min().min()
        if data_gen.max().max() > ymax:
            ymax = data_gen.max().max()

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
        plt.subplot(nb_subplots, 1, i+1)
        plt.fill_between(xa_fill, np.full_like(ya_fill, y_lim[0]), ya_fill, color='C0', alpha=0.1)
        plt.fill_between(xb_fill, np.full_like(yb_fill, y_lim[0]), yb_fill, color='C1', alpha=0.1)
        for s in [2, 3, 4, 5, 6]:
            plt.plot([(s - 1) * 200 + 145] * 2, y_lim, 'r%s' % s_mark, lw=1)
        for s in [8, 9]:
            plt.plot([(s - 1) * 200 + 45] * 2, y_lim, 'r%s' % s_mark, lw=1)
        plt.plot(data_gen, 'k-', alpha=.2, lw=.5)
        if np.any(~np.isnan(data_gen)):
            plt.plot(data_gen.mean(axis=1), 'k-', lw=2)
        plt.xticks(np.arange(50, 1700, 100), xticks)
        plt.xlim([0, 1700])
        plt.ylim(y_lim)
        plt.ylabel(genotype)

    plt.tight_layout()
    plt.show()


def plot_overlap(data, experiment="B+", phase2="reversal", title=None, score=None, zeros=False, individuals=False):
    import matplotlib.pyplot as plt

    if not isinstance(data, list):
        data = [data]
    if not isinstance(phase2, list):
        phase2 = [phase2]

    if title is None:
        title = "overlap-" + experiment + ("" if individuals else "avg") + ("_descr" if score is not None else "")

    plt.figure(title, figsize=(7 + 3 * len(data), 10))
    genotypes = np.sort(data[0][experiment].index)[[0, 1, 2, 4, 5, 3]]
    nb_rows = len(genotypes)
    nb_cols = 2 + len(data) * 2
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

    for genotype in genotypes:
        for k, dat_ in enumerate(data):
            data_exp = dat_[experiment]

            titles = []
            if k < 1:
                titles += ["acquisition (A)", "acquisition (B)"]
            titles += ["%s (A)" % phase2[k], "%s (B)" % phase2[k]]
            for title in titles:
                plt.subplot(nb_rows, nb_cols, i)
                plt.fill_between(x_fill, np.full_like(y_fill, y_lim[0]), y_fill,
                                 color='C%d' % (int("B" in title)), alpha=0.1)
                plt.plot([45] * 2, y_lim, 'r%s' % (s_mark if title in ["acquisition (B)", "reversal (A)"] else ":"), lw=1)
                if np.any(~np.isnan(data_exp[genotype])):
                    trials = []
                    if title == "acquisition (A)":
                        trials.extend([3, 5, 7, 9, 11])
                    elif title == "acquisition (B)":
                        trials.extend([4, 6, 8, 10, 12])
                    elif title == "%s (A)" % phase2[k]:
                        trials.extend([15, 17, 19, 21, 23, 25])
                    elif title == "%s (B)" % phase2[k]:
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
                plt.ylabel(genotype)
                plt.xlabel(title)
                i += 1

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    df = load_draft_data()
    # plot_traces(df, "A+", diff="A-")
    print(df)
    plot_overlap(df, "B+")
