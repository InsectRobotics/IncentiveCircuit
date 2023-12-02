import pandas as pd

from incentive.tmaze import TMaze
from incentive.tools import read_arg

from copy import copy

import numpy as np
# import matplotlib.pyplot as plt
import yaml
import sys
import os

np.set_printoptions(edgeitems=30, linewidth=100000)

# the directory of the file
__dir__ = os.path.dirname(os.path.abspath(__file__))
# the directory of the data
__data_dir__ = os.path.realpath(os.path.join(__dir__, "..", "src", "incentive", "data", "tmaze"))


def main(*args):

    continuous = True
    load_data = False
    nb_samples = read_arg(["-f", "--nb-flies"], vtype=int, default=100)
    in_trial_steps = read_arg(["-s", "--steps"], vtype=int, default=100)
    noise = read_arg(["-n", "--noise"], vtype=float, default=.2)
    repeats = read_arg(["-R", "--repeats"], vtype=int, default=10)

    # short_names = {
    #     "elemental": "Ele",
    #     "multi-element": "2Ele",
    #     "mixture": "Mix",
    #     "overlap": "OL",
    #     "positive-patterning": "PP",
    #     "negative-patterning": "NP",
    #     "biconditional-discrimination": "BD",
    #     "blocking": "Blk",
    #     "blocking-control": "cBlk"
    # }

    exp_path = os.path.join(__data_dir__, "conditioning-types.yaml")
    with open(exp_path, 'r') as f:
        experiments = yaml.load(f, Loader=yaml.Loader)

    maze = {}

    if continuous:
        print("Continuous", end=", ")
    else:
        print("Discontinuous", end=", ")

    data_df = {
        "CS+": [],
        "CS-": [],
        "test_1": [],
        "test_2": [],
        "experiment": [],
        "repeat": [],
        "PI": []
    }
    excel_file = os.path.join(__data_dir__, f"tmaze-results{('' if continuous else '-nu')}.xlsx")
    if load_data and os.path.exists(excel_file):
        df_load = pd.read_excel(excel_file)
    else:
        df_load = None
        load_data = False

    def get_key(series: pd.Series):
        order = list(experiments)
        for i in range(len(series)):
            series[i] = order.index(series[i])
        return series

    for repeat in range(1, repeats + 1):

        # exp_name = "olfactory-conditioning%s-%02d" % ("" if continuous else "-reset", repeat)

        # fig = plt.figure(exp_name, figsize=(17, 2.5))
        # timer = fig.canvas.new_timer(interval=3000)  # create a timer with 3 sec interval
        # timer.add_callback(plt.close)

        data = {}

        for e_id, exp in enumerate(experiments):

            if not continuous and exp in maze:
                del maze[exp]
            if not continuous or exp not in maze:
                maze[exp] = []

            print(repeat, exp)

            # plt.subplot(1, len(experiments) + 1, e_id + 1)
            # plt.title(exp.replace("-", "\n"), fontsize=8)

            vs = []
            labels = []

            for i, experiment in enumerate(experiments[exp]):
                # nb_train = experiment["training-cycles"]
                nb_test = 1
                nb_samples_i = int(np.ceil(nb_samples / len(experiments[exp])))
                train = experiment["train"]
                test = experiment["test"]

                print(' '.join(train), end='')

                cont = False
                csp = ' '.join(t[:-1] for t in train if '-' in t)
                csm = ' '.join(t for t in train if '-' not in t)
                if csp == '':
                    csp = "none"
                if csm == '':
                    csm = 'none'

                if load_data and np.sum(np.all([df_load["repeat"] == repeat, df_load["experiment"] == exp,
                                                df_load["CS+"] == csp, df_load["CS-"] == csm], axis=0)) < 1:
                    cont = True

                label = ' '.join(train)
                if cont or not load_data:
                    if i <= len(maze[exp]):
                        maze[exp].append(TMaze(train=train, test=test, nb_train=repeat, nb_test=nb_test,
                                               nb_in_trial=in_trial_steps, nb_samples=nb_samples))
                    elif not continuous:
                        maze[exp][i] = TMaze(train=train, test=test, nb_train=repeat, nb_test=nb_test,
                                             nb_in_trial=in_trial_steps, nb_samples=nb_samples)
                    maze[exp][i](noise=noise)

                    for tt in test:
                        t1 = tt.split(" vs ")[0]
                        t2 = tt.split(" vs ")[1]
                        v = maze[exp][i].get_test_result(tt, train=False).flatten()
                        vs.append(v)
                        labels.append(label)
                        print(",   %s    (PI): %.2f +/- %.2f" % (tt, float(np.nanmean(v)), float(np.nanstd(v))), end="")
                        data_df["CS+"].extend([csp] * len(v))
                        data_df["CS-"].extend([csm] * len(v))
                        data_df["test_1"].extend([t1] * len(v))
                        data_df["test_2"].extend([t2] * len(v))
                        data_df["experiment"].extend([exp] * len(v))
                        data_df["repeat"].extend([repeat] * len(v))
                        data_df["PI"].extend(v.tolist())
                else:

                    for tt in test:
                        t1 = tt.split(" vs ")[0]
                        t2 = tt.split(" vs ")[1]
                        data_temp = df_load[np.all([
                            df_load["repeat"] == repeat,
                            df_load["experiment"] == exp,
                            df_load["CS+"] == csp,
                            df_load["CS-"] == csm,
                            df_load["test_1"] == t1,
                            df_load["test_2"] == t2,
                        ], axis=0)]
                        v = data_temp["PI"]
                        vs.append(v)
                        labels.append(label)

                        print(",   %s    (PI): %.2f +/- %.4f" % (tt, float(np.nanmean(v)), float(np.nanstd(v))), end="")
                        data_df["CS+"].extend([csp] * len(v))
                        data_df["CS-"].extend([csm] * len(v))
                        data_df["test_1"].extend([t1] * len(v))
                        data_df["test_2"].extend([t2] * len(v))
                        data_df["experiment"].extend([exp] * len(v))
                        data_df["repeat"].extend([repeat] * len(v))
                        data_df["PI"].extend(v.tolist())
                print("")
            data[exp] = np.array(copy(vs)).flatten()

            # plt.plot([0, len(labels) + 1], [0, 0], 'grey', lw=2)
            # plt.boxplot(vs)
            # plt.yticks([-1, 0, 1], fontsize=8)
            # plt.xticks(np.arange(len(labels)) + 1, labels, rotation=40, fontsize=8)
            # plt.ylim(-1, 1)
            # plt.xlim(0, len(labels) + 1)

        df = pd.DataFrame(data_df)
        df.to_excel(excel_file)

        # plt.subplot(1, len(experiments) + 1, len(experiments) + 1)
        # plt.title("pooled data", fontsize=8)

        print("\nPOOLED DATA:")

        df_rep_mean = df[df["repeat"] == repeat].groupby(by="experiment")["PI"].mean()
        df_rep_std = df[df["repeat"] == repeat].groupby(by="experiment")["PI"].sem()
        df_rep = pd.concat([df_rep_mean, df_rep_std], axis=1, keys=["PI_mean", "PI_sem"])
        df_rep.sort_values(by="experiment", inplace=True, key=get_key)

        print(df_rep)
        print(f"DF.shape = {df.shape}\n")

        # plt.plot([0, len(data.keys()) + 1], [0, 0], 'grey', lw=2)
        # plt.boxplot(data.values())
        # plt.yticks([-1, 0, 1], fontsize=8)
        # plt.xticks(np.arange(len(data.keys())) + 1, [short_names[k] for k in data.keys()], rotation=70, fontsize=8)
        # plt.ylim(-1, 1)
        # plt.xlim(0, len(data.keys()) + 1)
        #
        # plt.tight_layout()
        # plt.savefig(os.path.join(__data_dir__, "%s.png" % exp_name), dpi=300)
        # timer.start()
        # plt.show()


if __name__ == '__main__':
    main(*sys.argv)
