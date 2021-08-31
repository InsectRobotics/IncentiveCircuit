from incentive.tmaze import TMaze

from copy import copy

import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys
import os
import re

# the directory of the file
__dir__ = os.path.dirname(os.path.abspath(__file__))
# the directory of the data
__data_dir__ = os.path.realpath(os.path.join(__dir__, "..", "src", "incentive", "data", "olfactory-conditioning"))


def main(*args):


    continuous = True
    nb_samples = read_arg(["-f", "--nb-flies"], vtype=int, default=100)
    in_trial_steps = read_arg(["-s", "--steps"], vtype=int, default=100)
    noise = read_arg(["-n", "--noise"], vtype=float, default=.2)
    repeats = read_arg(["-R", "--repeats"], vtype=int, default=10)

    short_names = {
        "elemental": "Ele",
        "multi-element": "2Ele",
        "mixture": "Mix",
        "overlap": "OL",
        "positive-patterning": "PP",
        "negative-patterning": "NP",
        "biconditional-discrimination": "BD",
        "blocking": "Blk",
        "blocking-control": "cBlk"
    }

    exp_path = os.path.join(__data_dir__, "learning-types.yaml")
    with open(exp_path, 'r') as f:
        experiments = yaml.load(f, Loader=yaml.Loader)

    maze = {}

    if continuous:
        print("Continuous")
    else:
        print("Discontinuous")
    for repeat in range(1, repeats + 1):

        exp_name = "olfactory-conditioning%s-%02d" % (
            "" if continuous else "-reset",
            repeat)

        fig = plt.figure(exp_name, figsize=(17, 2.5))
        timer = fig.canvas.new_timer(interval=3000)  # create a timer with 3 sec interval
        timer.add_callback(plt.close)

        data = {}

        for e_id, exp in enumerate(experiments):

            if not continuous and exp in maze:
                del maze[exp]
            if not continuous or exp not in maze:
                maze[exp] = []

            print(repeat, exp)

            plt.subplot(1, len(experiments) + 1, e_id + 1)
            plt.title(exp.replace("-", "\n"), fontsize=8)

            vs = []
            labels = []

            for i, experiment in enumerate(experiments[exp]):
                # nb_train = experiment["training-cycles"]
                nb_test = 1
                nb_samples_i = int(np.ceil(nb_samples / len(experiments[exp])))
                train = experiment["train"]
                test = experiment["test"]

                print(' '.join(train), end='')

                if i <= len(maze[exp]):
                    maze[exp].append(TMaze(train=train, test=test, nb_train=repeat, nb_test=nb_test,
                                           nb_in_trial=in_trial_steps, nb_samples=nb_samples_i))
                elif not continuous:
                    maze[exp][i] = TMaze(train=train, test=test, nb_train=repeat, nb_test=nb_test,
                                         nb_in_trial=in_trial_steps, nb_samples=nb_samples_i)
                maze[exp][i](noise=noise)

                label = ' '.join(train)
                for tt in test:
                    v = maze[exp][i].get_test_result(tt, train=False).flatten()
                    vs.append(v)
                    labels.append(label)
                    print(",   %s    (PI): %.2f +/- %.2f" % (tt, float(np.nanmean(v)), float(np.nanstd(v))), end="")
                print("")
            data[exp] = np.array(copy(vs)).flatten()

            plt.plot([0, len(labels) + 1], [0, 0], 'grey', lw=2)
            plt.boxplot(vs)
            plt.yticks([-1, 0, 1], fontsize=8)
            plt.xticks(np.arange(len(labels)) + 1, labels, rotation=40, fontsize=8)
            plt.ylim(-1, 1)
            plt.xlim(0, len(labels) + 1)

        plt.subplot(1, len(experiments) + 1, len(experiments) + 1)
        plt.title("pooled data", fontsize=8)

        print("POOLED DATA:")
        print("------------")
        for k in data:
            print("%s   \tPI: %.2f +/- %.2f" % (short_names[k], float(np.nanmean(data[k])), float(np.nanstd(data[k]))))
        print("")

        plt.plot([0, len(data.keys()) + 1], [0, 0], 'grey', lw=2)
        plt.boxplot(data.values())
        plt.yticks([-1, 0, 1], fontsize=8)
        plt.xticks(np.arange(len(data.keys())) + 1, [short_names[k] for k in data.keys()], rotation=70, fontsize=8)
        plt.ylim(-1, 1)
        plt.xlim(0, len(data.keys()) + 1)

        plt.tight_layout()
        plt.savefig(os.path.join(__data_dir__, "%s.png" % exp_name), dpi=300)
        timer.start()
        plt.show()


if __name__ == '__main__':
    main(*sys.argv)
