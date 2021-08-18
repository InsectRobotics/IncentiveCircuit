import numpy as np

from incentive.tmaze import TMaze

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
    control = False
    nb_train, nb_test = 1, 1
    noise = .4

    if control:
        exp_path = os.path.join(__data_dir__, "learning-types-control.yaml")
    else:
        exp_path = os.path.join(__data_dir__, "learning-types.yaml")
    with open(exp_path, 'r') as f:
        experiments = yaml.load(f, Loader=yaml.Loader)

    for repeat in range(1, 11):

        exp_name = "olfactory-conditioning%s%s-%02d" % (
            "" if continuous else "-reset",
            "-control" if control else "",
            repeat)

        fig = plt.figure(exp_name, figsize=(15, 2.5))
        timer = fig.canvas.new_timer(interval=3000)  # create a timer with 3 sec interval
        timer.add_callback(plt.close)

        maze = {}

        for e_id, exp in enumerate(experiments):

            train = experiments[exp]["train"]
            test = experiments[exp]["test"]
            print(repeat, exp, train, test)

            if not continuous and exp in maze:
                del maze[exp]
            if not continuous or exp not in maze:
                maze[exp] = TMaze(train=train, test=test, nb_train=nb_train * repeat, nb_test=nb_test, nb_in_trial=100,
                                  nb_samples=100)
            maze[exp](noise=noise)

            plt.subplot(1, len(experiments), e_id + 1)
            plt.title(exp.replace("-", "\n"), fontsize=8)

            vs = []
            for label in test:
                vs.append(maze[exp].get_test_result(label, train=False).flatten())
            plt.plot([0, len(test) + 1], [0, 0], 'grey', lw=2)
            plt.boxplot(vs)
            plt.yticks([-1, 0, 1], fontsize=8)
            plt.xticks(np.arange(len(test)) + 1, test, rotation=20, fontsize=8)
            plt.ylim(-1, 1)
            plt.xlim(0, len(test) + 1)

        plt.tight_layout()
        plt.savefig(os.path.join(__data_dir__, "%s.png" % exp_name), dpi=300)
        timer.start()
        plt.show()


if __name__ == '__main__':
    main(*sys.argv)
