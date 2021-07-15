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

with open(os.path.join(__data_dir__, "learning-types.yaml"), 'r') as f:
    experiments = yaml.load(f)


def main(*args):

    nb_train, nb_test = 0, 4

    plt.figure("olfactory-conditioning", figsize=(15, 2.5))

    for e_id, exp in enumerate(experiments):

        train = experiments[exp]["train"]
        test = experiments[exp]["test"]
        print(exp, train, test)

        maze = TMaze(train=train, test=test, nb_train=nb_train, nb_test=nb_test,
                     nb_in_trial=100)
        maze(noise=0.)

        plt.subplot(1, len(experiments), e_id + 1)
        plt.title(exp.replace("-", "\n"), fontsize=8)

        vs = []
        for label in test:
            vs.append(maze.get_test_result(label, train=False).flatten())
        plt.plot([0, len(test) + 1], [0, 0], 'grey', lw=2)
        plt.boxplot(vs)
        plt.yticks([-1, 0, 1], fontsize=8)
        plt.xticks(np.arange(len(test)) + 1, test, rotation=20, fontsize=8)
        plt.ylim(-1, 1)
        plt.xlim(0, len(test) + 1)
    plt.tight_layout()
    # plt.savefig("olfactory-conditioning-%02d.png" % nb_train, dpi=300)
    plt.show()


if __name__ == '__main__':
    main(*sys.argv)
