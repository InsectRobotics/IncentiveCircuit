import numpy as np

from incentive.bennett import Bennett

import matplotlib.pyplot as plt

import yaml
import sys
import os

# the directory of the file
__dir__ = os.path.dirname(os.path.abspath(__file__))
# the directory of the data
__data_dir__ = os.path.realpath(os.path.join(__dir__, "..", "src", "incentive", "data", "bennett2021"))

with open(os.path.join(__data_dir__, "intervention-examples.yaml"), 'r') as f:
    experiments = yaml.load(f)


def main(*args):

    nb_train, nb_test = 10, 2

    plt.figure("bennett-2021-box", figsize=(15, 5))

    for e_id, exp in enumerate(experiments):

        train = exp["train"]
        test = exp["test"]
        excite = exp["excite"]
        inhibit = exp["inhibit"]
        intervention = exp["intervention-schedule"]
        intervention_schedule = {
            "train_CS+": bool(int("{0:03b}".format(intervention)[0])),
            "train_CS-": bool(int("{0:03b}".format(intervention)[1])),
            "test": bool(int("{0:03b}".format(intervention)[2]))
        }
        print(e_id, train, test, excite, inhibit, intervention_schedule)

        ben = Bennett(train=train, test=test, nb_train=nb_train, nb_test=nb_test,
                      nb_in_trial=10)
        ben(excite=excite, inhibit=inhibit, intervention=intervention,
            noise=0.1)

        plt.subplot(1, len(experiments), e_id + 1)

        vs = []
        # for label in ["A", "B"]:
        for label in test:
            vs.append(ben.get_pi(label, train=False).flatten() / 2)
            # vs.append(ben.get_values(label, train=True, test=True).flatten())
        # plt.plot(np.array(vs).T, label=["A", "B"])
        # plt.plot(np.array(vs).T, label=test)
        plt.plot([0, len(test) + 1], [0, 0], 'grey', lw=2)
        plt.boxplot(vs)
        plt.yticks([-1, 0, 1], fontsize=8)
        # plt.yticks([-2, 0, 2], fontsize=8)
        plt.xticks(np.arange(len(test)) + 1, test, rotation=20, fontsize=8)
        plt.ylim(-1, 1)
        # plt.ylim(-2, 2)
        plt.xlim(0, len(test) + 1)
        # plt.legend()

    # plt.tight_layout()
    # plt.savefig("bennett-2021-box.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    main(*sys.argv)
