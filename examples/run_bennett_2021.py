import numpy as np

from incentive.bennett import Bennett, read_data, translate_condition_code, pi_binomial_adjustment

import matplotlib.pyplot as plt
import pandas as pd

import yaml
import sys
import os

# the directory of the file
__dir__ = os.path.dirname(os.path.abspath(__file__))
# the directory of the data
__data_dir__ = os.path.realpath(os.path.join(__dir__, "..", "src", "incentive", "data", "bennett2021"))

with open(os.path.join(__data_dir__, "intervention-examples.yaml"), 'r') as f:
    experiments = yaml.load(f, Loader=yaml.Loader)


def main(*args):

    learning_rule = "rpe"
    nb_train, nb_test = 10, 2
    nb_trials = 10
    nb_samples = 20

    data = read_data(os.path.join(__data_dir__, "41467_2021_22592_MOESM4_ESM.xlsx"))
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.max_rows', 93)
    pd.set_option('display.width', 1000)

    plt.figure("bennett-2021-%s" % learning_rule, figsize=(7, 7))

    targets = [
        ["s", "d"],
        ["r", "c"],
        ["m", "f"],
        ["s", "r", "d", "c"],
        ["s", "m", "d", "f"],
        ["r", "m", "c", "f"],
        ["s", "r", "m", "d", "c", "f"],
        "Plausible model",
        "Best model"
    ]
    indices = {
        "s": 0, "r": 1, "m": 2, "d": 0, "c": 1, "f": 2,
        "sr": 3, "sm": 4, "rm": 5, "dc": 3, "df": 4, "cf": 5,
        "srm": 6, "dcf": 6
    }

    delta_f_data = np.array(data["PI difference"])
    z_data = np.sqrt(np.nansum(np.square(delta_f_data)))
    delta_f_models = []
    rng = np.random.RandomState(2021)

    for p_i, target_intervention in enumerate(targets):

        plt.subplot(331 + p_i)
        plt.title("".join(target_intervention))
        if target_intervention == "Best model":
            data_corr = delta_f_data / z_data
            models_corr = np.array(delta_f_models) / np.sqrt(np.nansum(np.square(delta_f_models), axis=1))[..., np.newaxis]
            model_id = np.argmin(np.square(data_corr - models_corr), axis=0)
            delta_f_model = np.array(delta_f_models)[model_id, np.arange(len(model_id))]

            data["Plausible model"] = np.array(targets)[model_id]

            print(data[["Condition code", "Plausible model", "Cell types"]])
        elif target_intervention == "Plausible model":
            model_id = []
            for e_id, condition in enumerate(data["Condition code"]):
                mod = np.array(data["Plausible model"])[e_id]
                if isinstance(mod, str):
                    model_id.append(indices[mod])
            delta_f_model = np.array(delta_f_models)[model_id, np.arange(len(model_id))]

        else:
            pibas = []
            for e_id, condition in enumerate(data["Condition code"]):

                exp = translate_condition_code(condition)
                train = exp["train"]
                test = exp["test"]
                excite = exp["excite"]
                inhibit = exp["inhibit"]
                intervention = exp["intervention-schedule"]
                # intervention_schedule = {
                #     "train_CS+": bool(int("{0:03b}".format(intervention)[0])),
                #     "train_CS-": bool(int("{0:03b}".format(intervention)[1])),
                #     "test": bool(int("{0:03b}".format(intervention)[2]))
                # }
                # print(e_id, condition, train, test, excite, inhibit, intervention_schedule,
                #       np.array(data["Cell types"])[e_id])

                ben = Bennett(train=train, test=test, nb_train=nb_train, nb_test=nb_test, nb_in_trial=nb_trials,
                              rng=rng, learning_rule=learning_rule)
                ben(excite=excite, inhibit=inhibit, intervention=0, noise=0.1)

                pi_control = np.nanmean(ben.get_pi("A vs B", train=False).flatten())
                # pi_control = np.array(data["Mean control PI"])[e_id]

                if isinstance(target_intervention, list):
                    mod = target_intervention
                else:
                    print(f"Skipping '{target_intervention}'!")
                    continue

                pi_bas = []
                for i in range(nb_samples):
                    ben = Bennett(train=train, test=test, nb_train=nb_train, nb_test=nb_test, nb_in_trial=nb_trials,
                                  target_intervention=mod, rng=rng)
                    ben(excite=excite, inhibit=inhibit, intervention=intervention, noise=0.1)

                    pi_condition = np.nanmean(ben.get_pi("A vs B", train=False).flatten())
                    pi_bas.append(pi_binomial_adjustment(pi_condition, pi_control))
                pibas.append(np.median(pi_bas))

            delta_f_model = np.array(pibas)

        z_model = np.sqrt(np.nansum(np.square(delta_f_model)))
        delta_f_models.append(delta_f_model.copy())

        corr = np.nansum((delta_f_data / z_data) * (delta_f_model / z_model))
        print("".join(target_intervention), "Correlation: %.4f" % corr)

        condition_colours = {
            4312: "#00B51C",  # light green
            4411: "#B240D8",  # magenta
            3111: "#0089FA",  # light blue
            3212: "#FF7D1C",  # orange
            1223: "#EE2913",  # red
            1323: "#00700D",  # green
            2112: "#0030F8"  # blue
        }
        colours = []
        for code in data["Condition code"]:
            if code in condition_colours:
                colours.append(condition_colours[code])
            else:
                colours.append("lightgrey")
        plt.plot([-10, 10], [0, 0], 'lightgrey', ls=':')
        plt.plot([0, 0], [-3, 3], 'lightgrey', ls=':')
        plt.scatter(delta_f_model, delta_f_data, edgecolors=colours, facecolors='none', marker='o', s=20)
        plt.yticks([-2, -1, 0, 1, 2], fontsize=8)
        plt.xticks([-6, -3, 0, 3, 6], fontsize=8)
        plt.ylim(-3, 3)
        plt.xlim(-7, 7)
        plt.ylabel(r"$\Delta_f$ (experiment)")
        plt.xlabel(r"$\Delta_f$ (model)")

    plt.tight_layout()
    plt.savefig(os.path.join(__data_dir__, "bennett-2021-%s.png" % learning_rule), dpi=300)
    plt.show()


if __name__ == '__main__':
    main(*sys.argv)
