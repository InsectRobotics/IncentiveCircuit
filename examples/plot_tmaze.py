from scipy.stats import ttest_1samp

import pandas as pd
import seaborn as sns

import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys
import os

pd.set_option('display.max_columns', None)
np.set_printoptions(edgeitems=30, linewidth=100000)

# the directory of the file
__dir__ = os.path.dirname(os.path.abspath(__file__))
# the directory of the data
__data_dir__ = os.path.realpath(os.path.join(__dir__, "..", "src", "incentive", "data", "tmaze"))


def main(*args):

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

    exp_path = os.path.join(__data_dir__, "conditioning-types.yaml")
    with open(exp_path, 'r') as f:
        experiments = yaml.load(f, Loader=yaml.Loader)

    def get_key(series: pd.Series):
        order = list(experiments)
        for i in range(len(series)):
            series[i] = order.index(series[i])
        return series

    # df = pd.read_excel(os.path.join(__data_dir__, f"tmaze-results.xlsx"))
    df = pd.read_excel(os.path.join(__data_dir__, f"tmaze-results-nu.xlsx"))
    df.sort_values(by="experiment", axis=0, inplace=True, key=get_key)

    for name, short_name in short_names.items():
        # df.loc[df["experiment"] == name, "experiment"] = short_name
        df.loc[df["experiment"] == name, "experiment"] = name.replace("-", "\n")

    df.drop("Unnamed: 0", axis=1, inplace=True)
    df_mean = df.groupby(by=["experiment", "CS+", "CS-", "test_1", "test_2", "repeat"]).mean()
    df_mean = df_mean.unstack(level="repeat")
    df_std = df.groupby(by=["experiment", "CS+", "CS-", "test_1", "test_2", "repeat"]).std()
    df_std = df_std.unstack(level="repeat")
    # print(df_mean)
    # print(df_std)
    res = ttest_1samp(df.groupby(by=["experiment", "CS+", "CS-", "test_1", "test_2", "repeat"]), 0)
    print(res)

    plt.figure("t-maze", figsize=(10, 3))

    plt.plot([-1, len(experiments) + 1], [0, 0], 'grey', lw=2, zorder=0)
    sns.boxplot(x="experiment", y="PI", hue="repeat", palette="rocket", data=df)
    plt.ylim(-1, 1)
    plt.yticks([-1, 0, 1])
    # plt.xticks(rotation=50)
    plt.xlabel("")
    plt.legend(ncol=10, loc='lower center')

    plt.tight_layout()

    print(df)

    plt.show()


if __name__ == '__main__':
    main(*sys.argv)
