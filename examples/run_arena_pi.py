from incentive.tools import read_arg

import numpy as np
import os

# the directory of the file
__dir__ = os.path.dirname(os.path.abspath(__file__))
# the directory of the data
__data_dir__ = os.path.realpath(os.path.join(__dir__, "..", "data", "arena"))


if __name__ == '__main__':
    from incentive.plot import plot_arena_box
    from incentive.arena import load_arena_stats

    rw = read_arg(["-rw", "--rescorla-wagner"])
    file_names = [read_arg(["-f"], vtype=str, default=None)]
    directory = read_arg(["-d", "--dir"], vtype=str, default=__data_dir__)

    if file_names[0] is None:
        file_names = os.listdir(directory)

    df = load_arena_stats(file_names, rw=rw)

    df["avoid A"] = df["dist_A"] / 0.6 - 1
    df["avoid B"] = df["dist_B"] / 0.6 - 1
    df["avoid A/B"] = np.max([df["dist_A"], df["dist_B"]], axis=0) / 0.6 - 1
    df["attract A"] = df["dist_A"] / 0.6 - 1
    df["attract B"] = df["dist_B"] / 0.6 - 1
    df["attract A/B"] = np.min([df["dist_A"], df["dist_B"]], axis=0) / 0.6 - 1
    print(df.columns)

    plot_arena_box(df, "%sarena-box" % ("rw-" if rw else ""))
