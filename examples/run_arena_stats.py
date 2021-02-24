from tools import read_arg

import re
import os

# the directory of the file
__dir__ = os.path.dirname(os.path.abspath(__file__))
# the directory of the data
__data_dir__ = os.path.realpath(os.path.join(__dir__, "..", "data", "arena"))


if __name__ == '__main__':
    from plot import plot_arena_stats, plot_arena_box

    import pandas as pd
    import numpy as np

    rw = read_arg(["-rw", "--rescorla-wagner"])
    file_names = [read_arg(["-f"], vtype=str, default=None)]

    if file_names[0] is None:
        file_names = os.listdir(__data_dir__)

    d_names = ["susceptible", "reciprocal", "long-term memory", "reinforcement",
               "paired odour", "phase", "angle"]
    d_raw = [[], [], [], [], [], [], []]

    for fname in file_names:
        if rw:
            pattern = r'rw-arena-([\w]+)-(s{0,1})(r{0,1})(m{0,1})(a{0,1})(b{0,1})'
        else:
            pattern = r'arena-([\w]+)-(s{0,1})(r{0,1})(m{0,1})(a{0,1})(b{0,1})'
        details = re.findall(pattern, fname)
        if len(details) < 1:
            continue
        punishment = 'quinine' in details[0]
        susceptible = 's' in details[0]
        reciprocal = 'r' in details[0]
        ltm = 'm' in details[0]
        only_a = 'a' in details[0]
        only_b = 'b' in details[0]
        name = fname[:-4]

        data = np.load(os.path.join(__data_dir__, fname))["data"]

        nb_flies, nb_time_steps = data.shape

        e_pre, s_post = int(.2 * nb_time_steps), int(.5 * nb_time_steps)

        d_raw[0].extend([susceptible] * 3 * nb_flies)
        d_raw[1].extend([reciprocal] * 3 * nb_flies)
        d_raw[2].extend([ltm] * 3 * nb_flies)
        d_raw[3].extend(["punishment" if punishment else "reward"] * 3 * nb_flies)
        d_raw[4].extend([("A+B" if only_b else "A") if only_a else ("B" if only_b else "AB")] * 3 * nb_flies)
        d_raw[5].extend(["pre"] * nb_flies)
        d_raw[5].extend(["learn"] * nb_flies)
        d_raw[5].extend(["post"] * nb_flies)
        d_raw[6].extend(np.angle(data[:, e_pre]))
        d_raw[6].extend(np.angle(data[:, s_post]))
        d_raw[6].extend(np.angle(data[:, -1]))
        # plot_arena_paths(data, name=name, save=True, show=False)
    d_raw = np.array(d_raw)
    df = pd.DataFrame(d_raw, index=d_names).T
    df["angle"] = np.rad2deg(np.array(df["angle"], dtype=float))
    df["susceptible"] = np.array(df["susceptible"] == "True", dtype=bool)
    df["reciprocal"] = np.array(df["reciprocal"] == "True", dtype=bool)
    df["long-term memory"] = np.array(df["long-term memory"] == "True", dtype=bool)

    plot_arena_stats(df, "%sarena-stats" % ("rw-" if rw else ""))
    # plot_arena_box(df, "arena-box", print_stats=False)
