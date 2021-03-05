from incentive.tools import read_arg

import os

# the directory of the file
__dir__ = os.path.dirname(os.path.abspath(__file__))
# the directory of the data
__data_dir__ = os.path.realpath(os.path.join(__dir__, "..", "data", "arena"))


if __name__ == '__main__':
    from incentive.plot import plot_arena_stats
    from incentive.arena import load_arena_stats

    rw = read_arg(["-rw", "--rescorla-wagner"])
    file_names = [read_arg(["-f"], vtype=str, default=None)]

    if file_names[0] is None:
        file_names = os.listdir(__data_dir__)

    df = load_arena_stats(file_names, rw=rw)

    plot_arena_stats(df, "%sarena-stats" % ("rw-" if rw else ""))
