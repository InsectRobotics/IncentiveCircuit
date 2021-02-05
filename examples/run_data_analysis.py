from imaging import load_data, plot_phase_overlap_mean_responses_from_data
from tools import read_arg


if __name__ == '__main__':
    df = load_data("B+")
    nb_neurons, nb_flies, nb_flies_min, nb_flies_max = 0, 0, 14, 0

    experiment = read_arg(["-e", "--experiment"], vtype=str, default="B+")
    verbose = read_arg(["-v", "--verbose"])
    for name in df[experiment].index:
        nb_flies += df[experiment][name].shape[1]
        nb_neurons += 1
        if nb_flies_min > df[experiment][name].shape[1]:
            nb_flies_min = df[experiment][name].shape[1]
        if nb_flies_max < df[experiment][name].shape[1]:
            nb_flies_max = df[experiment][name].shape[1]
        if verbose:
            print(name, df[experiment][name].shape)

    if read_arg(["-s", "--stats", "--statistics"]):

        print("#neurons:", nb_neurons)
        print("#flies:", nb_flies)
        print("min #flies/neuron:", nb_flies_min)
        print("max #flies/neuron:", nb_flies_max)
        print("mean #flies/neuron:", nb_flies / nb_neurons)

    if read_arg(["-a", "--all"]):
        plot_phase_overlap_mean_responses_from_data(df, experiment)
    else:
        neurons = [33, 39, 13, 16, 21, 42, 14, 17, 41, 28, 12, 2]
        plot_phase_overlap_mean_responses_from_data(df, experiment, nids=neurons, only_nids=read_arg(["--only-nids"]))
