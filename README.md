# The Incentive Complex

Python replication for the results from the eLife manuscript:

Name and DOI/ref

The "**incentive complex**" is a model of the mushroom body in the fruit fly brain that recieves sensory input and reinforcements and modulates the motivation state of the animal. In this model, the motivation states are limited into: *attraction* and *avoidance*. An extenstion of this is the "**wheel of incentives**",
which creates a ranking of available tasks with respect to their importance given the current context.

## Environment

In order to be able to use this code, the required packages are listed below:
* [Python 3.7](https://www.python.org/downloads/release/python-370/)
* [NumPy](https://numpy.org/)
* [Matplotlib](https://matplotlib.org/)
* [pandas](https://pandas.pydata.org/)

## Usage

You can find the scripts that create the plots in the manuscript
in the [examples](examples) directory and run them using: 
```commandline
python any_file_you_want.py [--option value|--flag]
```
It is necessary to add the [src](src) directory to the PATH variable or
install the package.

You can reproduce the results by running:
```commandline
python run_subcircuit.py --nb-kc 10 --sm --rm --ltm --rrm --rfm --mam
python run_twinspokemodel.py --nb-kc 10 --only-nids --structure
python run_twinspokemodel.py --nb-kc 10 --only-nids --values --weights
python run_wheelmodel.py --nb-kc 10 --only-nids --structure
python run_wheelmodel.py --nb-kc 10 --only-nids --values --weights
python run_data_analysis.py --stats --only-nids --verbose
python run_arena_paths.py
python run_arena_stats.py
python create_paths.py
```
where `--nb-kc` specifies the number of KCs (default is 10), `--odour1` specifies the
number of KCs associated to odour A and `--odour2` specifies the number of KCs
associated to odour B; `--sm`, `--rm`, `--ltm`, `--rrm`, `--rfm` and `--mam` flags
the generating of the sub-circuit results for the SM, RM, LTM, RRM, RFM and MAM
sub-circuits respectively; `--only-nids` plots only the neurons associated to the
twin-spoke model; `--structure`, `--values` and `--weights` enable plotting of the
structure, responses and KC-MBON synaptic weights of the model over time respectively;
`--stats` prints the logistics of flies in the data-set; and `--verbose` allows
printing in the console during the processing of the files.

## Author

The code written by [Evripidis Gkanias](https://evgkanias.github.io/).

## Copyright

Copyright &copy; 2021, Insect robotics Group, Institute of Perception,
Action and Behaviour, School of Informatics, the University of Edinburgh.
