# The Incentive Circuit

Python replication for the results from the eLife manuscript:

Name and DOI/ref

The "**incentive circuit**" (IC) is a model of the mushroom body in the fruit fly brain
that receives sensory input and reinforcements and modulates the motivation state of
the animal. In this model, the motivation states are limited into: *attraction* and
*avoidance*. An extension of this is the "**incentive wheel**" (IW),  which creates a
ranking of available tasks with respect to their importance given the current context.
Although IC can explain complicated dynamics of behaviours regarding olfactory
conditioning and involving the attraction and avoidance behaviours, IW is able to
trigger more complicated dynamics by driving a bigger variety of behaviours, like
feeding, sleeping, escaping or attacking.

## Environment

In order to be able to use this code, the required packages are listed below:
* [Python 3.7](https://www.python.org/downloads/release/python-370/)
* [NumPy](https://numpy.org/)
* [Matplotlib](https://matplotlib.org/)
* [pandas](https://pandas.pydata.org/)

## Usage

The directory [notebooks](notebooks) contains an
[iPython notebook](notebooks/incentive_body.ipynb) which reproduces all the results
and associates them to the manuscript. Alternatively, you can find the scripts that create the plots in the manuscript
in the [examples](examples) directory and run them using: 
```commandline
python any_file_you_want.py --flag --option value
```
It is necessary to add the [src](src) directory to the PATH variable or
install the package.

You can reproduce the results by running:
```commandline
python3 run_subcircuit.py --nb-kc 10 --sm --rm --ltm --rrm --rfm --mam
python3 run_twinspokemodel.py --nb-kc 10 --only-nids --structure
python3 run_twinspokemodel.py --nb-kc 10 --only-nids --values --weights
python3 run_wheelmodel.py --nb-kc 10 --only-nids --structure
python3 run_wheelmodel.py --nb-kc 10 --only-nids --values --weights
python3 run_data_analysis.py --stats --only-nids --verbose
python3 create_paths.py --nb-flies 100 --nb-time-steps 100 
python3 run_arena_paths.py
python3 run_arena_stats.py
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
