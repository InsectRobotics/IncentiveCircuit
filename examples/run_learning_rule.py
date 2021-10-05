from incentive.plot import plot_learning_rule

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright 2021, School of Informatics, the University of Edinburgh"
__licence__ = "MIT"
__version__ = "1.1-alpha"
__maintainer__ = "Evripidis Gkanias"
__email__ = "ev.gkanias@ed.ac.uk"
__status__ = "Production"


if __name__ == '__main__':
    plot_learning_rule(wrt_k=False, wrt_d=False, colour_bar=True)
    # plot_learning_rule(colour_bar=True)
