"""
The basic mushroom body structure that contains the background attributes, properties and processing including the
learning rules.
"""

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2021, Insect Robotics Group," \
                "Institute of Perception, Action and Behaviour," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "GPLv3+"
__version__ = "v1.1-alpha"
__maintainer__ = "Evripidis Gkanias"

from .routines import extinction_routine, unpaired_routine, reversal_routine

import numpy as np

from copy import copy


class MBModel(object):
    def __init__(self, nb_pn=2, nb_kc=10, nb_mbon=6, nb_dan=6, learning_rule="default", nb_trials=26,
                 nb_timesteps=3, nb_kc_odour=5, nb_kc_odour_1=None, nb_kc_odour_2=None, leak=0., sharp_changes=True,
                 nb_active_kcs=5, rng=np.random.RandomState(2021)):
        """
        The model of the mushroom body from the Drosophila melanogaster brain. It creates the connections from the
        Kenyon cells (KCs) to the output neurons (MBONs), from the MBONs to the dopaminergic neurons (DANs) and from
        the DANs to the connections from the KCs to MBONs. It takes as input a routine and produces the responses and
        weights of the mushroom body for every time-step.

        Parameters
        ----------
        nb_pn: int, optional
            number of projection neurons (PNs). Default is 2
        nb_kc: int, optional
            number of intrinsic neurons (KCs). Default is 10
        nb_mbon: int, optional
            number of extrinsic output neurons (MBONs). Default is 6
        nb_dan: int, optional
            number of extrinsic reinforcement neurons (DANs). Default is 6
        learning_rule: {"dpr", "rw", "default"}
            the learning rule to use; one of "dpr" or "rw". Default is "dpr"
        nb_trials: int, optional
            number of trials that the experiments will run. Default is 24
        nb_timesteps: int, optional
            number of time-steps that each of the trials will run. Default is 3
        nb_kc_odour: int, optional
            number of KCs associated to each odour. Default is 5
        nb_kc_odour_1: int, optional
            number of KCs associated to the first odour. Default is nb_kc_odour
        nb_kc_odour_2: int, optional
            number of KCs associated to the second odour. Default is nb_kc_odour
        leak: float, optional
            the leak parameter of the leaky-ReLU activation function denotes the scale of the negative part of the
            function. Default is 0, which mean no negative part
        sharp_changes: bool, optional
            when true allows sharp changes in the values. Default is True
        nb_active_kcs: float, optional
            we assume that a fixed number of KCs is active in every time-step. Default is 2.
        rng : np.random.RandomState
            the random pattern generator
        """
        self.nb_trials = nb_trials
        self.nb_timesteps = nb_timesteps
        self.__leak = np.absolute(leak)
        self._learning_rule = learning_rule
        self.__routine_name = ""
        self.us_dims = 2  # dimensions of US signal
        self._t = 0  # internal time variable
        self._sharp_changes = sharp_changes
        self._nb_active_kcs = nb_active_kcs
        self.rng = rng

        # Set the PN-to-KC weights
        self.w_p2k = np.zeros((nb_pn, nb_kc), dtype=float)
        if nb_kc_odour_1 is None:
            nb_kc_odour_1 = nb_kc_odour
        if nb_kc_odour_2 is None:
            nb_kc_odour_2 = nb_kc_odour
        kc_p_pn = []
        for p in range(nb_pn):
            if p == 0:
                kc_p_pn.append(nb_kc_odour_1)
            elif p == 1:
                kc_p_pn.append(nb_kc_odour_2)
            else:
                kc_p_pn.append(nb_kc_odour)
        for p, nb_odour in enumerate(kc_p_pn):
            s = int(p * nb_kc / nb_pn)
            if s + nb_odour > nb_kc:
                s = nb_kc - nb_odour
            e = s + nb_odour
            # self.w_p2k[p, s:e] = 1
            self.w_p2k[p, s:e] = nb_pn / self._nb_active_kcs

        # create map from the US to the extrinsic neurons
        self.w_u2d = np.zeros((self.us_dims, nb_dan + nb_mbon), dtype=float)
        self.w_u2d[:, :nb_dan] = np.eye(self.us_dims, nb_dan) * 2.
        if nb_dan / self.us_dims > 1:
            self.w_u2d[:, self.us_dims:nb_dan] = np.eye(self.us_dims, nb_dan - self.us_dims) * 2

        # occupy space for the responses of the extrinsic neurons
        self._v = np.zeros((nb_trials * nb_timesteps + 1, nb_dan + nb_mbon), dtype=float)
        # bias is initialised as the initial responses of the neurons
        self.bias = self._v[0]
        # set-up the constraints for the responses and weights
        self.v_max = np.array([+2.] * (nb_dan+nb_mbon))
        self.v_min = np.array([-2.] * (nb_dan+nb_mbon))
        self.w_max = np.array([+2.] * (nb_dan+nb_mbon))
        self.w_min = np.array([-2.] * (nb_dan+nb_mbon))

        # initialise the KC-to-MBON weights (zero weight for DANs and 1 for MBONs)
        self.w_k2m = np.array([[[0.] * nb_dan + [1.] * nb_mbon] * nb_kc] * (nb_trials * nb_timesteps + 1))
        # set the resting values to be the initial weights
        self.w_rest = self.w_k2m[0, 0]

        # MBON-to-MBON and MBON-to-DAN synaptic weights
        self._w_m2v = np.zeros((nb_dan+nb_mbon, nb_dan+nb_mbon), dtype=float)
        # DAN-to-KCtoMBON synaptic weights for calculating the dopaminergic factor
        self._w_d2k = np.array([
            ([0.] * nb_dan + [-float(m == d) for m in range(nb_mbon)]) for d in range(nb_dan+nb_mbon)], dtype=float)

        self.nb_pn = nb_pn  # number of projection neurons (PN)
        self.nb_kc = nb_kc  # number of Kenyon cells (KC)
        self.nb_mbon = nb_mbon  # number of mushroom body output neurons (MBON)
        self.nb_dan = nb_dan  # number of dopaminergic neurons (DAN)

        # odour A and B patterns
        self.csa = np.array([1., 0.])
        self.csb = np.array([0., 1.])

        # interventions to the neural activity
        self.intervention = np.zeros_like(self._v)

        # names of the neurons for plotting
        self.names = ["DAN-%d" % (i+1) for i in range(nb_dan)] + ["MBON-%d" % (i+1) for i in range(nb_mbon)]
        # IDs of the neurons we want to plot
        self.neuron_ids = []

    @property
    def w_m2v(self):
        """
        The MBON-to-DAN and MBON-to-MBON synaptic weights
        """
        return np.eye(*self._w_m2v.shape) + self._w_m2v

    @property
    def w_d2k(self):
        """
        The DAN-to-KCtoMBON synaptic weights that transform DAN activity to the dopaminergic factor
        """
        return self._w_d2k

    @property
    def routine_name(self):
        """
        The name of the running routine
        """
        return self.__routine_name

    @routine_name.setter
    def routine_name(self, v):
        self.__routine_name = v

    def _a(self, v, v_max=None, v_min=None):
        """
        The activation function calls the leaky ReLU function and bounds the output in [v_min, v_max].
        It also adds the assigned intervention to the neural activity.

        Parameters
        ----------
        v: np.ndarray | float
            the raw responses of the neurons
        v_max: np.ndarray | float
            the maximum allowing output
        v_min: np.ndarray | float
            the minimum allowing output
        """
        if v_max is None:
            v_max = self.v_max
        if v_min is None:
            v_min = self.v_min
        return leaky_relu(v + self.intervention[np.minimum(self._t, self.intervention.shape[0] - 1)],
                          alpha=self.__leak, v_max=v_max, v_min=v_min)

    def add_intervention(self, neuron_name, inhibit=False, excite=False):
        """
        Adds interventions in the given neuron starting from the current time-step.

        Parameters
        ----------
        neuron_name: str
            the name of the neuron to add the intervention to
        inhibit: bool
            whether the intervention is inhibitory
        excite: bool
            whether the intervention is excitatory
        """
        i = self.names.index(neuron_name)
        self.intervention[self._t:, i] = 5. * float(excite) - 5. * float(inhibit)

    def __call__(self, *args, **kwargs):
        """
        Running the feed-forward process of the model for the given routine.

        Parameters
        ----------
        extinction: bool
            whether to run the 'extinction' routine. Default is False.
        unpaired: bool
            whether to run the 'unpaired' routine. Default is False.
        reversal: bool
            whether to run the 'reversal' routine. Default is False.
        routine: Callable[MBModel, Generator[Tuple[int, int, float, None], Any, None]]
            a customised routine. Default is the 'reversal_routine'.
        repeat: int
            number of repeats of each time-step in order to smoothen the bias of order of the updates. Default is 4
        """
        if kwargs.get('extinction', False):
            routine = extinction_routine(self)
        elif kwargs.get('unpaired', False):
            routine = unpaired_routine(self)
        elif kwargs.get('reversal', False):
            routine = reversal_routine(self)
        else:
            routine = kwargs.get('routine', reversal_routine(self))
        repeat = kwargs.get('repeat', 4)
        rng = kwargs.get('rng', self.rng)

        for _, _, cs, us in routine:

            # create dynamic memory for repeating loop
            w_k2m_pre, w_k2m_post = self.w_k2m[self._t].copy(), self.w_k2m[self._t].copy()
            v_pre, v_post = self._v[self._t].copy(), self._v[self._t].copy()

            # feed forward responses: PN(CS) -> KC
            # k = np.maximum(cs @ self.w_p2k + rng.rand(self.nb_kc) * .001, 0)
            k = cs @ self.w_p2k + rng.rand(self.nb_kc) * .001
            k[np.argsort(k)[:-self._nb_active_kcs]] = 0.

            # feed forward responses: KC -> MBON, US -> DAN
            mb = k @ w_k2m_pre + us @ self.w_u2d + self.bias
            mb = np.clip(mb, -100., 100.)

            eta = float(1) / float(repeat)
            for r in range(repeat):

                # Step 1: internal values update
                v_post = self.update_values(v_pre, mb)

                # Step 2: synaptic weights update
                w_k2m_post = self.update_weights(k, v_post, w_k2m_pre)

                # update dynamic memory for repeating loop
                v_pre += eta * (v_post - v_pre)
                w_k2m_pre += eta * (w_k2m_post - w_k2m_pre)

            # store values and weights in history
            self._v[self._t + 1], self.w_k2m[self._t + 1] = v_post, w_k2m_post

    def update_weights(self, kc, v, w_k2m):
        """
        Updates the KC-to-MBON synaptic weights.

        Parameters
        ----------
        kc: np.ndarray
            the responses of the KCs -- k(t)
        v: np.ndarray
            the responses of the DANs and MBONs (extrinsic neurons) -- d(t), m(t)
        w_k2m: np.ndarray
            the KC-to-MBON synaptic weights -- w_k2m(t)

        Returns
        -------
        w_post: np.ndarray
            the new weights -- w_k2m(t + 1)
        """

        # scale the learning based on the number of in-trial time-steps
        eta_w = 1. / np.maximum(float(self.nb_timesteps - 1), 1)

        # reformat the structure of the k(t)
        k = kc[..., np.newaxis]

        k = np.maximum(k, 0)  # do not learn negative values

        # the dopaminergic factor
        D = np.maximum(v, 0).dot(self.w_d2k)

        if self._learning_rule in ["dopaminergic", "dpr", "default"]:
            w_new = dopaminergic(k, D, w_k2m, eta=eta_w, w_rest=self.w_rest)
        elif self._learning_rule in ["reward-prediction-error", "rpe", "prediction-error"]:
            w_new = reward_prediction_error(k, v, D, w_k2m, eta=eta_w, w_rest=self.w_rest)
        elif callable(self._learning_rule):
            w_new = self._learning_rule(k, v, D, w_k2m, eta_w, self.w_rest)
        else:
            # if the learning rule is not valid then do nothing
            w_new = w_k2m

        # negative weights are not allowed
        return np.clip(w_new, 0, 50)

    def update_values(self, v_pre, v):
        """
        Updates the responses of the neurons using the feedback connections and the previous responses.

        Parameters
        ----------
        v_pre: np.ndarray
            the responses of the DANs and MBONs (extrinsic neurons) in the previous time-step -- d(t-1), m(t-1)
        v: np.ndarray
            the responses of teh DANs and MBONs in the new time-step based only on the sensory input

        Returns
        -------
        v_post: np.ndarray
            the responses with contributions from feedback connections
        """
        eta_v = 1. / float(self.nb_timesteps)
        if self._sharp_changes:
            eta_v = np.power(eta_v, eta_v)

        # update the responses with a smooth transition between the time-steps
        v_temp = v_pre + eta_v * (v.dot(self.w_m2v) - 2 * v_pre)
        # v_temp = v_pre + eta_v * (v.dot(self.w_m2v) - v_pre)

        return self._a(v_temp)

    def copy(self):
        """
        Creates a clone of the instance.

        Returns
        -------
        copy: MBModel
            another instance of exactly the same class and parameters.
        """
        return copy(self)

    def __copy__(self):
        m = self.__class__()
        for att in self.__dict__:
            m.__dict__[att] = copy(self.__dict__[att])

        return m

    def __repr__(self):
        s = "MBModel("
        if self.__routine_name != "":
            s += "routine='" + self.__routine_name + "'"
        else:
            s += "routine=None"
        if self._learning_rule != "default":
            s += ", lr='" + self._learning_rule + "'"
        s += ")"
        return s


def leaky_relu(v, alpha=.1, v_max=np.inf, v_min=-np.inf):
    """
    The leaky-ReLU activation function; if v >= 0, it returns v; if v < 0, it returns alpha * v.

    Parameters
    ----------
    v: np.ndarray | float
        the input value
    alpha: float
        the discount parameter
    v_max: np.ndarray | float
        the upper bound of the function
    v_min: np.ndarray | float
        the lower bound of the function

    Returns
    -------
    a: np.ndarray | float
        the transformed value
    """
    return np.clip(v, np.maximum(alpha * v, v_min), v_max)


def reward_prediction_error(r_pre, r_post, D, w, eta=1., w_rest=1.):
    """
    The reward-prediction-error learning rule introduced in [1]_.

        tau * dw / dt = r_pre * (rein - r_post - w_rest)

        tau = 1 / learning_rate

    When KC > 0 and DAN > W - W_rest increase the weight (if KC < 0 it is reversed).
    When KC > 0 and DAN < W - W_rest decrease the weight (if KC < 0 it is reversed).
    When KC = 0 no learning happens.

    Parameters
    ----------
    w: np.ndarray
        the current synaptic weights.
    r_pre: np.ndarray
        the pre-synaptic responses.
    r_post: np.ndarray
        the post-synaptic responses.
    D: np.ndarray
        the reinforcement signal.
    eta: float, optional
        the learning rate.
    w_rest: np.ndarray | float
        the resting value for the synaptic weights.

    Returns
    -------
    w_post: np.ndarray
        the updated synaptic weights

    Notes
    -----
    .. [1] Rescorla, R. A. & Wagner, A. R. A theory of Pavlovian conditioning: Variations in the effectiveness of
       reinforcement and nonreinforcement. in 64–99 (Appleton-Century-Crofts, 1972).
    """
    return w + eta * r_pre * (D - r_post + w_rest)


def dopaminergic(r_pre, D, w, eta=1., w_rest=1.):
    """
    The dopaminergic plasticity rule introduced in Gkanias et al (2021). Reinforcement here is assumed to be the
    dopaminergic factor.

        tau * dw / dt = rein * [r_pre + w(t) - w_rest]

        tau = 1 / eta

    When DAN > 0 and KC > W - W_rest increase the weight (if DAN < 0 it is reversed).
    When DAN > 0 and KC < W - W_rest decrease the weight (if DAN < 0 it is reversed).
    When DAN = 0 no learning happens.

    Parameters
    ----------
    w: np.ndarray
        the current synaptic weights.
    r_pre: np.ndarray
        the pre-synaptic responses.
    D: np.ndarray
        the dopaminergic factor.
    eta: float, optional
        the learning rate.
    w_rest: np.ndarray | float
        the resting value for the synaptic weights.

    Returns
    -------
    w_post: np.ndarray
        the updated synaptic weights
    """
    return w + eta * D * (r_pre + w - w_rest)
