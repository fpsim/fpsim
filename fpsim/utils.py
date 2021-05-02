'''
File for storing utilities and probability calculators needed to run FP model
'''


import numpy as np # Needed for a few things not provided by pl
import pylab as pl
import sciris as sc
import numba as nb
from . import defaults as fpd


# Specify all externally visible things this file defines
__all__ = ['set_seed', 'bt', 'bc', 'rbt', 'mt', 'fixaxis', 'dict2obj']

usenumba  = True

if usenumba:
    func_decorator = nb.njit

else:
    def func_decorator(*args, **kwargs):
        def wrap(func): return func
        return wrap
def class_decorator(*args, **kwargs):
    ''' Was class_decorator = nb.jitclass, but not used currently and removed from Numba '''
    def wrap(cls): return cls
    return wrap

def set_seed(seed=None):
    ''' Reset the random seed -- complicated because of Numba '''

    @func_decorator
    def set_seed_numba(seed):
        return np.random.seed(seed)

    def set_seed_regular(seed):
        return np.random.seed(seed)

    if seed is not None:
        set_seed_numba(seed)
        set_seed_regular(seed)
    return


@func_decorator((nb.float64,), cache=True)  # These types can also be declared as a dict, but performance is much slower...?
def bt(prob):
    ''' A simple Bernoulli (binomial) trial '''
    return np.random.random() < prob  # Or rnd.random() < prob, np.random.binomial(1, prob), which seems slower


@func_decorator((nb.float64, nb.int64), cache=True)
def bc(prob, repeats):
    ''' A binomial count '''
    return np.random.binomial(repeats, prob)  # Or (np.random.rand(repeats) < prob).sum()


@func_decorator((nb.float64, nb.int64), cache=True)
def rbt(prob, repeats):
    ''' A repeated Bernoulli (binomial) trial '''
    return np.random.binomial(repeats, prob) > 0  # Or (np.random.rand(repeats) < prob).any()


@func_decorator((nb.float64[:],), cache=True)
def mt(probs):
    ''' A multinomial trial '''
    return np.searchsorted(np.cumsum(probs), np.random.random())


def n_multinomial(probs, n): # No speed gain from Numba
    '''
    An array of multinomial trials.

    Args:
        probs (array): probability of each outcome, which usually should sum to 1
        n (int): number of trials

    Returns:
        Array of integer outcomes

    **Example**::

        outcomes = cv.multinomial(np.ones(6)/6.0, 50)+1 # Return 50 die-rolls
    '''
    return np.searchsorted(np.cumsum(probs), np.random.random(n))


def n_binomial(prob, n):
    '''
    Perform multiple binomial (Bernolli) trials

    Args:
        prob (float): probability of each trial succeeding
        n (int): number of trials (size of array)

    Returns:
        Boolean array of which trials succeeded

    **Example**::

        outcomes = cv.n_binomial(0.5, 100) # Perform 100 coin-flips
    '''
    return np.random.random(n) < prob


def binomial_arr(prob_arr): # No speed gain from Numba
    '''
    Binomial (Bernoulli) trials each with different probabilities.

    Args:
        prob_arr (array): array of probabilities

    Returns:
         Boolean array of which trials on the input array succeeded

    **Example**::

        outcomes = cv.binomial_arr([0.1, 0.1, 0.2, 0.2, 0.8, 0.8]) # Perform 6 trials with different probabilities
    '''
    return np.random.random(len(prob_arr)) < prob_arr


def annprob2ts(prob_annual, timestep=1):
    ''' Convert an annual probability into a timestep probability '''
    prob_timestep = 1 - ((1-prob_annual)**(timestep/fpd.mpy))
    return prob_timestep



@func_decorator((nb.float64[:], nb.float64, nb.float64), cache=True)
def numba_miscarriage_prob(miscarriage_rates, age, resolution):
    '''Run interpolation eval to check for probability of miscarriage here'''
    miscarriage_prob = miscarriage_rates[int(round(age*resolution))]
    return miscarriage_prob


def fixaxis(useSI=True):
    ''' Fix the plotting '''
    pl.legend()  # Add legend
    sc.setylim()  # Rescale y to start at 0
    if useSI:
        sc.SIticks()
    return


def dict2obj(d):
    ''' Convert a dictionary to an object '''
    o = sc.prettyobj()
    for k,v in d:
        setattr(o, k, v)
    return o