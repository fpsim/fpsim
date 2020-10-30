'''
File for storing utilities and probability calculators needed to run FP model
'''


import numpy as np # Needed for a few things not provided by pl
import pylab as pl
import sciris as sc
import numba as nb


# Specify all externally visible things this file defines
__all__ = ['set_seed', 'bt', 'bc', 'rbt', 'mt', 'fixaxis']

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


@func_decorator((nb.float64,))  # These types can also be declared as a dict, but performance is much slower...?
def bt(prob):
    ''' A simple Bernoulli (binomial) trial '''
    return np.random.random() < prob  # Or rnd.random() < prob, np.random.binomial(1, prob), which seems slower


@func_decorator((nb.float64, nb.int64))
def bc(prob, repeats):
    ''' A binomial count '''
    return np.random.binomial(repeats, prob)  # Or (np.random.rand(repeats) < prob).sum()


@func_decorator((nb.float64, nb.int64))
def rbt(prob, repeats):
    ''' A repeated Bernoulli (binomial) trial '''
    return np.random.binomial(repeats, prob) > 0  # Or (np.random.rand(repeats) < prob).any()


@func_decorator((nb.float64[:],))
def mt(probs):
    ''' A multinomial trial '''
    return np.searchsorted(np.cumsum(probs), np.random.random())

@func_decorator((nb.float64, nb.float64[:], nb.float64[:], nb.float64, nb.float64, nb.float64))
def numba_mortality_prob(t, mortality_fn, trend_fn, age, resolution, mpy):
    mortality_eval = mortality_fn[int(round(age * resolution))]
    trend_eval = trend_fn[int(round(t * resolution))]
    prob_annual = mortality_eval*trend_eval
    prob_annual = np.median(np.array([0, prob_annual, 1])) # Some annual probs go slightly over 1 in older age groups when corrected for mortality trend spline
    #prob_month = prob_annual/mpy
    prob_month = 1 - ((1-prob_annual)**(1/mpy))
    return prob_month


@func_decorator((nb.float64[:], nb.float64, nb.float64, nb.float64, nb.float64))
def numba_preg_prob(fertility_fn, age, resolution, method_eff, mpy):
    ''' Pull this out here since it's the most computationally expensive '''
    preg_eval = fertility_fn[int(round(age*resolution))]
    prob_annual = ((1-method_eff) * preg_eval)
    #prob_month = prob_annual/mpy
    prob_month = 1 - ((1-prob_annual)**(1/mpy))
    #trend_eval = trend_fn[int(round(t*resolution))]

    return prob_month


@func_decorator((nb.float64[:], nb.float64, nb.float64))
def numba_activity_prob(sexual_activity, age, resolution):
    '''Run interpolation eval to check for probability of sexual activity here'''
    sexually_active_prob = sexual_activity[int(round(age*resolution))]
    return sexually_active_prob

@func_decorator((nb.float64[:], nb.float64, nb.float64))
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
