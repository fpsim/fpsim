'''
File for storing utilities and probability calculators needed to run FP model
'''

import numpy as np
import sciris as sc
import numba as nb
from . import defaults as fpd
from . import version as fpv


# Specify all externally visible things this file defines
__all__ = ['set_seed', 'bt', 'bc', 'rbt', 'mt', 'sample', 'match_ages']
__all__ += ['DuplicateNameException']


@nb.jit((nb.float64[:], nb.float64, nb.float64), cache=True, nopython=True)
def match_ages(age, age_low, age_high):
    ''' Find ages between age low and age_high '''
    match_low  = (age >= age_low)
    match_high = (age <  age_high)
    return match_low & match_high


def set_seed(seed=None):
    ''' Reset the random seed -- complicated because of Numba '''

    @nb.njit
    def set_seed_numba(seed):
        return np.random.seed(seed)

    def set_seed_regular(seed):
        return np.random.seed(seed)

    if seed is not None:
        set_seed_numba(seed)
        set_seed_regular(seed)
    return


@nb.njit((nb.float64,), cache=True)  # These types can also be declared as a dict, but performance is much slower...?
def bt(prob):
    ''' A simple Bernoulli (binomial) trial '''
    return np.random.random() < prob  # Or rnd.random() < prob, np.random.binomial(1, prob), which seems slower


@nb.njit((nb.float64, nb.int64), cache=True)
def bc(prob, repeats):
    ''' A binomial count '''
    return np.random.binomial(repeats, prob)  # Or (np.random.rand(repeats) < prob).sum()


@nb.njit((nb.float64, nb.int64), cache=True)
def rbt(prob, repeats):
    ''' A repeated Bernoulli (binomial) trial '''
    return np.random.binomial(repeats, prob) > 0  # Or (np.random.rand(repeats) < prob).any()


@nb.njit((nb.float64[:],), cache=True)
def mt(probs):
    ''' A multinomial trial '''
    return np.searchsorted(np.cumsum(probs), np.random.random())


@nb.njit((nb.float64[:], nb.int64), cache=True)
def n_multinomial(probs, n):
    '''
    An array of multinomial trials.

    Equivalent to, but faster than, `np.random.choice(len(probs), size=n, p=probs)`

    Args:
        probs (array): probability of each outcome, which usually should sum to 1
        n (int): number of trials

    Returns:
        Array of integer outcomes

    **Example**::

        outcomes = fp.n_multinomial(np.ones(6)/6.0, 50)+1 # Return 50 die-rolls
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
    prob_timestep = 1 - ((1-np.minimum(1,prob_annual))**(timestep/fpd.mpy))
    return prob_timestep



@nb.njit((nb.float64[:], nb.float64, nb.float64), cache=True)
def numba_miscarriage_prob(miscarriage_rates, age, resolution):
    '''Run interpolation eval to check for probability of miscarriage here'''
    miscarriage_prob = miscarriage_rates[int(round(age*resolution))]
    return miscarriage_prob


def set_metadata(obj):
    ''' Set standard metadata for an object '''
    obj.created = sc.now()
    obj.version = fpv.__version__
    obj.git_info = sc.gitinfo(verbose=False)
    return


def sample(dist='uniform', par1=0, par2=1, size=1, **kwargs):
    '''
    Draw a sample from the distribution specified by the input. The available
    distributions are:

    - 'uniform'       : uniform distribution from low=par1 to high=par2; mean is equal to (par1+par2)/2
    - 'normal'        : normal distribution with mean=par1 and std=par2
    - 'lognormal'     : lognormal distribution with mean=par1 and std=par2 (parameters are for the lognormal distribution, *not* the underlying normal distribution)
    - 'normal_pos'    : right-sided normal distribution (i.e. only positive values), with mean=par1 and std=par2 *of the underlying normal distribution*
    - 'normal_int'    : normal distribution with mean=par1 and std=par2, returns only integer values
    - 'lognormal_int' : lognormal distribution with mean=par1 and std=par2, returns only integer values
    - 'poisson'       : Poisson distribution with rate=par1 (par2 is not used); mean and variance are equal to par1
    - 'neg_binomial'  : negative binomial distribution with mean=par1 and k=par2; converges to Poisson with k=∞

    Args:
        dist (str):   the distribution to sample from
        par1 (float): the "main" distribution parameter (e.g. mean)
        par2 (float): the "secondary" distribution parameter (e.g. std)
        size (int):   the number of samples (default=1)
        kwargs (dict): passed to individual sampling functions

    Returns:
        A length N array of samples

    **Examples**::

        fp.sample() # returns Unif(0,1)
        fp.sample(dist='normal', par1=3, par2=0.5) # returns Normal(μ=3, σ=0.5)
        fp.sample(dist='lognormal_int', par1=5, par2=3) # returns a lognormally distributed set of values with mean 5 and std 3

    Notes:
        Lognormal distributions are parameterized with reference to the underlying normal distribution (see:
        https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.lognormal.html), but this
        function assumes the user wants to specify the mean and std of the lognormal distribution.

        Negative binomial distributions are parameterized with reference to the mean and dispersion parameter k
        (see: https://en.wikipedia.org/wiki/Negative_binomial_distribution). The r parameter of the underlying
        distribution is then calculated from the desired mean and k. For a small mean (~1), a dispersion parameter
        of ∞ corresponds to the variance and standard deviation being equal to the mean (i.e., Poisson). For a
        large mean (e.g. >100), a dispersion parameter of 1 corresponds to the standard deviation being equal to
        the mean.
    '''

    # Some of these have aliases, but these are the "official" names
    choices = [
        'uniform',
        'normal',
        'normal_pos',
        'normal_int',
        'lognormal',
        'lognormal_int',
    ]

    # Ensure it's an integer
    if size is not None:
        size = int(size)

    # Compute distribution parameters and draw samples
    # NB, if adding a new distribution, also add to choices above
    if   dist in ['unif', 'uniform']: samples = np.random.uniform(low=par1, high=par2, size=size, **kwargs)
    elif dist in ['norm', 'normal']:  samples = np.random.normal(loc=par1, scale=par2, size=size, **kwargs)
    elif dist == 'normal_pos':        samples = np.abs(np.random.normal(loc=par1, scale=par2, size=size, **kwargs))
    elif dist == 'normal_int':        samples = np.round(np.abs(np.random.normal(loc=par1, scale=par2, size=size, **kwargs)))
    elif dist in ['lognorm', 'lognormal', 'lognorm_int', 'lognormal_int']:
        if par1>0:
            mean  = np.log(par1**2 / np.sqrt(par2**2 + par1**2)) # Computes the mean of the underlying normal distribution
            sigma = np.sqrt(np.log(par2**2/par1**2 + 1)) # Computes sigma for the underlying normal distribution
            samples = np.random.lognormal(mean=mean, sigma=sigma, size=size, **kwargs)
        else:
            samples = np.zeros(size)
        if '_int' in dist:
            samples = np.round(samples)
    else:
        errormsg = f'The selected distribution "{dist}" is not implemented; choices are: {sc.newlinejoin(choices)}'
        raise NotImplementedError(errormsg)

    return samples


def piecewise_linear(x, x0, y0, m1, m2):
    '''
    Compute a two-part piecewise linear function at given x values.

    This function calculates the values of a piecewise linear function defined
    by two slopes (k1 and k2), a point of intersection (x0, y0), and an array
    of x values. The function returns an array of corresponding y values based
    on the piecewise linear function.

    Args:
    x (array-like)
        The array of x values at which the piecewise linear function is evaluated
    x0 (float)
        The x-coordinate of the point of intersection between the two linear segments (inflection point)
    y0 (float)
        The y-coordinate of the point of intersection between the two linear segments
    m1 (float)
        The slope of the first linear segment (for x < x0).
    m2 (float)
        The slope of the second linear segment (for x >= x0).

    Returns:
    y : ndarray
        An array of y values corresponding to the piecewise linear function
        evaluated at the input x values.

    **Examples**::
    >>> x_values = np.array([1, 2, 3, 4, 5])
    >>> y_values = piecewise_linear(x_values, 3, 2, 1, -1)
    '''
    return np.piecewise(x, [x < x0], [lambda x:m1*x + y0-m1*x0, lambda x:m2*x + y0-m2*x0])


def logistic_5p(x, a, b, c, d, e):
    '''
    A logistic function with 5 parameters (5p) that enables asymmetry

    Args:
    x (array-like)
       The array of x values at which the function is evaluated
    a (float):
      Minimum value (baseline) as x -> -infinity.
    d (float):
      Maximum value (saturation) as x -> infinity.
    b (float):
      Slope parameter
    c (float):
      Value of x at which the function reaches the midpoint between a and d.
    e (float):
      Exponent parameter controlling asymmetry.
        - If e = 1, the curve is symmetric.
        - If e > 1, the curve asymptotes toward "a" more quickly than it asymptotes toward "d."
        - If e < 1, the curve asymptotes toward "d" more quickly than it asymptotes toward "a."

    Returns:
    y : (ndarray)
        An array of y values corresponding to the piecewise linear function
        evaluated at the input x values.
    '''

    return d + ((a - d)/(1.0 + np.exp(b*(x-c)))**e)


def logistic_5p_dfun(x, a, b, c, d, e):
    '''
    Derivative of the 5 paraemter logistic function, same parameters
    '''
    return b*(a - d)*e*np.exp(b*(-c + x))*(1.0 + np.exp(b*(-c + x)))**(-1.0 - e)


def sigmoid_product(x, a1, b1, a2, b2):
    '''
    A product of two sigmoid functions. A monotonically increasing sigmoidal curve,
    followed by a monotonically decreasing sigmoidal curve.

    Current form produces  0 <= f(x) <= 1
    '''
    max_exp = 709
    x1 = np.clip(a1 - b1*x, -max_exp, max_exp)
    x2 = np.clip(a2 - b2*x, -max_exp, max_exp)
    return (1.0 / (1.0 + np.exp(x1))) * (1.0 / (1.0 + np.exp(x2)))


def gompertz(x, a, b, c):
    '''
    Compute the Gompertz function for a given set of parameters.
    This function is used for describing mortality and ageing-like processes.

    See:
    https://en.wikipedia.org/wiki/Gompertz_function

    The Gompertz function is defined as:
    f(x) = a * exp(-b * exp(-c * x))

    Parameters:
    x (array-like): The array of x values at which the function is evaluated
    a (float): The asymptote of the function as x approaches infinity.
    b (float): Displacement along the x-axis
    c (float): The growth rate

    Returns:
    y : (ndarray)
        An array of y values corresponding to the gomeprtz function
        evaluated at the input x values.
    """
    '''
    return a*np.exp(-b*np.exp(-c*x))


def gompertz_dfun(x, a, b, c):
    '''
    Compute the derivative of the Gompertz function with respect to x for a given set of parameters.

    The derivative of the Gompertz function is defined as:
    f'(x) = a * b * c * exp(-(b / exp(c * x)) - c * x)

    Parameters:
    x (array-like): The array of x values at which the function is evaluated
    a (float): The asymptote of the Gompertz function as x approaches infinity.
    b (float): Displacement along the x-axis
    c (float): The growth rate

    Returns:
    ndarray: An array of derivative values corresponding to the input x values.
    '''
    return a*b*c*np.exp(-(b/np.exp(c*x)) - c*x)


#% Exceptions

class DuplicateNameException(Exception):
    """
    Raised when either multiple instances of Module or State, or of any other type
    passed to ndict have duplicate names."""


    def __init__(self, obj):
        msg = f"A {type(obj)} with name `{obj.name}` has already been added."
        super().__init__(msg)
        return