'''
Define defaults for use throughout FPsim
'''

import numpy as np
import sciris as sc

__all__ = ['pars']


def pars(location=None, **kwargs):
    '''
    Function for getting default parameters.

    Args:
        location (str): the location to use for the parameters; use 'test' for a simple test set of parameters
        kwargs (dict): custom parameter values

    **Example**::
        pars = fp.pars(location='senegal')
    '''
    from . import locations as fplocs # Here to avoid circular import

    if not location:
        location = 'default'

    # Set test parameters
    if location == 'test':
        location = 'default'
        kwargs.setdefault('n', 100)
        kwargs.setdefault('verbose', 0)
        kwargs.setdefault('start_year', 2000)
        kwargs.setdefault('end_year', 2010)

    # Define valid locations
    if location in ['senegal', 'default']:
        pars = fplocs.senegal.make_pars()

    # Else, error
    else:
        errormsg = f'Location "{location}" is not currently supported'
        raise NotImplementedError(errormsg)

    # Merge with kwargs and copy
    mismatch = set(kwargs.keys()) - set(pars.keys())
    if len(mismatch):
        errormsg = f'The following key(s) are not valid: {sc.strjoin(mismatch)}'
        raise sc.KeyNotFoundError(errormsg)
    pars = sc.mergedicts(pars, kwargs, _copy=True)

    return pars


#%% Global defaults
useSI          = True
mpy            = 12   # Months per year, to avoid magic numbers
eps            = 1e-9 # To avoid divide-by-zero
max_age        = 99   # Maximum age
max_age_preg   = 50   # Maximum age to become pregnant
max_parity     = 20   # Maximum number of children

#%% Defaults when creating a new person
person_defaults = dict(
    uid                  = -1,
    age                  = 0,
    sex                  = 0,
    parity               = 0,
    method               = 0,
    barrier              = 0,
    postpartum_dur       = 0,
    gestation            = 0,
    preg_dur             = 0,
    stillbirth           = 0,
    miscarriage          = 0,
    abortion             = 0,
    remainder_months     = 0,
    breastfeed_dur       = 0,
    breastfeed_dur_total = 0,
    alive                = True,
    pregnant             = False,
    sexually_active      = False,
    sexual_debut         = False,
    sexual_debut_age     = -1,
    first_birth_age      = -1,
    lactating            = False,
    postpartum           = False,
    lam                  = False,
    mothers              = -1,
)

#%% Age bins for different method switching matrices
method_age_mapping = {
    '<18':   [ 0, 18],
    '18-20': [18, 20],
    '21-25': [20, 25],
    '>25':   [25, max_age+1], # +1 since we're using < rather than <=
}

postpartum_mapping = {
    'pp0to5':   [ 0, 6],
    'pp6to11':  [6, 12],
    'pp12to23': [12, 24]
}

## Age bins for tracking age-specific fertility rate

age_bin_mapping = {
    '10-14':   [10, 15],
    '15-19':   [15, 20],
    '20-24':   [20, 25],
    '25-29':   [25, 30],
    '30-34':   [30, 35],
    '35-39':   [35, 40],
    '40-44':   [40, 45],
    '45-49':   [45, 50]
}

spline_ages      = np.arange(max_age + 1)
spline_preg_ages = np.arange(max_age_preg + 1)
spline_parities  = np.arange(max_parity + 1)
debug_states = ["alive", "breastfeed_dur", "gestation", "lactating", "lam", "postpartum", "pregnant", "sexually_active", "postpartum_dur", \
                "parity", "method", "age", "first_birth_age", "dobs", "age", "sexual_debut", "sexual_debut_age"]
method_list = ["None", "Pill", "IUDs", "Injectable", "Condoms", "BTL", "Rhythm", "Withdrawal", "Implants", "Other"]