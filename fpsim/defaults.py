'''
Define defaults for use throughout FPsim
'''

import numpy as np

#%% Global defaults
useSI        = True
mpy          = 12   # Months per year, to avoid magic numbers
eps          = 1e-9 # To avoid divide-by-zero
max_age      = 99   # Maximum age
max_age_preg = 49   # Maximum age to become pregnant
max_parity   = 20   # Maximum number of childre

#%% Defaults when creating a new person
person_defaults = dict(
    uid = -1,
    age = 0,
    sex = 0,
    parity = 0,
    method = 0,
    barrier = 0,
    postpartum_dur = 0,
    gestation = 0,
    remainder_months = 0,
    breastfeed_dur = 0,
    breastfeed_dur_total = 0,
    alive = True,
    pregnant = False,
    sexually_active = False,
    lactating = False,
    postpartum = False,
    lam = False,
)

#%% Age bins for different method switching matrices
method_age_mapping = {
    '<18':   [ 0, 18],
    '18-20': [18, 20],
    '20-25': [20, 25],
    '>25':   [25, max_age+1], # +1 since we're using < rather than <=
}

postpartum_mapping = {
    'pp0to5':   [ 0, 6],
    'pp6to11':  [6, 12],
    'pp12to23': [12, 24]
}

spline_ages      = np.arange(max_age + 2) # +2 so e.g. 0-100
spline_preg_ages = np.arange(max_age_preg + 2)
spline_parities  = np.arange(max_parity + 1)