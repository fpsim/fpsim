'''
Define defaults for use throughout FPsim
'''

import numpy as np

#%% Global defaults
useSI        = True
mpy          = 12   # Months per year, to avoid magic numbers
resolution   = 1    # For spline interpolation, steps per year
eps          = 1e-9 # To avoid divide-by-zero
max_age      = 99   # Maximum age
max_age_preg = 50
if resolution != 1:
    raise NotImplementedError('Currently, resolutions other than 1 year for splines do not work')

#%% Defaults when creating a new person
person_defaults = dict(
    age = 0,
    sex = 0,
    parity = 0,
    method = None,
    barrier = 'None',
    postpartum_dur = 0,
    gestation = 0,
    remainder_months = 0,
    breastfeed_dur = 0,
    breastfeed_dur_total = 0
)

#%% Age bins for different method switching matrices
method_age_mapping = {
    '<18':   [ 0, 18],
    '18-20': [18, 20],
    '20-25': [20, 25],
    '>25':   [25, max_age+1], # +1 since we're using < rather than <=
}

spline_ages      = np.arange(resolution*max_age + 1) / resolution
spline_preg_ages = np.arange(resolution*max_age_preg + 1) / resolution