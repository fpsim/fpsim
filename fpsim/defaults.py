"""
Define defaults for use throughout FPsim
"""

import numpy as np
import sciris as sc

from . import base as fpb

#%% Global defaults
useSI          = True
mpy            = 12   # Months per year, to avoid magic numbers
eps            = 1e-9 # To avoid divide-by-zero
min_age        = 15   # Minimum age to be considered eligible to use contraceptive methods
max_age        = 99   # Maximum age
max_age_preg   = 50   # Maximum age to become pregnant
max_parity     = 20   # Maximum number of children


#%% Defaults when creating a new person
class State:
    def __init__(self, name, val=None, dtype=None):
        """
        Initialize a state
        Args:
            name (str): name of state
            val (list, array, float, or str): value(s) to populate array with
            dtype (dtype): datatype. Inferred from val if not provided.
        """
        self.name = name
        self.val = val
        self.dtype = dtype

    def new(self, n, vals=None):
        """
        Define an empty array with the correct value and data type
        """
        if vals is None: vals = self.val  # Use default if none provided

        if isinstance(vals, np.ndarray):
            assert len(vals) == n
            arr = vals
        elif isinstance(vals, list):
            arr = [[] for _ in range(n)]
        else:
            if self.dtype is None: dtype = object if isinstance(vals, str) else None
            else: dtype = self.dtype
            arr = np.full(shape=n, fill_value=vals, dtype=dtype)
        return arr

# Defaults states and values of any new(born) agent unless initialized with data or other strategy
# or updated during the course of a simulation.
person_defaults = [
    # Basic demographics
    State('uid',                -1, int),
    State('age',                0, float),
    State('age_by_group',       0, float),
    State('sex',                0, bool),
    State('alive',              1, bool),

    # Contraception
    State('method',             0, int),
    State('barrier',            0, int),

    # Sexual and reproductive history
    State('parity',             0, int),
    State('pregnant',           0, bool),
    State('fertile',            0, bool),
    State('sexually_active',    0, bool),
    State('sexual_debut',       0, bool),
    State('sexual_debut_age',   -1, float),
    State('fated_debut',        -1, float),
    State('first_birth_age',    -1, float),
    State('lactating',          0, bool),
    State('gestation',          0, int),
    State('preg_dur',           0, int),
    State('stillbirth',         0, int),
    State('miscarriage',        0, int),
    State('abortion',           0, int),
    State('pregnancies',        0, int),
    State('months_inactive',    0, int),
    State('postpartum',         0, bool),
    State('mothers',            -1, int),
    State('short_interval',     0, int),
    State('secondary_birth',    0, int),
    State('postpartum_dur',     0, int),
    State('lam',                0, bool),
    State('breastfeed_dur',     0, int),
    State('breastfeed_dur_total', 0, int),

    # Indices of children -- list of lists
    State('children',           [], list),

    # Dates
    State('dobs',               [], list),  # Dates of birth of children
    State('still_dates',        [], list),  # Dates of stillbirths -- list of lists
    State('miscarriage_dates',  [], list),  # Dates of miscarriages -- list of lists
    State('abortion_dates',     [], list),  # Dates of abortions -- list of lists
    State('short_interval_dates',[], list),  # age of agents at short birth interval -- list of lists

    # Fecundity
    State('remainder_months',   0, int),
    State('personal_fecundity', 0, int),

    # Empowerment - states will remain at these values if use_empowerment is False
    State('paid_employment',    0, bool),
    State('decision_wages',     0, float),
    State('decision_health',    0, float),
    State('sexual_autonomy',    0, float),

    # Partnership information -- states will remain at these values if use_partnership is False
    State('partnered',    0, bool),
    State('partnership_age', -1, float),

    # Urban (bsic demographics) -- state will remain at these values if use_urban is False
    State('urban', 1, bool),
    State('region', None, str),

    # Education - states will remain at these values if use_education is False
    State('edu_objective',      0, float),
    State('edu_attainment',     0, float),
    State('edu_dropout',        0, bool),
    State('edu_interrupted',    0, bool),
    State('edu_completed',      0, bool),
    State('edu_started',        0, bool)
]

person_defaults = fpb.ndict(person_defaults)

# Postpartum keys to months
postpartum_map = {
    'pp0to5':   [0, 6],
    'pp6to11':  [6, 12],
    'pp12to23': [12, 24]
}

# Age bins for tracking age-specific fertility rate
age_bin_map = {
    '10-14':   [10, 15],
    '15-19':   [15, 20],
    '20-24':   [20, 25],
    '25-29':   [25, 30],
    '30-34':   [30, 35],
    '35-39':   [35, 40],
    '40-44':   [40, 45],
    '45-49':   [45, 50]
}

# Age and parity splines
spline_ages      = np.arange(max_age + 1)
spline_preg_ages = np.arange(max_age_preg + 1)
spline_parities  = np.arange(max_parity + 1)

# Define allowable keys to select all (all ages, all methods, etc)
none_all_keys = [None, 'all', ':', [None], ['all'], [':']]

# Definition of contraceptive methods and corresponding numbers -- can be overwritten by locations
method_map = {
    'None'              : 0,
    'Pill'              : 1,
    'IUDs'              : 2,
    'Injectables'       : 3,
    'Condoms'           : 4,
    'BTL'               : 5,
    'Withdrawal'        : 6,
    'Implants'          : 7,
    'Other traditional' : 8,
    'Other modern'      : 9,
}

# Age bins for different method switching matrices -- can be overwritten by locations
method_age_map = {
    '<18':   [ 0, 18],
    '18-20': [18, 20],
    '21-25': [20, 25],
    '26-35': [25, 35],
    '>35':   [35, max_age+1], # +1 since we're using < rather than <=
}

method_youth_age_map = {
    '<16': [10, 16],
    '16-17': [16, 18],
    '18-19': [18, 20],
    '20-22': [20, 23],
    '23-25': [23, 26],
    '>25': [26, max_age+1]
}

age_specific_channel_bins = method_youth_age_map

by_age_results = sc.autolist(
    'acpr',
    'cpr',
    'mcpr',
    'pregnancies',
    'births',
    'imr_numerator',
    'imr_denominator',
    'mmr_numerator',
    'mmr_denominator',
    'imr',
    'mmr',
    'as_stillbirths',
    'stillbirths',
)

array_results = sc.autolist(
    't',
    'pop_size_months',
    'pregnancies',
    'births',
    'deaths',
    'stillbirths',
    'miscarriages',
    'abortions',
    'total_births',
    'maternal_deaths',
    'infant_deaths',
    'cum_maternal_deaths',
    'cum_infant_deaths',
    'on_methods_mcpr',
    'no_methods_mcpr',
    'on_methods_cpr',
    'no_methods_cpr',
    'on_methods_acpr',
    'no_methods_acpr',
    'mcpr',
    'cpr',
    'acpr',
    'pp0to5',
    'pp6to11',
    'pp12to23',
    'nonpostpartum',
    'total_women_fecund',
    'unintended_pregs',
    'birthday_fraction',
    'short_intervals',
    'secondary_births',
    'proportion_short_interval'
)
for age_group in age_bin_map.keys():
    array_results += 'total_births_' + age_group
    array_results += 'total_women_' + age_group

list_results = sc.autolist(
    'tfr_years',
    'tfr_rates',
    'pop_size',
    'mcpr_by_year',
    'cpr_by_year',
    'method_failures_over_year',
    'infant_deaths_over_year',
    'total_births_over_year',
    'live_births_over_year',
    'stillbirths_over_year',
    'miscarriages_over_year',
    'abortions_over_year',
    'pregnancies_over_year',
    'short_intervals_over_year',
    'secondary_births_over_year',
    'risky_pregs_over_year',
    'maternal_deaths_over_year',
    'proportion_short_interval_by_year',
    'mmr',
    'imr',
    'birthday_fraction',
    'method_usage',
)
