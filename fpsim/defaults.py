"""
Define defaults for use throughout FPsim
"""

import numpy as np
import sciris as sc
import starsim as ss
import fpsim.arrays as fpa

#%% Global defaults
useSI          = True
mpy            = 12   # Months per year, to avoid magic numbers
eps            = 1e-9 # To avoid divide-by-zero
min_age        = 15   # Minimum age to be considered eligible to use contraceptive methods
max_age        = 99   # Maximum age (inclusive)
max_age_preg   = 50   # Maximum age to become pregnant
max_parity     = 20   # Maximum number of children to track - also applies to abortions, miscarriages, stillbirths
max_parity_spline = 20   # Used for parity splines
location_registry = {}  # Registry for external custom locations
valid_country_locs = ['senegal', 'kenya', 'ethiopia']
valid_region_locs = {
    'ethiopia': ['addis_ababa', 'afar', 'amhara', 'benishangul_gumuz', 'dire_dawa', 'gambela', 'harari', 'oromia', 'snnpr', 'somali', 'tigray']
}


# Parse locations
def get_location(location, printmsg=False):
    default_location = 'senegal'
    if not location:
        if printmsg: print('Location not supplied: using parameters from Senegal')
        location = default_location
    location = location.lower()  # Ensure it's lowercase
    if location == 'test':
        if printmsg: print('Running test simulation using parameters from Senegal')
        location = default_location
    if location == 'default':
        if printmsg: print('Running default simulation using parameters from Senegal')
        location = default_location

    # External locations override internal ones
    if location in location_registry:
        return location

    # Define valid locations
    if location not in valid_country_locs and not any(location in v for v in valid_region_locs.values()):
        errormsg = f'Location "{location}" is not currently supported'
        raise NotImplementedError(errormsg)

    return location


# Register custom location (for external users)
def register_location(name, location_ref):
    """
    Register a custom location, either a function (make_pars) or a module (with make_pars + data_utils).
    """
    if callable(location_ref):
        # wrap into a fake module-like object with just make_pars
        location_ref = type('LocationStub', (), {'make_pars': location_ref})()

    location_registry[name.lower()] = location_ref


# Defaults states and values of any new(born) agent unless initialized with data or other strategy
# or updated during the course of a simulation.
fpmod_states = [
    # Contraception
    ss.State('on_contra', default=False),  # whether she's on contraception
    ss.FloatArr('method', default=0),  # Which method to use. 0 used for those on no method
    ss.FloatArr('ti_contra', default=0),  # time point at which to set method
    ss.FloatArr('barrier', default=0),
    ss.State('ever_used_contra', default=False),  # Ever been on contraception. 0 for never having used

    # Sexual and reproductive history
    ss.FloatArr('parity', default=0),
    ss.State('pregnant', default=False),
    ss.State('fertile', default=False),
    ss.State('sexually_active', default=False),
    ss.State('sexual_debut', default=False),
    ss.FloatArr('sexual_debut_age', default=-1),
    ss.FloatArr('fated_debut', default=-1),
    ss.FloatArr('first_birth_age', default=-1),
    ss.State('lactating', default=False),
    ss.FloatArr('gestation', default=0),
    ss.FloatArr('preg_dur', default=0),
    ss.FloatArr('stillbirth', default=0),
    ss.FloatArr('miscarriage', default=0),
    ss.FloatArr('abortion', default=0),
    ss.FloatArr('pregnancies', default=0),
    ss.FloatArr('months_inactive', default=0),
    ss.State('postpartum', default=False),
    ss.FloatArr('mothers', default=-1),
    ss.FloatArr('short_interval', default=0),
    ss.FloatArr('secondary_birth', default=0), # no functions ever set this value. Used in empowerment?
    ss.FloatArr('postpartum_dur', default=0),
    ss.State('lam', default=False),
    ss.FloatArr('breastfeed_dur', default=0),
    ss.FloatArr('breastfeed_dur_total', default=0),

    # Fecundity
    ss.FloatArr('remainder_months', default=0),
    ss.FloatArr('personal_fecundity', default=0),

    # Add these states to the people object. They are not tracked by timestep in the way other states are, so they
    # need to be added manually. Eventually these will become part of a separate module tracking pregnancies and
    # pregnancy outcomes.
    fpa.TwoDimensionalArr('birth_ages', ncols=max_parity),  # Ages at time of live births
    fpa.TwoDimensionalArr('stillborn_ages', ncols=max_parity),  # Ages at time of stillbirths
    fpa.TwoDimensionalArr('miscarriage_ages', ncols=max_parity),  # Ages at time of miscarriages
    fpa.TwoDimensionalArr('abortion_ages', ncols=max_parity),  # Ages at time of abortions

    ss.State('partnered', default=False),  # Will remain at these values if use_partnership is False
    ss.FloatArr('partnership_age', default=-1),  # Will remain at these values if use_partnership is False
    ss.State('urban', default=True),  # Urban/rural
    ss.FloatArr('wealthquintile', default=3),  # Wealth quintile
]

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
spline_parities  = np.arange(max_parity_spline + 1)

# Define allowable keys to select all (all ages, all methods, etc)
none_all_keys = [None, 'all', ':', [None], ['all'], [':']]


# Age bins for different method switching matrices -- can be overwritten by locations
method_age_map = {
    '<18':   [ 0, 18],
    '18-20': [18, 20],
    '20-25': [20, 25],
    '25-35': [25, 35],
    '>35':   [35, max_age+1], # +1 since we're using < rather than <=
}

immutable_method_age_map = {
    '<18':   [ 0, 18],
    '18-20': [18, 20],
    '20-25': [20, 25],
    '25-30': [25, 30],
    '30-35': [30, 35],
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

# Counts - we compute number of new events each timestep, plus number of cumulative events
event_counts = sc.autolist(
    'births',
    'stillbirths',
    'miscarriages',
    'abortions',
    'short_intervals',
    'secondary_births',
    'pregnancies',
    'total_births',
    'maternal_deaths',
    'infant_deaths',
    'method_failures',
)

for age_group in age_bin_map.keys():
    event_counts += 'total_births_' + age_group
    event_counts += 'total_women_' + age_group

people_counts = sc.autolist(
    'contra_access',
    'new_users',
    'ever_used_contra',
    'switchers',
    'n_fecund',
    'pp0to5',
    'pp6to11',
    'pp12to23',
    'parity0to1',
    'parity2to3',
    'parity4to5',
    'parity6plus',
    'nonpostpartum',
)

sim_results = sc.autolist(
    'n_urban',
    'n_wq1',
    'n_wq2',
    'n_wq3',
    'n_wq4',
    'n_wq5',
)

# Rates and other results that aren't scaled
rate_results = sc.autolist(
    'tfr_years',
    'tfr_rates',
    'mmr',
    'imr',
    'p_short_interval',
)

dict_annual_results = sc.autolist(
    'method_usage',
)

# # Map between key names in results and annualised results, some of them are different
# # TODO, remove??
# to_annualize = {
#     'method_failures' :'method_failures',
#     'infant_deaths'   :'infant_deaths',
#     'total_births'    :'total_births',
#     'births'          :'live_births',  # X
#     'stillbirths'     :'stillbirths',
#     'miscarriages'    :'miscarriages',
#     'abortions'       :'abortions',
#     'short_intervals' : 'short_intervals',
#     'secondary_births': 'secondary_births',
#     'maternal_deaths' : 'maternal_deaths',
#     'pregnancies'     : 'pregnancies',
#     # 'contra_access'   : 'contra_access', # do these get moved?
#     # 'new_users'       : 'new_users'
#     }


