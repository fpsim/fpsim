"""
Define defaults for use throughout FPsim
"""

import numpy as np
import sciris as sc
import starsim as ss
import fpsim.settings as fps
import fpsim.arrays as fpa

from . import base as fpb

#%% Global defaults
useSI          = True
mpy            = 12   # Months per year, to avoid magic numbers
eps            = 1e-9 # To avoid divide-by-zero
min_age        = 15   # Minimum age to be considered eligible to use contraceptive methods
max_age        = 99   # Maximum age (inclusive)
max_age_preg   = 50   # Maximum age to become pregnant
max_parity     = 20   # Maximum number of children to track - also applies to abortions, miscarriages, stillbirths
max_parity_spline = 20   # Used for parity splines


#%% Defaults when creating a new person
class State:
    def __init__(self, name, val=None, dtype=None, ncols=None):
        """
        Initialize a state
        Args:
            name (str): name of state
            val (list, array, float, or str): value(s) to populate array with
            dtype (dtype): datatype. Inferred from val if not provided.
            ncols (int): number of cols, needed for 2d states like birth_ages (n_agents * n_births)
        """
        self.name = name
        self.val = val
        self.dtype = dtype
        self.ncols = ncols

    @property
    def ndim(self):
        return 1 if self.ncols is None else 2

    def new(self, n, vals=None):
        """
        Define an empty array with the correct value and data type
        """
        if vals is None: vals = self.val  # Use default if none provided

        if isinstance(vals, np.ndarray):
            assert len(vals) == n
            arr = vals
        else:
            if self.dtype is None: dtype = object if isinstance(vals, str) else None
            else: dtype = self.dtype
            shape = n if self.ncols is None else (n, self.ncols)
            arr = np.full(shape=shape, fill_value=vals, dtype=dtype)
        return arr


# Parse locations
def get_location(location, printmsg=False):
    default_location = 'senegal'  # Need to change this back to Senegal once parameters have been added
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

    # Define valid locations
    valid_country_locs = ['senegal', 'kenya', 'ethiopia']
    if location not in valid_country_locs:
        errormsg = f'Location "{location}" is not currently supported'
        raise NotImplementedError(errormsg)

    return location


# Defaults states and values of any new(born) agent unless initialized with data or other strategy
# or updated during the course of a simulation.
person_defaults = [
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
    ss.FloatArr('secondary_birth', default=0),
    ss.FloatArr('postpartum_dur', default=0),
    ss.State('lam', default=False),
    ss.FloatArr('breastfeed_dur', default=0),
    ss.FloatArr('breastfeed_dur_total', default=0),

    # Fecundity
    ss.FloatArr('remainder_months', default=0),
    ss.FloatArr('personal_fecundity', default=0),

    # Empowerment - states will remain at these values if use_empowerment is False
    ss.State('paid_employment', default=False),
    ss.State('decision_wages', default=False),
    ss.State('decision_health', default=False),
    ss.State('decision_purchase', default=False),
    ss.State('buy_decision_major', default=False),  # whether she has decision making ability over major purchases
    ss.State('buy_decision_daily', default=False),  # whether she has decision making over daily household purchases
    ss.State('buy_decision_clothes', default=False),  # whether she has decision making over clothing purchases
    ss.State('decide_spending_partner', default=False),  # whether she has decision makking over her partner's wages
    ss.State('has_savings', default=False),  # whether she has savings
    ss.State('has_fin_knowl', default=False),  # whether she knows where to get financial info
    ss.State('has_fin_goals', default=False),  # whether she has financial goals
    ss.State('sexual_autonomy', default=False),  # whether she has ability to refuse sex

    # Composite empowerment attributes
    ss.FloatArr('financial_autonomy', default=0),
    ss.FloatArr('decision_making', default=0),

    # Empowerment - fertility intent
    ss.State('fertility_intent', default=False),
    ss.Arr('categorical_intent', dtype="<U6",
           default="no"),  # default listed as "cannot", but its overridden with "no" during init
    ss.State('intent_to_use', default=False),

    # Partnership information -- states will remain at these values if use_partnership is False
    ss.State('partnered', default=False),
    ss.FloatArr('partnership_age', default=-1),

    # Urban (basic demographics) -- state will remain at these values if use_urban is False
    ss.State('urban', default=True),
    ss.Arr('region', dtype="<U64", default=None),
    ss.FloatArr('wealthquintile', default=3), # her current wealth quintile, an indicator of the economic status of her household, 1: poorest quintile; 5: wealthiest quintile

    # Education - states will remain at these values if use_education is False
    ss.FloatArr('edu_objective', default=0),
    ss.FloatArr('edu_attainment', default=0),
    ss.FloatArr('edu_dropout', default=0),
    ss.FloatArr('edu_interrupted', default=0),
    ss.FloatArr('edu_completed', default=0),
    ss.FloatArr('edu_started', default=0),

    # Add these states to the people object. They are not tracked by timestep in the way other states are, so they
    # need to be added manually. Eventually these will become part of a separate module tracking pregnancies and
    # pregnancy outcomes.
    #self.child_inds = np.full(max_parity, -1, int),
    fpa.MultiFloat('birth_ages', default=np.full(max_parity, np.nan, float)),  # Ages at time of live births
    fpa.MultiFloat('stillborn_ages', default=np.full(max_parity, np.nan, float)),  # Ages at time of stillbirths
    fpa.MultiFloat('miscarriage_ages', default=np.full(max_parity, np.nan, float)),  # Ages at time of miscarriages
    fpa.MultiFloat('abortion_ages', default=np.full(max_parity, np.nan, float)),  # Ages at time of abortions
    # State('short_interval_ages', np.nan, float, ncols=max_parity)  # Ages of agents at short birth interval



]

# person_defaults = ss.ndict(person_defaults)

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
    'contra_access',
    'new_users',
    'mcpr',
    'cpr',
    'acpr',
    'ever_used_contra',
    'switchers',
    'urban_women',
    'pp0to5',
    'pp6to11',
    'pp12to23',
    'parity0to1',
    'parity2to3',
    'parity4to5',
    'parity6plus',
    'wq1',
    'wq2',
    'wq3',
    'wq4',
    'wq5',
    'nonpostpartum',
    'total_women_fecund',
    'method_failures',
    # 'birthday_fraction', in both list results and here, not sure where the correct spot is
    'short_intervals',
    'secondary_births',
    'proportion_short_interval',
    # Education
    'edu_objective',
    'edu_attainment',
    # Empowerment and intent: all zero unless using an empowerment module
    'perc_contra_intent',
    'perc_fertil_intent',
    'paid_employment',
    'decision_wages',
    'decide_spending_partner',
    "buy_decision_major", 
    "buy_decision_daily", 
    "buy_decision_clothes", 
    "decision_health",
    "has_savings", 
    "has_fin_knowl", 
    "has_fin_goals",
    "financial_autonomy", 
    "decision_making",
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
    'contra_access_over_year',
    'new_users_over_year',
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
    # 'method_usage',
)

# Map between key names in results and annualised results, some of them are different
to_annualize = {
    'method_failures' :'method_failures',
    'infant_deaths'   :'infant_deaths',
    'total_births'    :'total_births',
    'births'          :'live_births',  # X
    'stillbirths'     :'stillbirths',
    'miscarriages'    :'miscarriages',
    'abortions'       :'abortions',
    'short_intervals' : 'short_intervals',
    'secondary_births': 'secondary_births',
    'maternal_deaths' : 'maternal_deaths',
    'pregnancies'     : 'pregnancies',
    'contra_access'   : 'contra_access',
    'new_users'       : 'new_users'}

# People's states for which we will need circular buffers
longitude_keys = [
    'on_contra',
    'intent_to_use',
    'buy_decision_major',
    'buy_decision_clothes',
    'buy_decision_daily',
    'has_fin_knowl',
    'has_fin_goals',
    'financial_autonomy',
    'has_fin_goals',
    'paid_employment',
    'has_savings',
    'decision_wages',
    'decide_spending_partner',
    'decision_health'
]
