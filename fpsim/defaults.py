"""
Define defaults for use throughout FPsim
"""

import numpy as np
import sciris as sc
import starsim as ss


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
    valid_country_locs = ['senegal', 'kenya', 'ethiopia']
    if location not in valid_country_locs:
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
person_defaults = [
    # Basic demographics
    State('uid',                -1, int),
    State('age',                0, float),
    State('age_by_group',       0, float),
    State('sex',                0, bool),
    State('alive',              1, bool),

    # Contraception
    State('on_contra',          0, bool),  # whether she's on contraception
    State('method',             0, int),  # Which method to use. 0 used for those on no method
    State('ti_contra',          0, int),  # time point at which to set method
    State('barrier',            0, int),
    State('ever_used_contra',   0, bool),  # Ever been on contraception. 0 for never having used

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

    # Fecundity
    State('remainder_months',   0, int),
    State('personal_fecundity', 0, int),

    # Empowerment - states will remain at these values if use_empowerment is False
    # NOTE: to use empowerment metrics, please refer to the kenya_empowerment repo
    # These states will be refactored into a separate module as part of the V3 release.
    State('paid_employment',    0, bool),
    State('decision_wages',     0, bool),
    State('decision_health',    0, bool),
    State('decision_purchase',  0, bool),
    State('buy_decision_major', 0, bool),       # whether she has decision making ability over major purchases
    State('buy_decision_daily', 0, bool),       # whether she has decision making over daily household purchases
    State('buy_decision_clothes', 0, bool),     # whether she has decision making over clothing purchases
    State('decide_spending_partner', 0, bool),  # whether she has decision makking over her partner's wages
    State('has_savings', 0, bool),              # whether she has savings
    State('has_fin_knowl', 0, bool),            # whether she knows where to get financial info
    State('has_fin_goals', 0, bool),            # whether she has financial goals
    State('sexual_autonomy',    0, bool),       # whether she has ability to refuse sex

    # Composite empowerment attributes
    State('financial_autonomy',    0, float),
    State('decision_making', 0, float),

    # Empowerment - fertility intent
    State('fertility_intent', 0, bool),
    State('categorical_intent', "cannot", "<U6"),
    State('intent_to_use', 0, bool),            # for women not on contraception, whether she has intent to use contraception

    # Partnership information -- states will remain at these values if use_partnership is False
    State('partnered',    0, bool),
    State('partnership_age', -1, float),

    # Socioeconomic
    State('urban', 1, bool),
    State('wealthquintile', 3, int),       # her current wealth quintile, an indicator of the economic status of her household, 1: poorest quintile; 5: wealthiest quintile

    # Education - states will remain at these values if use_education is False
    State('edu_objective',      0, float),
    State('edu_attainment',     0, float),
    State('edu_dropout',        0, bool),
    State('edu_interrupted',    0, bool),
    State('edu_completed',      0, bool),
    State('edu_started',        0, bool),

    State('child_inds',         -1,     int,    ncols=max_parity),
    State('birth_ages',         np.nan, float,  ncols=max_parity),  # Ages at time of live births
    State('stillborn_ages',     np.nan, float,  ncols=max_parity),  # Ages at time of stillbirths
    State('miscarriage_ages',   np.nan, float,  ncols=max_parity),  # Ages at time of miscarriages
    State('abortion_ages',      np.nan, float,  ncols=max_parity),  #  Ages at time of abortions
    # State('short_interval_ages', np.nan, float, ncols=max_parity)  # Ages of agents at short birth interval
]

person_defaults = ss.ndict(person_defaults)

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
    'birthday_fraction',
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
    'method_usage',
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
