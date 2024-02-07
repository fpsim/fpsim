"""
Define defaults for use throughout FPsim
"""

import numpy as np
import sciris as sc

#%% Global defaults
useSI          = True
mpy            = 12   # Months per year, to avoid magic numbers
eps            = 1e-9 # To avoid divide-by-zero
max_age        = 99   # Maximum age
max_age_preg   = 50   # Maximum age to become pregnant
max_parity     = 20   # Maximum number of children

#%% Defaults when creating a new person
person_defaults = dict(
    # Basic demographics
    uid=-1,
    age=0,
    age_by_group=0,
    sex=0,
    alive=True,

    # Contraception
    method=0,
    barrier=0,

    # Sexual and reproductive history
    parity=0,
    pregnant=False,
    fertile=False,
    sexually_active=False,
    sexual_debut=False,
    sexual_debut_age=-1,
    fated_debut=-1,
    first_birth_age=-1,
    lactating=False,
    gestation=0,
    preg_dur=0,
    stillbirth=0,
    miscarriage=0,
    abortion=0,
    pregnancies=0,
    months_inactive=0,
    postpartum=False,
    mothers=-1,
    short_interval=0,
    secondary_birth=0,
    postpartum_dur=0,
    lam=False,
    breastfeed_dur=0,
    breastfeed_dur_total=0,

    # Indices of children -- list of lists
    children=[],

    # Dates
    dobs=[],  # Dates of birth of children
    still_dates=[],  # Dates of stillbirths -- list of lists
    miscarriage_dates=[],  # Dates of miscarriages -- list of lists
    abortion_dates=[],  # Dates of abortions -- list of lists
    short_interval_dates=[],  # age of agents at short birth interval -- list of lists

    # Fecundity
    remainder_months=0,
    personal_fecundity=0,

    # Empowerment - attributes will remain at these values if use_empowerment is False
    paid_employment=False,
    partnered=False,
    partnership_age=-1,
    urban=True,
    decision_wages=0,
    decision_health=0,
    sexual_autonomy=0,

    # Education - will remain at these values if use_empowerment is False
    edu_objective=0,
    edu_attainment=0,
    edu_dropout=False,
    edu_interrupted=False,
    edu_completed=False,
    edu_started=False,
)

# Postpartum keys to months
postpartum_map = {
    'pp0to5':   [0, 6],
    'pp6to11':  [6, 12],
    'pp12to23': [12, 24]
}

## Age bins for tracking age-specific fertility rate
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