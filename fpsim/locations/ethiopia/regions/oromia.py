'''
Set the parameters for FPsim, specifically for Ethiopia.
'''

import os
import numpy as np
import pandas as pd
import sciris as sc
from .. import ethiopia as eth
from fpsim import defaults as fpd
import fpsim.locations.data_utils as fpld


# %% Housekeeping

def this_dir():
    thisdir = sc.path(sc.thisdir(__file__))  # For loading CSV files
    return thisdir

def scalar_pars():
    scalar_pars = eth.scalar_pars()
    scalar_pars['location'] = 'oromia'
    # calibrated params
    scalar_pars['fecundity_var_low'] = 0.958
    scalar_pars['fecundity_var_high'] = 1.024
    scalar_pars['exposure_factor'] = 1.308  # Overall exposure correction factor
    return scalar_pars

def filenames():
    ''' Data files for use with calibration, etc -- not needed for running a sim '''
    base_dir = os.path.dirname(os.path.abspath(__file__))
    files = eth.filenames()
    files['base'] = os.path.join(base_dir, 'data')
    files['mcpr'] = 'cpr.csv'
    files['tfr'] = 'tfr.csv' ## From DHS 2016
    files['asfr'] = 'asfr.csv' ## From DHS 2016
    files['methods'] = 'mix.csv' ## From DHS 20
    files['use'] = 'use.csv'  ## From PMA 2019
    return files

def exposure_age():
    '''
    Returns an array of experimental factors to be applied to account for
    residual exposure to either pregnancy or live birth by age.  Exposure to pregnancy will
    increase factor number and residual likelihood of avoiding live birth (mostly abortion,
    also miscarriage), will decrease factor number
    '''
    exposure_correction_age = np.array([[0, 5, 10, 12.5, 15, 18, 20, 25, 30, 35, 40, 45, 50],
                                        [1, 1, 1, 1, 1, 1, 1, 0.8, 0.8, 0.8, 1, 1, 1]])
    exposure_age_interp = fpld.data2interp(exposure_correction_age, fpd.spline_preg_ages)

    return exposure_age_interp

def exposure_parity():
    '''
    Returns an array of experimental factors to be applied to account for residual exposure to either pregnancy
    or live birth by parity.
    '''
    exposure_correction_parity = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                                           [1, 1, 1, 1, 1, 1, 1, 0.8, 0.5, 0.3, 0.15, 0.10, 0.05, 0.01]])
    exposure_parity_interp = fpld.data2interp(exposure_correction_parity, fpd.spline_parities)

    return exposure_parity_interp

def birth_spacing_pref():
    '''
    Returns an array of birth spacing preferences by closest postpartum month.
    Applied to postpartum pregnancy likelihoods.

    NOTE: spacing bins must be uniform!
    '''
    postpartum_spacing = np.array([
        [0, 1],
        [3, 1],
        [6, 1],
        [9, 1],
        [12, 1],
        [15, 1],
        [18, 1],
        [21, 1],
        [24, 1],
        [27, 1],
        [30, 1],
        [33, 1],
        [36, 1],
    ])

    # Calculate the intervals and check they're all the same
    intervals = np.diff(postpartum_spacing[:, 0])
    interval = intervals[0]
    assert np.all(
        intervals == interval), f'In order to be computed in an array, birth spacing preference bins must be equal width, not {intervals}'
    pref_spacing = {}
    pref_spacing['interval'] = interval  # Store the interval (which we've just checked is always the same)
    pref_spacing['n_bins'] = len(intervals)  # Actually n_bins - 1, but we're counting 0 so it's OK
    pref_spacing['months'] = postpartum_spacing[:, 0]
    pref_spacing['preference'] = postpartum_spacing[:, 1]  # Store the actual birth spacing data

    return pref_spacing

def region_proportions(location):
    '''
    Defines the proportion of the population in the region to establish the probability of living there.
    '''
    region_data = pd.read_csv(this_dir() / 'data' / 'region.csv')
    region_dict = {}
    region_dict['mean'] = region_data.loc[region_data['region'] == location]['mean']
    region_dict['urban'] = region_data.loc[region_data['region'] == location]['urban']

    return region_dict

# %% Make and validate parameters

def make_pars(location='oromia', seed=None):
    '''
    Take all parameters and construct into a dictionary
    '''

    # Scalar parameters and filenames
    pars = scalar_pars()
    pars['abortion_prob'], pars['twins_prob'] = fpld.scalar_probs('ethiopia')
    pars.update(fpld.bf_stats(location))
    pars['filenames'] = filenames()

    # Demographics and pregnancy outcome
    pars['age_pyramid'] = fpld.age_pyramid(location) # Addis Ababa 1994
    pars['age_mortality'] = fpld.age_mortality('ethiopia', data_year=2020)
    pars['urban_prop'] = fpld.urban_proportion('ethiopia')
    pars['maternal_mortality'] = fpld.maternal_mortality('ethiopia')
    pars['infant_mortality'] = fpld.infant_mortality('ethiopia')
    pars['miscarriage_rates'] = fpld.miscarriage()
    pars['stillbirth_rate'] = fpld.stillbirth('ethiopia')

    # Fecundity
    pars['age_fecundity'] = fpld.female_age_fecundity()
    pars['fecundity_ratio_nullip'] = fpld.fecundity_ratio_nullip()
    pars['lactational_amenorrhea'] = fpld.lactational_amenorrhea(location) # From DHS 2016

    # Pregnancy exposure
    pars['sexual_activity'] = fpld.sexual_activity(location) # From DHS 2016
    pars['sexual_activity_pp'] = fpld.sexual_activity_pp(location) # From DHS 2016
    pars['debut_age'] = fpld.debut_age(location) # From DHS 2016
    pars['exposure_age'] = exposure_age()
    pars['exposure_parity'] = exposure_parity()
    pars['spacing_pref'] = birth_spacing_pref()

    # Contraceptive methods
    pars['mcpr'] = fpld.mcpr(location)

    # Regional parameters
    pars['region'] = region_proportions(location) # This function returns extrapolated and raw data

    #TODO: The latest version of FPsim uses age_partnership and wealth_quintile which these region files don't incorporate;
    # could possibly retrieve this data regionally?

    # Demographics: partnership and wealth status
    pars['age_partnership'] = fpld.age_partnership('ethiopia')
    pars['wealth_quintile'] = fpld.wealth('ethiopia')

    return pars