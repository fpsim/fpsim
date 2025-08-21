'''
Set the parameters for FPsim, specifically for Ethiopia.
'''

from pathlib import Path
import numpy as np
from .. import ethiopia as eth
from fpsim import defaults as fpd
import fpsim.locations.data_utils as fpld


# %% Housekeeping

def scalar_pars():
    scalar_pars = eth.scalar_pars()
    # calibrated params
    scalar_pars['fecundity_var_low'] = 0.95
    scalar_pars['fecundity_var_high'] = 1.36
    scalar_pars['exposure_factor'] = 1.65
    return scalar_pars

def filenames():
    ''' Data files for use with calibration, etc -- not needed for running a sim '''
    base_dir = Path(__file__).resolve().parent / 'data'
    files = eth.filenames()
    files['base'] = base_dir
    files['mcpr'] = base_dir / 'cpr.csv'
    files['tfr'] = base_dir / 'tfr.csv' ## From DHS 2016
    files['asfr'] = base_dir / 'asfr.csv' ## From DHS 2016
    files['methods'] = base_dir / 'mix.csv' ## From DHS 2016
    files['use'] = base_dir / 'use.csv'  ## From PMA 2019
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

# %% Make and validate parameters

def make_pars(location='tigray', seed=None):
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
    pars['spacing_pref'] = fpld.birth_spacing_pref(location)

    # Contraceptive methods
    pars['mcpr'] = fpld.mcpr(location)

    #TODO: The latest version of FPsim uses age_partnership and wealth_quintile which these region files don't incorporate;
    # could possibly retrieve this data regionally?

    # Demographics: partnership and wealth status
    pars['age_partnership'] = fpld.age_partnership('ethiopia')
    pars['wealth_quintile'] = fpld.wealth('ethiopia')

    return pars
