"""
This is a template configuration file for an FPsim model specific to a location.
Users should update values marked as USER-EDITABLE to match the context
they are modeling.
"""

import numpy as np
import sciris as sc
from fpsim import defaults as fpd
import fpsim.locations.data_utils as fpld

# %% Housekeeping

def scalar_pars():
    scalar_pars = {
        'location':             'test', # <<< USER-EDITABLE: Adjust name of location
        'postpartum_dur':       23,     # <<< USER-EDITABLE: Adjust/override any parameters that are defined in fpsim/defaults.py
    }
    return scalar_pars


def filenames():
    """ Data files for use with calibration, etc -- not needed for running a sim """
    files = {}
    files['base'] = sc.thisdir(aspath=True) / 'data' # Location-specific data directory
    files['basic_wb'] = 'basic_wb.yaml' # From World Bank https://data.worldbank.org/indicator/SH.STA.MMRT
    files['popsize'] = 'popsize.csv' # Downloaded from World Bank: https://data.worldbank.org/indicator/SP.POP.TOTL
    files['mcpr'] = 'cpr.csv'  # From UN Population Division Data Portal, married women 1970-1986, all women 1990-2030
    files['tfr'] = 'tfr.csv'   # From World Bank https://data.worldbank.org/indicator/SP.DYN.TFRT.IN
    files['asfr'] = 'asfr.csv' # From UN World Population Prospects 2022: https://population.un.org/wpp/Download/Standard/Fertility/
    files['ageparity'] = 'ageparity.csv' # Choose from either DHS 2014 or PMA 2022
    files['spacing'] = 'birth_spacing_dhs.csv' # From DHS
    files['methods'] = 'mix.csv' # From PMA
    files['afb'] = 'afb.table.csv' # From DHS
    files['use'] = 'use.csv' # From PMA
    files['education'] = 'edu_initialization.csv' # From DHS
    return files


# %% Pregnancy exposure

def exposure_age():
    """
    Returns an array of experimental factors to be applied to account for
    residual exposure to either pregnancy or live birth by age.  Exposure to pregnancy will
    increase factor number and residual likelihood of avoiding live birth (mostly abortion,
    also miscarriage), will decrease factor number
    """
    exposure_correction_age = np.array([[0, 5, 10, 12.5, 15, 18, 20, 25, 30, 35, 40, 45, 50],
                                        [1, 1, 1,  1 ,   1,  1,  1 , 1,  1,  1,   1,  1, 1]])  # <<< USER-EDITABLE: Can be modified for calibration

    exposure_age_interp = fpld.data2interp(exposure_correction_age, fpd.spline_preg_ages)
    return exposure_age_interp


def exposure_parity():
    """
    Returns an array of experimental factors to be applied to account for residual exposure to either pregnancy
    or live birth by parity.
    """
    exposure_correction_parity = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                                           [1, 1, 1, 1, 1, 1, 1, 0.8, 0.5, 0.3, 0.15, 0.10, 0.05, 0.01]])  # <<< USER-EDITABLE: Can be modified for calibration
    exposure_parity_interp = fpld.data2interp(exposure_correction_parity, fpd.spline_parities)

    return exposure_parity_interp


# %% Contraceptive methods
def barriers():
    """ Reasons for nonuse -- taken from DHS. """

    barriers = sc.odict({
        'No need': 40.3,
        'Opposition': 22.7,
        'Knowledge': 3.5,
        'Access': 13.4,
        'Health': 32.5,
    })

    barriers[:] /= barriers[:].sum()  # Ensure it adds to 1
    return barriers


# %% Make and validate parameters

def make_pars(location='test', seed=None):  # <<< USER-EDITABLE: Change name of location
    """
    Take all parameters and construct into a dictionary
    """

    # Scalar parameters and filenames
    pars = scalar_pars()
    pars['abortion_prob'], pars['twins_prob'] = fpld.scalar_probs(location)
    pars.update(fpld.bf_stats(location))
    pars['filenames'] = filenames()

    # Demographics and pregnancy outcome
    pars['age_pyramid'] = fpld.age_pyramid(location)
    pars['age_mortality'] = fpld.age_mortality(location, data_year=2010)
    pars['urban_prop'] = fpld.urban_proportion(location)
    pars['maternal_mortality'] = fpld.maternal_mortality(location)
    pars['infant_mortality'] = fpld.infant_mortality(location)
    pars['miscarriage_rates'] = fpld.miscarriage()
    pars['stillbirth_rate'] = fpld.stillbirth(location)

    # Fecundity
    pars['age_fecundity'] = fpld.female_age_fecundity()
    pars['fecundity_ratio_nullip'] = fpld.fecundity_ratio_nullip()
    pars['lactational_amenorrhea'] = fpld.lactational_amenorrhea(location)

    # Pregnancy exposure
    pars['sexual_activity'] = fpld.sexual_activity(location)
    pars['sexual_activity_pp'] = fpld.sexual_activity_pp(location)
    pars['debut_age'] = fpld.debut_age(location)
    pars['exposure_age'] = exposure_age()
    pars['exposure_parity'] = exposure_parity()
    pars['spacing_pref'] = fpld.birth_spacing_pref(location)

    # Contraceptive methods
    pars['barriers'] = barriers()
    pars['mcpr'] = fpld.mcpr(location)

    # Demographics: partnership and wealth status
    pars['age_partnership'] = fpld.age_partnership(location)
    pars['wealth_quintile'] = fpld.wealth(location)

    return pars