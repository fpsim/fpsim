'''
Set the parameters for FPsim, specifically for Kenya.
'''

import numpy as np
import sciris as sc
from fpsim import defaults as fpd
import fpsim.locations.data_utils as fpld

# %% Housekeeping

def scalar_pars():
    scalar_pars = {
        'location':             'kenya',
        'postpartum_dur':       23,
        'breastfeeding_dur_mu': 11.4261936291137,   # Location parameter of gumbel distribution. Requires children's recode DHS file, see data_processing/breastfeedin_stats.R
        'breastfeeding_dur_beta': 7.5435309020483,  # Location parameter of gumbel distribution. Requires children's recode DHS file, see data_processing/breastfeedin_stats.R
        'abortion_prob':        0.201,              # From https://bmcpregnancychildbirth.biomedcentral.com/articles/10.1186/s12884-015-0621-1, % of all pregnancies calculated
        'twins_prob':           0.016,              # From https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0025239
    }
    return scalar_pars


# TODO- these need to be changed for Kenya calibration and commented with their data source
def filenames():
    ''' Data files for use with calibration, etc -- not needed for running a sim '''
    files = {}
    files['base'] = sc.thisdir(aspath=True) / 'data'
    files['basic_dhs'] = 'basic_dhs.yaml' # From World Bank https://data.worldbank.org/indicator/SH.STA.MMRT?locations=KE
    files['popsize'] = 'popsize.csv' # Downloaded from World Bank: https://data.worldbank.org/indicator/SP.POP.TOTL?locations=KE
    files['mcpr'] = 'cpr.csv'  # From UN Population Division Data Portal, married women 1970-1986, all women 1990-2030
    files['tfr'] = 'tfr.csv'   # From World Bank https://data.worldbank.org/indicator/SP.DYN.TFRT.IN?locations=KE
    files['asfr'] = 'asfr.csv' # From UN World Population Prospects 2022: https://population.un.org/wpp/Download/Standard/Fertility/
    files['ageparity'] = 'ageparity.csv' # Choose from either DHS 2014 or PMA 2022
    files['spacing'] = 'birth_spacing_dhs.csv'
    files['methods'] = 'mix.csv'
    files['afb'] = 'afb.table.csv'
    files['use'] = 'use.csv'
    # files['empowerment'] = 'empowerment.csv'
    files['education'] = 'edu_initialization.csv'
    return files


# %% Pregnancy exposure

def exposure_age():
    '''
    Returns an array of experimental factors to be applied to account for
    residual exposure to either pregnancy or live birth by age.  Exposure to pregnancy will
    increase factor number and residual likelihood of avoiding live birth (mostly abortion,
    also miscarriage), will decrease factor number
    '''
    # Previously set to all 1's
    exposure_correction_age = np.array([[0, 5, 10, 12.5, 15, 18, 20, 25, 30, 35, 40, 45, 50],
                                        [1, 1, 1,  1 ,   .4, 1.3, 1.5 ,.8, .8, .5, .3, .5, .5]])

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


# %% Contraceptive methods
def barriers():
    ''' Reasons for nonuse -- taken from Kenya DHS 2014. '''

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

def make_pars(location='kenya', seed=None):
    """
    Take all parameters and construct into a dictionary
    """

    # Scalar parameters and filenames
    pars = scalar_pars()
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

    # Demographics: geography and partnership status
    pars['urban_prop'] = fpld.urban_proportion(location)
    pars['age_partnership'] = fpld.age_partnership(location)
    pars['wealth_quintile'] = fpld.wealth(location)

    return pars