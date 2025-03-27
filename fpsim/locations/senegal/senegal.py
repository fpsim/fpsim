'''
Set the parameters for FPsim, specifically for Senegal.
'''

import numpy as np
import sciris as sc
from scipy import interpolate as si
from ... import defaults as fpd
import fpsim.locations.data_utils as fpld


# %% Parameters

def scalar_pars():
    scalar_pars = {
        'location': 'senegal',
        'breastfeeding_dur_mu': 19.66828,
        # Location parameter of gumbel distribution. Requires children's recode DHS file, see data_processing/breastfeedin_stats.R
        'breastfeeding_dur_beta': 7.2585,
        # Scale parameter of gumbel distribution. Requires children's recode DHS file, see data_processing/breastfeedin_stats.R
        'abortion_prob': 0.08,  # From https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4712915/
        'twins_prob': 0.015,  # From https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0025239
    }
    return scalar_pars


def filenames():
    ''' Data files for use with calibration, etc -- not needed for running a sim '''
    files = {}
    files['base'] = sc.thisdir(aspath=True) / 'data'
    files['basic_dhs']        = 'basic_dhs.yaml'
    files['popsize']          = 'popsize.csv'
    files['mcpr']             = 'cpr.csv'
    files['tfr']              = 'tfr.csv'
    files['asfr']             = 'asfr.csv'
    files['ageparity']        = 'ageparity.csv'
    files['spacing']          = 'birth_spacing_dhs.csv'
    files['methods']          = 'mix.csv'
    files['afb'] = 'afb.table.csv'
    files['use'] = 'use.csv'
    return files


# %% Fecundity

def female_age_fecundity():
    '''
    Use fecundity rates from PRESTO study: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5712257/
    Fecundity rate assumed to be approximately linear from onset of fecundity around age 10 (average age of menses 12.5) to first data point at age 20
    45-50 age bin estimated at 0.10 of fecundity of 25-27 yr olds, based on fertility rates from Senegal
    '''
    fecundity = {
        'bins': np.array([0., 5, 10, 15, 20, 25, 28, 31, 34, 37, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99]),
        'f': np.array([0., 0, 0, 65, 70.8, 79.3, 77.9, 76.6, 74.8, 67.4, 55.5, 7.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])}
    fecundity[
        'f'] /= 100  # Conceptions per hundred to conceptions per woman over 12 menstrual cycles of trying to conceive

    fecundity_interp_model = si.interp1d(x=fecundity['bins'], y=fecundity['f'])
    fecundity_interp = fecundity_interp_model(fpd.spline_preg_ages)
    fecundity_interp = np.minimum(1, np.maximum(0, fecundity_interp))  # Normalize to avoid negative or >1 values

    return fecundity_interp


def fecundity_ratio_nullip():
    '''
    Returns an array of fecundity ratios for a nulliparous woman vs a gravid woman
    from PRESTO study: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5712257/
    Approximates primary infertility and its increasing likelihood if a woman has never conceived by age
    '''
    fecundity_ratio_nullip = np.array([[0, 5, 10, 12.5, 15, 18, 20, 25, 30, 34, 37, 40, 45, 50],
                                       [1, 1, 1, 1, 1, 1, 1, 0.96, 0.95, 0.71, 0.73, 0.42, 0.42, 0.42]])
    fecundity_nullip_interp = fpld.data2interp(fecundity_ratio_nullip, fpd.spline_preg_ages)

    return fecundity_nullip_interp


# %% Pregnancy exposure

def exposure_age():
    '''
    Returns an array of experimental factors to be applied to account for
    residual exposure to either pregnancy or live birth by age.  Exposure to pregnancy will
    increase factor number and residual likelihood of avoiding live birth (mostly abortion,
    also miscarriage), will decrease factor number
    '''
    exposure_correction_age = np.array([[0, 5, 10, 12.5, 15, 18, 20, 25, 30, 35, 40, 45, 50],
                                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
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
    ''' Reasons for nonuse -- taken from DHS '''

    barriers = sc.odict({
        'No need': 54.2,
        'Opposition': 30.5,
        'Knowledge': 1.7,
        'Access': 4.5,
        'Health': 12.9,
    })

    barriers[:] /= barriers[:].sum()  # Ensure it adds to 1
    return barriers


# %% Make and validate parameters

<<<<<<< HEAD
def make_pars(location='senegal', seed=None, use_subnational=None):
=======
def make_pars(use_empowerment=None, use_education=None, use_partnership=None, seed=None):
>>>>>>> main
    """
    Take all parameters and construct into a dictionary
    """

    # Scalar parameters and filenames
    pars = scalar_pars()
    pars['filenames'] = filenames()

    # Demographics and pregnancy outcome
    pars['age_pyramid'] = fpld.age_pyramid(location)
    pars['age_mortality'] = fpld.age_mortality(location, data_year=1990)
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

    # Handle modules that have not been implemented yet
    kwargs = locals()
<<<<<<< HEAD
    not_implemented_args = ['use_subnational']
=======
    not_implemented_args = ['use_empowerment', 'use_education', 'use_partnership']
>>>>>>> main
    true_args = [key for key in not_implemented_args if kwargs[key] is True]
    if true_args:
        errmsg = f"{true_args} not implemented yet for {pars['location']}"
        raise NotImplementedError(errmsg)

    return pars
