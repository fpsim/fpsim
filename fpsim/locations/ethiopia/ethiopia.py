'''
Set the parameters for FPsim, specifically for Ethiopia.
'''

import numpy as np
import pandas as pd
import sciris as sc
from scipy import interpolate as si
from fpsim import defaults as fpd
from . import regions
import fpsim.locations.data_utils as fpld


# %% Utilities
def this_dir():
    thisdir = sc.path(sc.thisdir(__file__))  # For loading CSV files
    return thisdir


# %% Parameters
def scalar_pars():
    scalar_pars = {
        'location':             'ethiopia',
        'postpartum_dur':       23,
        'breastfeeding_dur_mu': 9.30485863,     # Location parameter of gumbel distribution. Requires children's recode DHS file, see data_processing/breastfeedin_stats.R
        'breastfeeding_dur_beta': 8.20149079,   # Location parameter of gumbel distribution. Requires children's recode DHS file, see data_processing/breastfeedin_stats.R
        'abortion_prob':        0.176,          # From https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5568682/, % of all pregnancies calculated
        'twins_prob':           0.011,          # From https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0025239
    }
    return scalar_pars


def filenames():
    ''' Data files for use with calibration, etc -- not needed for running a sim '''
    files = {}
    files['base'] = sc.thisdir(aspath=True) / 'data'
    files['basic_dhs'] = 'basic_dhs.yaml' # From World Bank https://data.worldbank.org/indicator/SH.STA.MMRT?locations=ET
    files['popsize'] = 'popsize.csv' # From UN World Population Prospects 2022: https://population.un.org/wpp/Download/Standard/Population/
    files['mcpr'] = 'cpr.csv'  # From UN Population Division Data Portal, married women 1970-1986, all women 1990-2030
    files['tfr'] = 'tfr.csv'   # From World Bank https://data.worldbank.org/indicator/SP.DYN.TFRT.IN?locations=ET
    files['asfr'] = 'asfr.csv' # From UN World Population Prospects 2022: https://population.un.org/wpp/Download/Standard/Fertility/
    files['ageparity'] = 'ageparity.csv' # Choose from either DHS 2016 or PMA 2022
    files['spacing'] = 'birth_spacing_dhs.csv'
    files['methods'] = 'mix.csv'
    files['afb'] = 'afb.table.csv'
    files['use'] = 'use.csv'
    files['urban'] = 'urban.csv'
    #data files
    files['region'] = '../regions/data/region.csv' ## From DHS 2016
    files['asfr_region'] = '../regions/data/asfr_region.csv' ## From DHS 2016
    files['tfr_region'] = '../regions/data/tfr_region.csv' ## From DHS 2016
    files['methods_region'] = '../regions/data/mix_region.csv' ## From DHS 2016
    files['use_region'] = '../regions/data/use_region.csv'  ## From PMA 2019
    files['barriers_region'] = '../regions/data/barriers_region.csv' ## From PMA 2019
    files['lactational_amenorrhea_region'] = '../regions/data/lam_region.csv' ## From DHS 2016
    files['sexual_activity_region'] = '../regions/data/sexual_activity_region.csv' ## From DHS 2016
    files['sexual_activity_pp_region'] = '../regions/data/sexual_activity_pp_region.csv' ## From DHS 2016
    files['debut_age_region'] = '../regions/data/sexual_debut_region.csv' ## From DHS 2016
    return files


# %% Demographics and pregnancy outcome

def region_proportions():
    '''
    Defines the proportion of the population in each region to establish the probability of living in a given region.
    Uses 2016 Ethiopia DHS individual recode (v025) for region and V024 for urban to produce subnational estimates
    '''
    region_data = pd.read_csv(thisdir() / 'data' / 'region.csv')

    region_dict = {}
    region_dict['region'] = region_data['region'] # Return region names
    region_dict['mean'] = region_data['mean'] # Return proportion living in each region
    region_dict['urban'] = region_data['urban'] # Return proportion living in an urban area by region

    return region_dict


# %% Fecundity

def female_age_fecundity():
    '''
    Use fecundity rates from PRESTO study: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5712257/
    Fecundity rate assumed to be approximately linear from onset of fecundity around age 10 (average age of menses 12.5) to first data point at age 20
    45-50 age bin estimated at 0.10 of fecundity of 25-27 yr olds
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


def lactational_amenorrhea_region():
    '''
    Returns an array of the percent of breastfeeding women by month postpartum 0-11 months who meet criteria for LAM, stratified by region
    '''
    lam_region = pd.read_csv(thisdir() / 'data' / 'lam_region.csv')
    lam_dict = {}
    lam_dict['region'] = lam_region['region']  # Return region names
    lam_dict['month'] = lam_region['month']  # Return month postpartum
    lam_dict['rate'] = lam_region['rate']  # Return percent of breastfeeding women

    return lam_dict


# %% Pregnancy exposure

def sexual_activity_region():
    '''
    Returns a linear interpolation of rates of female sexual activity, stratified by region
    '''
    sexually_active_region_data = pd.read_csv(thisdir() / 'data' / 'sexual_activity_region.csv')
    sexually_active_region_dict = {}
    sexually_active_region_dict['region'] = sexually_active_region_data.iloc[:, 0]  # Return region names
    sexually_active_region_dict['age'] = sexually_active_region_data.iloc[:, 1]   # Return age
    sexually_active_region_dict['perc'] = sexually_active_region_data.iloc[:, 2] / 100  # Return perc divide by 100 to convert to a rate
    activity_ages_region = sexually_active_region_dict['age']
    activity_interp_model_region = si.interp1d(x=activity_ages_region, y=sexually_active_region_dict['perc'])
    activity_interp_region = activity_interp_model_region(fpd.spline_preg_ages)

    return activity_interp_region


def sexual_activity_pp_region():
    '''
     # Returns an additional array of monthly likelihood of having resumed sexual activity by region
    '''
    pp_activity_region = pd.read_csv(thisdir() / 'data' / 'sexual_activity_pp_region.csv')
    pp_activity_region_dict = {}
    pp_activity_region_dict['region'] = pp_activity_region['region'] # Return region names
    pp_activity_region_dict['month'] = pp_activity_region['month'] # Return month postpartum
    pp_activity_region_dict['perc'] = pp_activity_region['perc'] # Return likelihood of resumed sexual activity

    return pp_activity_region_dict


def debut_age_region():
    '''
 #   Returns an additional array of weighted probabilities of sexual debut by region
    '''
    sexual_debut_region_data = pd.read_csv(thisdir() / 'data' / 'sexual_debut_region.csv')
    debut_age_region_dict = {}
    debut_age_region_dict['region'] = sexual_debut_region_data['region'] # Return region names
    debut_age_region_dict['age'] = sexual_debut_region_data['age'] # Return month postpartum
    debut_age_region_dict['prob'] = sexual_debut_region_data['prob'] # Return weighted probabilities of sexual debut
    return debut_age_region_dict


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
    ''' Reasons for nonuse -- taken from Ethiopia PMA 2019. '''
    barriers = sc.odict({ #updated based on PMA cross-sectional data
        'No need': 58.5,
        'Opposition': 16.6,
        'Knowledge': 1.28,
        'Access': 2.73,
        'Health': 20.9,
    })

    barriers[:] /= barriers[:].sum()  # Ensure it adds to 1
    return barriers


def barriers_region():
    '''
    Returns reasons for nonuse by region
    '''
    reasons_region = pd.read_csv(thisdir() / 'regions' / 'data' / 'barriers_region.csv')
    reasons_region_dict = {}
    reasons_region_dict['region'] = reasons_region['region'] # Return region names
    reasons_region_dict['barrier'] = reasons_region['barrier'] # Return the reason for nonuse
    reasons_region_dict['perc'] = reasons_region['perc'] # Return retuned the percentage

    return reasons_region_dict


# %% Make and validate parameters

def make_pars(location='ethiopia', seed=None, use_subnational=None):
    '''
    Take all parameters and construct into a dictionary
    '''

    # Scalar parameters and filenames
    pars = scalar_pars()
    pars['filenames'] = filenames()

    # Demographics and pregnancy outcome
    pars['age_pyramid'] = fpld.age_pyramid(location)
    pars['age_mortality'] = fpld.age_mortality(location, data_year=2020)
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

    # Regional parameters
    if use_subnational:
        pars['region'] = region_proportions()  # This function returns extrapolated and raw data
        pars['lactational_amenorrhea_region'] = lactational_amenorrhea_region()
        pars['sexual_activity_region'] = sexual_activity_region()
        pars['sexual_activity_pp_region'] = sexual_activity_pp_region()
        pars['debut_age_region'] = debut_age_region()
        pars['barriers_region'] = barriers_region()

    kwargs = locals()
    not_implemented_args = []
    true_args = [key for key in not_implemented_args if kwargs[key] is True]
    if true_args:
        errmsg = f"{true_args} not implemented yet for {pars['location']}"
        raise NotImplementedError(errmsg)

    return pars
