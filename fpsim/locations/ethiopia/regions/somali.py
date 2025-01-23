'''
Set the parameters for FPsim, specifically for Ethiopia.
'''

import numpy as np
import pandas as pd
import sciris as sc
from .. import ethiopia as eth
from scipy import interpolate as si
from fpsim import defaults as fpd

# %% Housekeeping

thisdir = sc.thispath(__file__)  # For loading CSV files

def scalar_pars():
    scalar_pars = eth.scalar_pars()
    scalar_pars['location'] = 'somali'
    # durations
    scalar_pars['breastfeeding_dur_mu'] = 6.26613513894357
    scalar_pars['breastfeeding_dur_beta'] = 5.53459861844008
    # basic parameters
    scalar_pars['end_year'] =  2016  # End year of simulation
    # fecunditity and exposure
    scalar_pars['fecundity_var_low'] = 2.5
    scalar_pars['fecundity_var_high'] = 3
    scalar_pars['exposure_factor'] = 0.6

    return scalar_pars

def data2interp(data, ages, normalize=False):
    ''' Convert unevenly spaced data into an even spline interpolation '''
    model = si.interp1d(data[0], data[1])
    interp = model(ages)
    if normalize:
        interp = np.minimum(1, np.maximum(0, interp))
    return interp

def filenames():
    ''' Data files for use with calibration, etc -- not needed for running a sim '''
    files = eth.filenames()

    files['asfr'] = '../regions/data/asfr_region.csv' ## From DHS 2016
    files['tfr'] = '../regions/data/tfr_region.csv' ## From DHS 2016
    files['methods'] = '../regions/data/mix_region.csv' ## From DHS 2016
    files['use'] = '../regions/data/use_region.csv'  ## From PMA 2019
    files['barriers'] = '../regions/data/barriers_region.csv' ## From PMA 2019
    files['lactational_amenorrhea'] = '../regions/data/lam_region.csv' ## From DHS 2016
    files['sexual_activity'] = '../regions/data/sexual_activity_region.csv' ## From DHS 2016
    files['sexual_activity_pp'] = '../regions/data/sexual_activity_pp_region.csv' ## From DHS 2016
    files['debut_age'] = '../regions/data/sexual_debut_region.csv' ## From DHS 2016
    files['mcpr'] = '../regions/data/cpr_region.csv'

    return files

# %% Demographics and pregnancy outcome
'''
Data from 1994 Census Report for Somali Region
https://www.statsethiopia.gov.et/wp-content/uploads/2019/06/Population-and-Housing-Census-1994-Somali-Region.pdf
'''
def age_pyramid():
    pyramid = np.array([[0, 236849, 209061], # Somali 1994 
                        [5, 354039, 286088],
                        [10, 365196, 267903],
                        [15, 248098, 173908],
                        [20, 146947, 114943],
                        [25, 86685, 101700],
                        [30, 80398, 101909],
                        [35, 57492, 78121],
                        [40, 80933, 78380],
                        [45, 41953, 36890],
                        [50, 53574, 40614],
                        [55, 19403, 11803],
                        [60, 37735, 18706],
                        [65, 9732, 4490],
                        [70, 15531, 7237],
                        [75, 2951, 1430],
                        [80, 7901, 4565]
                        ], dtype=float)    
    return pyramid

def urban_proportion(): # TODO: Flagging this - currently this is being used for the urban ratio for somali; I assume you'd want to use the region-specific value?
    urban_data = eth.urban_proportion()
    
    return urban_data  # Return this value as a float

def region_proportions():
    '''
    Defines the proportion of the population in the Somali region to establish the probability of living there.
    '''
    region_data = pd.read_csv(thisdir / 'data' / 'region.csv')
    region_dict = {}
    region_dict['mean'] = region_data.loc[region_data['region'] == 'Somali']['mean'] #HOLDER FOR NOW. NEED TO CALL ON LOCATION.
    region_dict['urban'] = region_data.loc[region_data['region'] == 'Somali']['urban']
    
    return region_dict

def age_mortality():
    mortality = eth.age_mortality()
    
    return mortality

def maternal_mortality():
    maternal_mortality = eth.maternal_mortality()
    
    return maternal_mortality

def infant_mortality():
    infant_mortality = eth.infant_mortality()
    
    return infant_mortality

def miscarriage():
    miscarriage_interp = eth.miscarriage()
    
    return miscarriage_interp

def stillbirth():
    stillbirth_rate = eth.stillbirth()
    
    return stillbirth_rate

# %% Fecundity

def female_age_fecundity():
    fecundity_interp = eth.female_age_fecundity()
    
    return fecundity_interp

def fecundity_ratio_nullip(): 
    fecundity_nullip_interp = eth.fecundity_ratio_nullip()
    
    return fecundity_nullip_interp

def lactational_amenorrhea_region():
    '''
    Returns a dictionary containing the percent of breastfeeding women by month postpartum 0-11 months who meet criteria for LAM, specifically for the Somali region.
    '''
    lam_region = pd.read_csv(thisdir / 'data' / 'lam_region.csv')
    lam_dict = {}
    lam_dict['month'] = lam_region.loc[lam_region['region'] == 'Somali']['month'].tolist()
    lam_dict['month'] = np.array(lam_dict['month'], dtype=np.float64)
    lam_dict['rate'] = lam_region.loc[lam_region['region'] == 'Somali']['rate'].tolist()
    lam_dict['rate'] = np.array(lam_dict['rate'], dtype=np.float64)
 

    return lam_dict

# %% Pregnancy exposure

def sexual_activity_region(): #NEEDS UPDATING
    '''
    Returns a linear interpolation of rates of female sexual activity, stratified by region
    '''
    sexually_active_region_data = pd.read_csv(thisdir / 'data' / 'sexual_activity_region.csv')
    sexually_active_region_dict = {}
    sexually_active_region_dict['age'] = sexually_active_region_data.loc[sexually_active_region_data['region']== 'Somali']['age'].tolist()   # Return age
    sexually_active_region_dict['age'] = np.array(sexually_active_region_dict['age'], dtype=np.float64)
    sexually_active_region_dict['perc'] = [x / 100 for x in sexually_active_region_data.loc[sexually_active_region_data['region']== 'Somali']['perc'].tolist()]
    sexually_active_region_dict['perc'] = np.array(sexually_active_region_dict['perc'], dtype=np.float64)

    activity_interp_model_region = si.interp1d(x=sexually_active_region_dict['age'], y=sexually_active_region_dict['perc'])
    activity_interp_region = activity_interp_model_region(fpd.spline_preg_ages) 
    
    return activity_interp_region

def sexual_activity_pp_region():
    '''
     # Returns an additional array of monthly likelihood of having resumed sexual activity by region
    '''
    pp_activity_region = pd.read_csv(thisdir / 'data' / 'sexual_activity_pp_region.csv')
    pp_activity_region_dict = {}
    pp_activity_region_dict['month'] = pp_activity_region.loc[pp_activity_region['region'] == 'Somali']['month'].tolist()
    pp_activity_region_dict['month'] = np.array(pp_activity_region_dict['month'], dtype=np.float64)
    pp_activity_region_dict['percent_active'] = pp_activity_region.loc[pp_activity_region['region'] == 'Somali']['perc'].tolist()
    pp_activity_region_dict['percent_active'] = np.array(pp_activity_region_dict['percent_active'], dtype=np.float64)
    
    return pp_activity_region_dict

def debut_age_region():
    '''
 #   Returns an additional array of weighted probabilities of sexual debut by region
    '''
    sexual_debut_region_data = pd.read_csv(thisdir / 'data' / 'sexual_debut_region.csv')
    debut_age_region_dict = {}
    debut_age_region_dict['ages'] = sexual_debut_region_data.loc[sexual_debut_region_data['region'] == 'Somali']['age'].tolist()
    debut_age_region_dict['ages'] = np.array(debut_age_region_dict['ages'], dtype=np.float64)
    debut_age_region_dict['probs'] = sexual_debut_region_data.loc[sexual_debut_region_data['region'] == 'Somali']['prob'].tolist()
    debut_age_region_dict['probs'] = np.array(debut_age_region_dict['probs'], dtype=np.float64)

    return debut_age_region_dict

def exposure_age():
    '''
    Returns an array of experimental factors to be applied to account for
    residual exposure to either pregnancy or live birth by age.  Exposure to pregnancy will
    increase factor number and residual likelihood of avoiding live birth (mostly abortion,
    also miscarriage), will decrease factor number
    '''
    exposure_correction_age = np.array([[0, 5, 10, 12.5, 15, 18, 20, 25, 30, 35, 40, 45, 50],
                                        [1, 1, 1, 1, 1, 1, 1, 0.8, 0.8, 0.8, 1, 1, 1]])
    exposure_age_interp = data2interp(exposure_correction_age, fpd.spline_preg_ages)

    return exposure_age_interp

def exposure_parity():
    '''
    Returns an array of experimental factors to be applied to account for residual exposure to either pregnancy
    or live birth by parity.
    '''
    exposure_correction_parity = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                                           [1, 1, 1, 1, 1, 1, 1, 0.8, 0.5, 0.3, 0.15, 0.10, 0.05, 0.01]])
    exposure_parity_interp = data2interp(exposure_correction_parity, fpd.spline_parities)

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

# %% Contraceptive methods

def mcpr():

    mcpr = {}
    cpr_data = pd.read_csv(thisdir / 'data' / 'cpr_region.csv')
    region_cpr_data = cpr_data.loc[cpr_data['region'] == 'Somali']
    mcpr['mcpr_years'] = region_cpr_data['year'].to_numpy()
    mcpr['mcpr_rates'] = region_cpr_data['cpr'].to_numpy() / 100  # convert from percent to rate

    return mcpr

# Define methods region based on empowerment work (self, region)
# Anything that is definied for the an individual person would be done in sim.py
# 


def barriers_region():
    '''
    Returns reasons for nonuse by region
    '''

    reasons_region = pd.read_csv(thisdir / 'data' / 'barriers_region.csv')
    reasons_region_dict = {}
    barriers = reasons_region.loc[reasons_region['region'] == 'Somali']['barrier'].tolist() # Return the reason for nonuse
    percs = reasons_region.loc[reasons_region['region'] == 'Somali']['perc'].tolist() # Return the percentage

    for i in range(len(barriers)):
        reasons_region_dict[barriers[i]] = percs[i]

    perc_total = sum(reasons_region_dict.values())
    normalized_dict = sc.odict({key: value / perc_total for key, value in reasons_region_dict.items()})   # Ensure the perc value sum to 100

    return normalized_dict

# %% Make and validate parameters

def make_pars(use_empowerment=None, use_education=None, use_partnership=None, use_subnational=None, seed=None):
    '''
    Take all parameters and construct into a dictionary
    '''

    # Scalar parameters and filenames
    pars = scalar_pars()
    pars['filenames'] = filenames()

    # Demographics and pregnancy outcome
    pars['age_pyramid'] = age_pyramid()
    pars['age_mortality'] = age_mortality()
    pars['maternal_mortality'] = maternal_mortality()
    pars['infant_mortality'] = infant_mortality()
    pars['miscarriage_rates'] = miscarriage()
    pars['stillbirth_rate'] = stillbirth()

    # Fecundity
    pars['age_fecundity'] = female_age_fecundity()
    pars['fecundity_ratio_nullip'] = fecundity_ratio_nullip()
    pars['lactational_amenorrhea'] = lactational_amenorrhea_region()

    # Pregnancy exposure
    pars['sexual_activity'] = sexual_activity_region()
    pars['sexual_activity_pp'] = sexual_activity_pp_region()
    pars['debut_age'] = debut_age_region()
    pars['exposure_age'] = exposure_age()
    pars['exposure_parity'] = exposure_parity()
    pars['spacing_pref'] = birth_spacing_pref()

    # Contraceptive methods
    pars['mcpr'] = mcpr()
    pars['barriers'] = barriers_region()

    # Regional parameters
    pars['urban_prop'] = urban_proportion()
    pars['region'] = region_proportions() # This function returns extrapolated and raw data

    kwargs = locals()
    not_implemented_args = ['use_empowerment', 'use_education', 'use_partnership']
    true_args = [key for key in not_implemented_args if kwargs[key] is True]
    if true_args:
        errmsg = f"{true_args} not implemented yet for {pars['location']}"
        raise NotImplementedError(errmsg)

    return pars