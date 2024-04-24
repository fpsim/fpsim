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
    scalar_pars['location'] = 'amhara'
    scalar_pars['breastfeeding_dur_mu'] = 12.5669
    scalar_pars['breastfeeding_dur_beta'] = 10.75467

    return scalar_pars

def data2interp(data, ages, normalize=False):
    ''' Convert unevenly spaced data into an even spline interpolation '''
    model = si.interp1d(data[0], data[1])
    interp = model(ages)
    if normalize:
        interp = np.minimum(1, np.maximum(0, interp))
    return interp

def filenames():
    files = eth.filenames()
    
    return files

# %% Demographics and pregnancy outcome
'''
Data from 1994 Census Report for Amhara Region
https://www.statsethiopia.gov.et/wp-content/uploads/2019/06/Population-and-Housing-Census-1994-Amhara-Region.pdf
'''
def age_pyramid():
    pyramid = np.array([[0, 1063425, 1039862], # Amhara 1994 
                        [5, 1083627, 1072118],
                        [10, 962649, 885335],
                        [15, 742766, 740028],
                        [20, 525138, 564480],
                        [25, 481007, 531523],
                        [30, 366492, 420412],
                        [35, 342041, 367473],
                        [40, 292577, 309360],
                        [45, 248468, 211516],
                        [50, 224383, 236333],
                        [55, 149310, 120785],
                        [60, 164406, 154539],
                        [65, 109904, 78971],
                        [70, 92528, 80203],
                        [75, 45471, 31283],
                        [80, 53354, 42530]
                        ], dtype=float)    
    return pyramid

def urban_proportion(): # TODO: Flagging this - currently this is being used for the urban ratio for amhara; I assume you'd want to use the region-specific value?
    urban_data = eth.urban_proportion()
    
    return urban_data  # Return this value as a float

def region_proportions():
    '''
    Defines the proportion of the population in the Amhara region to establish the probability of living there.
    '''
    region_data = pd.read_csv(thisdir / '..' / 'ethiopia' / 'subnational_data' / 'region.csv')
    region_dict = {}
    region_dict['mean'] = region_data.loc[region_data['region'] == 'Amhara']['mean'] #HOLDER FOR NOW. NEED TO CALL ON LOCATION.
    region_dict['urban'] = region_data.loc[region_data['region'] == 'Amhara']['urban']
    
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
    Returns a dictionary containing the percent of breastfeeding women by month postpartum 0-11 months who meet criteria for LAM, specifically for the Amhara region.
    '''
    lam_region = pd.read_csv(thisdir / '..' / 'ethiopia' / 'subnational_data' / 'lam_region.csv')
    lam_dict = {}
    lam_dict['month'] = lam_region.loc[lam_region['region'] == 'Amhara']['month'].tolist()
    lam_dict['month'] = np.array(lam_dict['month'], dtype=np.float64)
    lam_dict['rate'] = lam_region.loc[lam_region['region'] == 'Amhara']['rate'].tolist()
    lam_dict['rate'] = np.array(lam_dict['rate'], dtype=np.float64)
 

    return lam_dict

# %% Pregnancy exposure

def sexual_activity_region(): #NEEDS UPDATING
    '''
    Returns a linear interpolation of rates of female sexual activity, stratified by region
    '''
    sexually_active_region_data = pd.read_csv(thisdir / '..' / 'ethiopia' / 'subnational_data' / 'sexual_activity_region.csv')
    sexually_active_region_dict = {}
    sexually_active_region_dict['age'] = sexually_active_region_data.loc[sexually_active_region_data['region']== 'Amhara']['age'].tolist()   # Return age
    sexually_active_region_dict['age'] = np.array(sexually_active_region_dict['age'], dtype=np.float64)
    sexually_active_region_dict['perc'] = [x / 100 for x in sexually_active_region_data.loc[sexually_active_region_data['region']== 'Amhara']['perc'].tolist()]
    sexually_active_region_dict['perc'] = np.array(sexually_active_region_dict['perc'], dtype=np.float64)

    activity_interp_model_region = si.interp1d(x=sexually_active_region_dict['age'], y=sexually_active_region_dict['perc'])
    activity_interp_region = activity_interp_model_region(fpd.spline_preg_ages) 
    
    return activity_interp_region

def sexual_activity_pp_region():
    '''
     # Returns an additional array of monthly likelihood of having resumed sexual activity by region
    '''
    pp_activity_region = pd.read_csv(thisdir / '..' / 'ethiopia' / 'subnational_data' / 'sexual_activity_pp_region.csv')
    pp_activity_region_dict = {}
    pp_activity_region_dict['month'] = pp_activity_region.loc[pp_activity_region['region'] == 'Amhara']['month'].tolist()
    pp_activity_region_dict['month'] = np.array(pp_activity_region_dict['month'], dtype=np.float64)
    pp_activity_region_dict['percent_active'] = pp_activity_region.loc[pp_activity_region['region'] == 'Amhara']['perc'].tolist()
    pp_activity_region_dict['percent_active'] = np.array(pp_activity_region_dict['percent_active'], dtype=np.float64)
    
    return pp_activity_region_dict

def debut_age_region():
    '''
 #   Returns an additional array of weighted probabilities of sexual debut by region
    '''
    sexual_debut_region_data = pd.read_csv(thisdir / '..' / 'ethiopia' / 'subnational_data' / 'sexual_debut_region.csv')
    debut_age_region_dict = {}
    debut_age_region_dict['ages'] = sexual_debut_region_data.loc[sexual_debut_region_data['region'] == 'Amhara']['age'].tolist()
    debut_age_region_dict['ages'] = np.array(debut_age_region_dict['ages'], dtype=np.float64)
    debut_age_region_dict['probs'] = sexual_debut_region_data.loc[sexual_debut_region_data['region'] == 'Amhara']['prob'].tolist()
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
                                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
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

def methods():
    methods = eth.methods()
    return methods

# Define methods region based on empowerment work (self, region)
# Anything that is definied for the an individual person would be done in sim.py
# 

def method_probs():
    '''
    Define "raw" (un-normalized, un-trended) matrices to give transitional probabilities
    from PMA Ethiopia contraceptive calendar data.

    Probabilities in this function are annual probabilities of initiating (top row), discontinuing (first column),
    continuing (diagonal), or switching methods (all other entries).

    Probabilities at postpartum month 1 are 1 month transitional probabilities
    for starting a method after delivery.

    Probabilities at postpartum month 6 are 5 month transitional probabilities
    for starting or changing methods over the first 6 months postpartum.

    Data from Ethiopia PMA contraceptive calendars, 2019-2020
    Processed from matrices_ethiopia_pma_2019_20.csv using process_matrices.py
    '''

    raw = {

        # Main switching matrix: all non-postpartum women
        'annual': {
            '<18': np.array([
                [0.9236, 0.0004, 0.0017, 0.0007, 0.0042, 0.0596, 0.0076, 0.0006, 0.    , 0.0017],
                [0.    , 1.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ],
                [0.1374, 0.    , 0.8571, 0.    , 0.0003, 0.0044, 0.0005, 0.    , 0.    , 0.0001],
                [0.2815, 0.0001, 0.0003, 0.6995, 0.0007, 0.0164, 0.0012, 0.0001, 0.    , 0.0003],
                [0.4681, 0.0001, 0.0006, 0.0013, 0.3023, 0.1869, 0.003 , 0.0005, 0.    , 0.0371],
                [0.3267, 0.0001, 0.0017, 0.0005, 0.0043, 0.6549, 0.009 , 0.0023, 0.    , 0.0005],
                [0.1937, 0.    , 0.0008, 0.0001, 0.0007, 0.0087, 0.7957, 0.0001, 0.    , 0.0002],
                [0.4158, 0.0001, 0.0004, 0.0002, 0.0011, 0.0142, 0.0017, 0.5662, 0.    , 0.0004],
                [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 1.    , 0.    ],
                [0.5988, 0.0001, 0.0006, 0.024 , 0.0186, 0.0258, 0.0027, 0.0002, 0.    , 0.3293]]),
            '18-20': np.array([
                [0.8783, 0.0002, 0.0021, 0.0017, 0.0115, 0.0845, 0.0194, 0.0016, 0.    , 0.0006],
                [0.0599, 0.9362, 0.0001, 0.0001, 0.0004, 0.0027, 0.0006, 0.0001, 0.    , 0.    ],
                [0.1874, 0.    , 0.8002, 0.0002, 0.0012, 0.0088, 0.002 , 0.0002, 0.    , 0.0001],
                [0.8023, 0.0001, 0.0012, 0.117 , 0.0068, 0.0498, 0.0109, 0.009 , 0.    , 0.0028],
                [0.2326, 0.    , 0.0003, 0.0003, 0.7384, 0.025 , 0.0026, 0.0003, 0.    , 0.0004],
                [0.279 , 0.    , 0.0003, 0.0004, 0.0069, 0.6868, 0.0219, 0.0045, 0.    , 0.0002],
                [0.2046, 0.    , 0.0002, 0.0018, 0.0058, 0.0224, 0.7647, 0.0002, 0.    , 0.0001],
                [0.3239, 0.    , 0.0004, 0.0004, 0.0372, 0.1151, 0.0048, 0.5179, 0.    , 0.0002],
                [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 1.    , 0.    ],
                [0.8974, 0.0002, 0.0017, 0.0016, 0.0093, 0.0693, 0.0156, 0.0013, 0.    , 0.0036]]),
            '21-25': np.array([
                [0.8704, 0.0008, 0.0026, 0.0011, 0.0106, 0.0794, 0.0291, 0.0047, 0.    , 0.0012],
                [0.1535, 0.8357, 0.0002, 0.0001, 0.0009, 0.0067, 0.0024, 0.0004, 0.    , 0.0001],
                [0.0473, 0.    , 0.8681, 0.    , 0.0333, 0.0408, 0.0047, 0.0004, 0.    , 0.0053],
                [0.0851, 0.    , 0.0001, 0.8887, 0.0006, 0.0237, 0.0014, 0.0003, 0.    , 0.0001],
                [0.3553, 0.0002, 0.0027, 0.0002, 0.4916, 0.1088, 0.0297, 0.0111, 0.    , 0.0003],
                [0.2505, 0.0001, 0.0014, 0.0002, 0.0118, 0.7151, 0.0152, 0.0052, 0.    , 0.0005],
                [0.1413, 0.0001, 0.0002, 0.0001, 0.0027, 0.0409, 0.8127, 0.0019, 0.    , 0.0001],
                [0.0687, 0.    , 0.0001, 0.    , 0.0007, 0.0038, 0.0265, 0.9002, 0.    , 0.0001],
                [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 1.    , 0.    ],
                [0.7042, 0.0004, 0.0254, 0.0005, 0.0214, 0.061 , 0.0145, 0.0024, 0.    , 0.1703]]),
            '26-35': np.array([
                [0.875 , 0.0001, 0.0014, 0.0008, 0.0078, 0.0827, 0.0261, 0.0058, 0.0002, 0.0001],
                [0.0664, 0.8974, 0.0001, 0.    , 0.0005, 0.0342, 0.0012, 0.0003, 0.    , 0.    ],
                [0.1795, 0.    , 0.8028, 0.0001, 0.0008, 0.0081, 0.0081, 0.0005, 0.    , 0.    ],
                [0.1613, 0.    , 0.0002, 0.7594, 0.001 , 0.0613, 0.0027, 0.0009, 0.    , 0.013 ],
                [0.3407, 0.    , 0.0018, 0.0002, 0.4605, 0.1452, 0.0463, 0.0052, 0.0001, 0.    ],
                [0.2059, 0.0002, 0.0016, 0.0002, 0.0094, 0.7561, 0.0203, 0.0059, 0.0004, 0.    ],
                [0.1098, 0.    , 0.0007, 0.    , 0.0014, 0.0176, 0.8692, 0.001 , 0.    , 0.0002],
                [0.0641, 0.    , 0.0013, 0.    , 0.0003, 0.0028, 0.0009, 0.9306, 0.    , 0.    ],
                [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 1.    , 0.    ],
                [0.3603, 0.    , 0.0003, 0.0002, 0.0017, 0.0172, 0.0052, 0.0338, 0.    , 0.5813]]),
            '>35': np.array([
                [0.9342, 0.0001, 0.0009, 0.0001, 0.0034, 0.0482, 0.0118, 0.001 , 0.    , 0.0002],
                [0.    , 1.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ],
                [0.0867, 0.    , 0.9104, 0.    , 0.0002, 0.0021, 0.0005, 0.    , 0.    , 0.    ],
                [0.1769, 0.    , 0.0001, 0.8162, 0.0003, 0.0053, 0.0011, 0.0001, 0.    , 0.    ],
                [0.1985, 0.    , 0.0141, 0.    , 0.6921, 0.066 , 0.0075, 0.0183, 0.    , 0.0033],
                [0.1747, 0.0003, 0.0005, 0.    , 0.0061, 0.8054, 0.0117, 0.0012, 0.    , 0.    ],
                [0.0998, 0.    , 0.0013, 0.0002, 0.0061, 0.0315, 0.861 , 0.0001, 0.    , 0.    ],
                [0.1299, 0.    , 0.0002, 0.    , 0.0148, 0.0037, 0.0008, 0.8504, 0.    , 0.    ],
                [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 1.    , 0.    ],
                [0.0957, 0.    , 0.    , 0.    , 0.0002, 0.0023, 0.0006, 0.    , 0.    , 0.9012]])
        },

        # Postpartum switching matrix, 1 to 6 months
        'pp1to6': {
            '<18': np.array([
                [0.8316, 0.    , 0.0046, 0.    , 0.0069, 0.1376, 0.018 , 0.0012,0.    , 0.    ],
                [0.    , 1.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,0.    , 0.    ],
                [0.    , 0.    , 1.    , 0.    , 0.    , 0.    , 0.    , 0.    ,0.    , 0.    ],
                [0.    , 0.    , 0.    , 1.    , 0.    , 0.    , 0.    , 0.    ,0.    , 0.    ],
                [0.0135, 0.    , 0     , 0     , 0.1861, 0.8004, 0.    , 0.    ,0.    , 0.    ], 
                [0.0036, 0.    , 0.    , 0.    , 0.    , 0.9964, 0.    , 0.    ,0.    , 0.    ], 
                [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 1.    , 0.    ,0.    , 0.    ],
                [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 1.    ,0.    , 0.    ],
                [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,1.    , 0.    ],
                [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,0.    , 1.    ]]),
            '18-20': np.array([
                [0.7971, 0.    , 0.0027, 0.    , 0.0046, 0.1534, 0.0383, 0.004, 0.    , 0.    ],
                [0.    , 1.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,0.    , 0.    ],
                [0.    , 0.    , 1.    , 0.    , 0.    , 0.    , 0.    , 0.    ,0.    , 0.    ],
                [0.    , 0.    , 0.    , 1.    , 0.    , 0.    , 0.    , 0.    ,0.    , 0.    ],
                [0.    , 0.    , 0.    , 0.    , 1.    , 0.    , 0.    , 0.    ,0.    , 0.    ],
                [0.2151, 0.    , 0.    , 0.    , 0.0337, 0.7016, 0.0496, 0.    ,0.    , 0     ],
                [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 1.    , 0.    ,0.    , 0.    ],
                [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 1.    ,0.    , 0.    ],
                [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,1.    , 0.    ],
                [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,0.    , 1.    ]]),
            '21-25': np.array([ 
                [0.807 , 0.0005, 0.0011, 0.0011, 0.0187, 0.1338, 0.0307, 0.0067,0.    , 0.0004],
                [0.    , 1.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,0.    , 0.    ],
                [0.    , 0.    , 1.    , 0.    , 0.    , 0.    , 0.    , 0.    ,0.    , 0.    ],
                [0.    , 0.    , 0.    , 1.    , 0.    , 0.    , 0.    , 0.    ,0.    , 0.    ],
                [0.    , 0.    , 0.    , 0.    , 0.3813, 0.6187, 0.    , 0.    ,0.    , 0.    ],
                [0.0722, 0.    , 0.    , 0.    , 0.    , 0.9278, 0.    , 0.    ,0.    , 0.    ],
                [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 1.    , 0.    ,0.    , 0.    ],
                [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 1.    ,0.    , 0.    ],
                [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,1.    , 0.    ],
                [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,0.    , 1.    ]]),
            '26-35': np.array([
                [0.8582, 0.0007, 0.0019, 0.0001, 0.0118, 0.1083, 0.0151, 0.0039, 0, 0],
                [0.    , 1.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,0.    , 0.    ],
                [0.    , 0.    , 1.    , 0.    , 0.    , 0.    , 0.    , 0.    ,0.    , 0.    ],
                [0.    , 0.    , 0.    , 1.    , 0.    , 0.    , 0.    , 0.    ,0.    , 0.    ],
                [0.2954, 0.    , 0.    , 0.    , 0.6214, 0.    , 0.    , 0.0832,0.    , 0.    ],
                [0.    , 0.    , 0.    , 0.    , 0.    , 0.989 , 0.    , 0.0067,0.0043, 0.    ],
                [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 1.    , 0.    ,0.    , 0.    ],
                [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 1.    ,0.    , 0.    ],
                [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,1.    , 0.    ],
                [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,0.    , 1.    ]]),
            '>35': np.array([
                [0.8897, 0.    , 0.0035, 0.0004, 0.0012, 0.0733, 0.0194, 0.0099,0.    , 0.0025],
                [0.    , 1.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,0.    , 0.    ],
                [0.    , 0.    , 1.    , 0.    , 0.    , 0.    , 0.    , 0.    ,0.    , 0.    ],
                [0.    , 0.    , 0.    , 1.    , 0.    , 0.    , 0.    , 0.    ,0.    , 0.    ],
                [0.    , 0.    , 0.    , 0.    , 1.    , 0.    , 0.    , 0.    ,0.    , 0.    ],
                [0.0568, 0.    , 0.    , 0.    , 0.    , 0.9432, 0.    , 0.    ,0.    , 0.    ],
                [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 1.    , 0.    ,0.    , 0.    ],
                [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 1.    ,0.    , 0.    ],
                [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,1.    , 0.    ],
                [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,0.    , 1.    ]])
        },

        # Postpartum initiation vectors, 0 to 1 month 
        'pp0to1': {
            '<18': np.array([0.95, 0., 0., 0., 0.0031, 0.0383, 0.0073, 0.0013, 0., 0.]),
            '18-20': np.array([0.9382, 0., 0., 0., 0.0015, 0.0466, 0.0118, 0.0019, 0., 0.]),
            '21-25': np.array([0.9583, 0.0006, 0., 0., 0.0003, 0.0343, 0.0053, 0.0013, 0., 0.]),
            '26-35': np.array([0.9693, 0., 0.0005, 0., 0.0032, 0.0195, 0.0044, 0.0025, 0.0005, 0.]),
            '>35': np.array([0.9556, 0., 0., 0., 0.0009, 0.0298, 0.0028, 0.0048, 0.006, 0.]),
        }
    }

    return raw

def barriers_region():
    '''
    Returns reasons for nonuse by region
    '''

    reasons_region = pd.read_csv(thisdir / '..' / 'ethiopia' / 'subnational_data' / 'barriers_region.csv')
    reasons_region_dict = {}
    barriers = reasons_region.loc[reasons_region['region'] == 'Amhara']['barrier'].tolist() # Return the reason for nonuse
    percs = reasons_region.loc[reasons_region['region'] == 'Amhara']['perc'].tolist() # Return the percentage

    for i in range(len(barriers)):
        reasons_region_dict[barriers[i]] = percs[i]

    perc_total = sum(reasons_region_dict.values())
    normalized_dict = sc.odict({key: value / perc_total for key, value in reasons_region_dict.items()})   # Ensure the perc value sum to 100

    return normalized_dict

# %% Make and validate parameters

def make_pars():
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
    pars['methods'] = methods()
    pars['methods']['raw'] = method_probs()
    pars['barriers'] = barriers_region()

    # Regional parameters
    pars['urban_prop'] = urban_proportion()
    pars['region'] = region_proportions() # This function returns extrapolated and raw data

    return pars