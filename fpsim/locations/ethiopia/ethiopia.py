'''
Set the parameters for FPsim, specifically for Ethiopia.
'''

import numpy as np
import pandas as pd
import sciris as sc
from scipy import interpolate as si
from fpsim import defaults as fpd
<<<<<<< HEAD
from . import regions
import fpsim.locations.data_utils as fpld
=======
>>>>>>> main


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
    return files


# %% Demographics and pregnancy outcome

<<<<<<< HEAD
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
=======
def age_pyramid():
    '''
    Starting age bin, male population, female population
    Data are from World Population Prospects
    https://population.un.org/wpp/Download/Standard/Population/
     '''
    pyramid = np.array([[0, 2018504, 1986199],  # Ethiopia 1962
                        [5, 1508878, 1515088],
                        [10, 1349237, 1359040],
                        [15, 1227673, 1215562],
                        [20, 1021618, 1018324],
                        [25, 862087, 864220],
                        [30, 727361, 732051],
                        [35, 612416, 620469],
                        [40, 510553, 521148],
                        [45, 423644, 434699],
                        [50, 345374, 360522],
                        [55, 276116, 297462],
                        [60, 182129, 224203],
                        [65, 117217, 162750],
                        [70, 77018, 111877],
                        [75, 40877,	66069],
                        [80, 21275,	40506],
                        ], dtype=float)

    return pyramid


def age_mortality():
    '''
    Age-dependent mortality rates taken from UN World Population Prospects 2022.  From probability of dying each year.
    https://population.un.org/wpp/
    Used CSV WPP2022_Life_Table_Complete_Medium_Female_1950-2021, Ethiopia, 2020 
    Used CSV WPP2022_Life_Table_Complete_Medium_Male_1950-2021, Ethiopia, 2020
    Mortality rate trend from crude death rate per 1000 people, also from UN Data Portal, 1950-2030:
    https://population.un.org/dataportal/data/indicators/59/locations/231/start/1950/end/2030/table/pivotbylocation
    Projections go out until 2030, but the csv file can be manually adjusted to remove any projections and stop at your desired year
    '''
    data_year = 2020 # NORMED TO 2020 BASED ON ETHIOPIA PROBABILITY DATA
    mortality_data = pd.read_csv(thisdir / 'data' / 'mortality_prob.csv')
    mortality_trend = pd.read_csv(thisdir / 'data' / 'mortality_trend.csv')

    mortality = {
        'ages': mortality_data['age'].to_numpy(),
        'm': mortality_data['male'].to_numpy(),
        'f': mortality_data['female'].to_numpy()
    }

    mortality['year'] = mortality_trend['year'].to_numpy()
    mortality['probs'] = mortality_trend['crude_death_rate'].to_numpy()
    trend_ind = np.where(mortality['year'] == data_year)
    trend_val = mortality['probs'][trend_ind]

    mortality['probs'] /= trend_val  # Normalize around data year for trending
    m_mortality_spline_model = si.splrep(x=mortality['ages'],
                                         y=mortality['m'])  # Create a spline of mortality along known age bins
    f_mortality_spline_model = si.splrep(x=mortality['ages'], y=mortality['f'])
    m_mortality_spline = si.splev(fpd.spline_ages,
                                  m_mortality_spline_model)  # Evaluate the spline along the range of ages in the model with resolution
    f_mortality_spline = si.splev(fpd.spline_ages, f_mortality_spline_model)
    m_mortality_spline = np.minimum(1, np.maximum(0, m_mortality_spline))  # Normalize
    f_mortality_spline = np.minimum(1, np.maximum(0, f_mortality_spline))

    mortality['m_spline'] = m_mortality_spline
    mortality['f_spline'] = f_mortality_spline

    return mortality


def maternal_mortality():
    '''
    From World Bank indicators for maternal mortality ratio (modeled estimate) per 100,000 live births:
    https://data.worldbank.org/indicator/SH.STA.MMRT?locations=ET
    '''

    data = np.array([
        [2000, 953],
        [2001, 955],
        [2002, 960],
        [2003, 906],
        [2004, 900],
        [2005, 880],
        [2006, 814],
        [2007, 780],
        [2008, 725],
        [2009, 684],
        [2010, 635],
        [2011, 603],
        [2012, 543],
        [2013, 498],
        [2014, 447],
        [2015, 399],
        [2016, 365],
        [2017, 348],
        [2018, 312],
        [2019, 294],
        [2020, 267],

    ])

    maternal_mortality = {}
    maternal_mortality['year'] = data[:, 0]
    maternal_mortality['probs'] = data[:, 1] / 100000  # ratio per 100,000 live births
    # maternal_mortality['ages'] = np.array([16, 17,   19, 22,   25, 50])
    # maternal_mortality['age_probs'] = np.array([2.28, 1.63, 1.3, 1.12, 1.0, 1.0]) #need to be added

    return maternal_mortality


def infant_mortality():
    '''
    From World Bank indicators for infant mortality (< 1 year) for Ethiopia, per 1000 live births
    From API_SP.DYN.IMRT.IN_DS2_en_csv_v2_5358355
    Adolescent increased risk of infant mortality gradient taken
    from Noori et al for Sub-Saharan African from 2014-2018.  Odds ratios with age 23-25 as reference group:
    https://www.medrxiv.org/content/10.1101/2021.06.10.21258227v1
    '''

    data = np.array([
        [1966, 146.6],        
        [1967, 146.1],
        [1968, 145.9],
        [1969, 145.8],
        [1970, 145.6],
        [1971, 145.4],
        [1972, 145.1],
        [1973, 145],
        [1974, 144.7],
        [1975, 144.4],
        [1976, 144.1],
        [1977, 143.8],
        [1978, 143.4],
        [1979, 142.6],
        [1980, 141.5],
        [1981, 139.9],
        [1982, 138],
        [1983, 135.8],
        [1984, 133.4],
        [1985, 131],
        [1986, 128.7],
        [1987, 126.4],
        [1988, 124.2],
        [1989, 121.9],
        [1990, 119.5],
        [1991, 116.9],
        [1992, 114],
        [1993, 110.7],
        [1994, 107.2],
        [1995, 103.7],
        [1996, 100.3],
        [1997, 96.8],
        [1998, 93.5],
        [1999, 90.3],
        [2000, 87],
        [2001, 83.7],
        [2002, 80.2],
        [2003, 76.7],
        [2004, 73.1],
        [2005, 69.5],
        [2006, 66.1],
        [2007, 62.8],
        [2008, 59.8],
        [2009, 57],
        [2010, 54.4],
        [2011, 51.9],
        [2012, 49.5],
        [2013, 47.3],
        [2014, 45.2],
        [2015, 43.2],
        [2016, 41.3],
        [2017, 39.6],
        [2018, 38],
        [2019, 36.6],
        [2020, 35.4],
        [2021, 34.3]
    ])

    infant_mortality = {}
    infant_mortality['year'] = data[:, 0]
    infant_mortality['probs'] = data[:, 1] / 1000  # Rate per 1000 live births, used after stillbirth is filtered out
    infant_mortality['ages'] = np.array([16, 17, 19, 22, 25, 50])
    infant_mortality['age_probs'] = np.array([2.28, 1.63, 1.3, 1.12, 1.0, 1.0])

    return infant_mortality


def miscarriage():
    '''
    Returns a linear interpolation of the likelihood of a miscarriage
    by age, taken from data from Magnus et al BMJ 2019: https://pubmed.ncbi.nlm.nih.gov/30894356/
    Data to be fed into likelihood of continuing a pregnancy once initialized in model
    Age 0 and 5 set at 100% likelihood.  Age 10 imputed to be symmetrical with probability at age 45 for a parabolic curve
    '''
    miscarriage_rates = np.array([[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
                                  [1, 1, 0.569, 0.167, 0.112, 0.097, 0.108, 0.167, 0.332, 0.569, 0.569]])
    miscarriage_interp = data2interp(miscarriage_rates, fpd.spline_preg_ages)
    return miscarriage_interp


def stillbirth():
    '''
    From Report of the UN Inter-agency Group for Child Mortality Estimation, 2020
    https://childmortality.org/wp-content/uploads/2020/10/UN-IGME-2020-Stillbirth-Report.pdf

    Age adjustments come from an extension of Noori et al., which were conducted June 2022. 
    '''

    data = np.array([ 
        [2000, 35.8],
        [2010, 31.1],
        [2019, 24.6],
    ])

    stillbirth_rate = {}
    stillbirth_rate['year'] = data[:, 0]
    stillbirth_rate['probs'] = data[:, 1] / 1000  # Rate per 1000 total births
    stillbirth_rate['ages'] = np.array([15, 16, 17, 19, 20, 28, 31, 36, 50])
    stillbirth_rate['age_probs'] = np.array([3.27, 1.64, 1.85, 1.39, 0.89, 1.0, 1.5, 1.55, 1.78])  # odds ratios

    return stillbirth_rate
>>>>>>> main


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


<<<<<<< HEAD
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
=======
def lactational_amenorrhea():
    '''
    Returns an array of the percent of breastfeeding women by month postpartum 0-11 months who meet criteria for LAM:
    Exclusively breastfeeding (bf + water alone), menses have not returned.  Extended out 5-11 months to better match data
    as those women continue to be postpartum insusceptible.
    From DHS Ethiopia 2016 calendar data
    '''
    data = np.array([
        [0, 0.9892715],
        [1, 0.8494371],
        [2, 0.8265375],
        [3, 0.730395],
        [4, 0.666934],
        [5, 0.4947101],
        [6, 0.3667066],
        [7, 0.2191162],
        [8, 0.3135374],
        [9, 0.1113177],
        [10, 0.0981782],
        [11, 0.0847675],
    ])

    lactational_amenorrhea = {}
    lactational_amenorrhea['month'] = data[:, 0]
    lactational_amenorrhea['rate'] = data[:, 1]

    return lactational_amenorrhea
>>>>>>> main


# %% Pregnancy exposure

<<<<<<< HEAD
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
=======
def sexual_activity():
    '''
    Returns a linear interpolation of rates of female sexual activity, defined as
    percentage women who have had sex within the last four weeks.
    From STAT Compiler DHS https://www.statcompiler.com/en/
    Using indicator "Timing of sexual intercourse"
    Includes women who have had sex "within the last four weeks"
    Excludes women who answer "never had sex", probabilities are only applied to agents who have sexually debuted
    Data taken from 2018 DHS, no trend over years for now
    Onset of sexual activity probabilities assumed to be linear from age 10 to first data point at age 15
    Last value duplicated so that it interpolates out to 50 and then stops
    '''

    sexually_active = np.array([[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
                                [0, 0, 0, 15, 51.2, 69.4, 69.3, 68.6, 68.1, 59.7, 59.7]])

    sexually_active[1] /= 100  # Convert from percent to rate per woman
    activity_ages = sexually_active[0]
    activity_interp_model = si.interp1d(x=activity_ages, y=sexually_active[1])
    activity_interp = activity_interp_model(fpd.spline_preg_ages)  # Evaluate interpolation along resolution of ages

    return activity_interp

>>>>>>> main


<<<<<<< HEAD
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
=======
def debut_age():
    '''
    Returns an array of weighted probabilities of sexual debut by a certain age 10-45.
    Data taken from DHS variable v531 (imputed age of sexual debut, imputed with data from age at first union)
    Use sexual_debut_age_probs.py under locations/data_processing to output for other DHS countries
    '''

    sexual_debut = np.array([
        [10, 0.00671240845335203],
        [11, 0.00918135820646243],
        [12, 0.0309770814788788],
        [13, 0.059400507503726],
        [14, 0.130291755291],
        [15, 0.196183864268175],
        [16, 0.130556610013873],
        [17, 0.103290455840828],
        [18, 0.110776245328648],
        [19, 0.0530816775521274],
        [20, 0.0588590881799291],
        [21, 0.026991174849838],
        [22, 0.0271788262050103],
        [23, 0.0188626403851833],
        [24, 0.0112214052863469],
        [25, 0.0109271507351524],
        [26, 0.00443999952806908],
        [27, 0.00359275321149036],
        [28, 0.00303477463739577],
        [29, 0.0017573689141809],
        [30, 0.00121215246872525],
        [31, 0.000711491329468429],
        [32, 0.000137332034070925],
        [33, 0.000279848072025066],
        [34, 7.17053090713206E-06],
        [35, 9.65008015799441E-05],
        [36, 8.46224502635213E-06],
        [37, 3.97705796721265E-05],
        [43, 0.00019012606885453]])

    debut_age = {}
    debut_age['ages'] = sexual_debut[:, 0]
    debut_age['probs'] = sexual_debut[:, 1]

    return debut_age
>>>>>>> main


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

<<<<<<< HEAD

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

=======
>>>>>>> main

# %% Make and validate parameters

<<<<<<< HEAD
def make_pars(location='ethiopia', seed=None, use_subnational=None):
=======
def make_pars(use_empowerment=None, use_education=None, use_partnership=None, seed=None):
>>>>>>> main
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

    kwargs = locals()
    not_implemented_args = []
    true_args = [key for key in not_implemented_args if kwargs[key] is True]
    if true_args:
        errmsg = f"{true_args} not implemented yet for {pars['location']}"
        raise NotImplementedError(errmsg)

    return pars
