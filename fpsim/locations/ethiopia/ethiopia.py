'''
Set the parameters for FPsim, specifically for Ethiopia.
'''

import numpy as np
import pandas as pd
import sciris as sc
from scipy import interpolate as si
from fpsim import defaults as fpd
from . import regions

# %% Housekeeping

thisdir = sc.thispath(__file__)  # For loading CSV files


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


def data2interp(data, ages, normalize=False):
    ''' Convert unevenly spaced data into an even spline interpolation '''
    model = si.interp1d(data[0], data[1])
    interp = model(ages)
    if normalize:
        interp = np.minimum(1, np.maximum(0, interp))
    return interp


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


def region_proportions():
    '''
    Defines the proportion of the population in each region to establish the probability of living in a given region.
    Uses 2016 Ethiopia DHS individual recode (v025) for region and V024 for urban to produce subnational estimates
    '''
    region_data = pd.read_csv(thisdir / 'data' / 'region.csv')

    region_dict = {}
    region_dict['region'] = region_data['region'] # Return region names
    region_dict['mean'] = region_data['mean'] # Return proportion living in each region
    region_dict['urban'] = region_data['urban'] # Return proportion living in an urban area by region

    return region_dict


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
    fecundity_nullip_interp = data2interp(fecundity_ratio_nullip, fpd.spline_preg_ages)

    return fecundity_nullip_interp


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


def lactational_amenorrhea_region():
    '''
    Returns an array of the percent of breastfeeding women by month postpartum 0-11 months who meet criteria for LAM, stratified by region
    '''
    lam_region = pd.read_csv(thisdir / 'data' / 'lam_region.csv')
    lam_dict = {}
    lam_dict['region'] = lam_region['region']  # Return region names
    lam_dict['month'] = lam_region['month']  # Return month postpartum
    lam_dict['rate'] = lam_region['rate']  # Return percent of breastfeeding women

    return lam_dict


# %% Pregnancy exposure

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

def sexual_activity_region():
    '''
    Returns a linear interpolation of rates of female sexual activity, stratified by region
    '''
    sexually_active_region_data = pd.read_csv(thisdir / 'data' / 'sexual_activity_region.csv')
    sexually_active_region_dict = {}
    sexually_active_region_dict['region'] = sexually_active_region_data.iloc[:, 0]  # Return region names
    sexually_active_region_dict['age'] = sexually_active_region_data.iloc[:, 1]   # Return age
    sexually_active_region_dict['perc'] = sexually_active_region_data.iloc[:, 2] / 100  # Return perc divide by 100 to convert to a rate
    activity_ages_region = sexually_active_region_dict['age']
    activity_interp_model_region = si.interp1d(x=activity_ages_region, y=sexually_active_region_dict['perc'])
    activity_interp_region = activity_interp_model_region(fpd.spline_preg_ages)

    return activity_interp_region

def sexual_activity_pp():
    '''
    Returns an array of monthly likelihood of having resumed sexual activity within 0-35 months postpartum
    Uses 2016 Ethiopia DHS individual recode (postpartum (v222), months since last birth, and sexual activity within 30 days.
    Data is weighted.
    Limited to 23 months postpartum (can use any limit you want 0-23 max)
    Postpartum month 0 refers to the first month after delivery
    '''

    postpartum_sex = np.array([
        [0, 0.19253],
        [1, 0.31858],
        [2, 0.49293],
        [3, 0.64756],
        [4, 0.78684],
        [5, 0.67009],
        [6, 0.75804],
        [7, 0.79867],
        [8, 0.84857],
        [9, 0.80293],
        [10, 0.89259],
        [11, 0.8227],
        [12, 0.85876],
        [13, 0.83104],
        [14, 0.77563],
        [15, 0.79917],
        [16, 0.79582],
        [17, 0.84817],
        [18, 0.77804],
        [19, 0.80811],
        [20, 0.82049],
        [21, 0.77607],
        [22, 0.79261],
        [23, 0.8373],
    ])



    postpartum_activity = {}
    postpartum_activity['month'] = postpartum_sex[:, 0]
    postpartum_activity['percent_active'] = postpartum_sex[:, 1]

    return postpartum_activity


def sexual_activity_pp_region():
    '''
     # Returns an additional array of monthly likelihood of having resumed sexual activity by region
    '''
    pp_activity_region = pd.read_csv(thisdir / 'data' / 'sexual_activity_pp_region.csv')
    pp_activity_region_dict = {}
    pp_activity_region_dict['region'] = pp_activity_region['region'] # Return region names
    pp_activity_region_dict['month'] = pp_activity_region['month'] # Return month postpartum
    pp_activity_region_dict['perc'] = pp_activity_region['perc'] # Return likelihood of resumed sexual activity

    return pp_activity_region_dict


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

def debut_age_region():
    '''
 #   Returns an additional array of weighted probabilities of sexual debut by region
    '''
    sexual_debut_region_data = pd.read_csv(thisdir / 'data' / 'sexual_debut_region.csv')
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


def age_spline(which):
    return pd.read_csv(thisdir / 'data' / f'age_spline_{which}.csv', index_col=0)


def age_partnership():
    """ Probabilities of being partnered at age X"""
    age_partnership_data = pd.read_csv(thisdir / 'data' / 'age_partnership.csv')
    partnership_dict = {}
    partnership_dict["age"] = age_partnership_data["age_partner"].to_numpy()
    partnership_dict["partnership_probs"] = age_partnership_data["percent"].to_numpy()
    return  partnership_dict


def wealth():
    """ Process percent distribution of people in each wealth quintile"""
    cols = ["quintile", "percent"]
    wealth_data = pd.read_csv(thisdir / 'data' / 'wealth.csv', header=0, names=cols)
    return wealth_data


# %% Education
def education_objective(df):
    """
    Transforms education objective data from a DataFrame into a numpy array. The DataFrame represents
    the proportion of women with different education objectives, stratified by urban/rural residence.

    The 'percent' column represents the proportion of women aiming for 'edu' years of education. The data
    is based on education completed by women over 20 with no children, stratified by urban/rural residence
    from the Demographic and Health Surveys (DHS).

    Args:
        df (pd.DataFrame): Contains 'urban', 'edu' and 'percent' columns, with 'edu' ranging from
                           0 to a maximum value representing years of education.

    Returns:
        arr (np.array): A 2D array of shape (2, n_edu_objective_years) containing proportions. The first
                        row corresponds to 'rural' women and the second row to 'urban' women.
    """
    arr = df["percent"].to_numpy().reshape(df["urban"].nunique(), df["edu"].nunique())
    return arr


def education_attainment(df):
    """
    Transforms education attainment data (from education_initialization.csv) from a DataFrame
    into a numpy array. The DataFrame represents the average (mean) number of years of
    education 'edu', a woman aged 'age' has attained/completed.

    Args:
        df (pd.DataFrame): Contains 'age', 'edu' columns.

    Returns:
        arr (np.array): A 1D with array with interpolated values of 'edu' years attained.

    NOTE: The data in education_initialization.csv have been extrapolated to cover the age range
    [0, 99], inclusive range. Here we only interpolate data for the group 15-49 (inclusive range).
    """
    # This df has columns
    # age:age in years and edu: mean years of education
    df.sort_values(by="age", ascending=True, inplace=True)
    ages = df["age"].to_numpy()
    arr  = df["edu"].to_numpy()

    # Get indices of those ages
    inds = np.array(sc.findinds(ages >= fpd.min_age, ages < fpd.max_age_preg+5)) # interpolate beyond DHS max age to avoid discontinuities
    arr[inds] = sc.smooth(arr[inds], 3)
    return arr


def education_dropout_probs(df):
    """
    Transforms education dropout probabilities (from edu_stop.csv) from a DataFrame
    into a dictionary. The dataframe will be used as the probability that a woman aged 'age'
    and with 'parity' number of children, would drop out of school if she is enrolled in
    education and becomes pregnant.

    The data comes from PMA Datalab and 'percent' represents the probability of stopping/droppping
    out of education within 1 year of the birth represented in 'parity'.
        - Of women with a first birth before age 18, 12.6% stopped education within 1 year of that birth.
        - Of women who had a subsequent (not first) birth before age 18, 14.1% stopped school within 1 year of that birth.

    Args:
        df (pd.DataFrame): Contains 'age', 'parity' and 'percent' columns.

    Returns:
       data (dictionary): Dictionary with keys '1' and '2+' indicating a woman's parity. The values are dictionaries,
           each with keys 'age' and 'percent'.
    """
    data = {}
    for k in df["parity"].unique():
        data[k] = {"age": None, "percent": None}
        data[k]["age"] = df["age"].unique()
        data[k]["percent"] = df["percent"][df["parity"] == k].to_numpy()
    return data


def education_distributions():
    """
    Loads and processes all education-related data files. Performs additional interpolation
    to reduce noise from empirical data.

    Returns:
        Tuple[dict, dict]: A tuple of two dictionaries:
            education_data (dict): Contains the unmodified empirical data from CSV files.
            education_dict (dict): Contains the processed empirical data to use in simulations.
    """

    # Load empirical data
    data_path = thisdir / "data"

    education ={"edu_objective":
                    {"data_file": "edu_objective.csv",
                     "process_function": education_objective},
                "edu_attainment":
                    {"data_file": "edu_initialization.csv",
                     "process_function": education_attainment},
                "edu_dropout_probs":
                    {"data_file": "edu_stop.csv",
                     "process_function": education_dropout_probs}}

    education_data = dict()
    education_dict = dict()
    for edu_key in education.keys():
        education_data[edu_key] = pd.read_csv(data_path / education[edu_key]["data_file"])
        education_dict[edu_key] = education[edu_key]["process_function"](education_data[edu_key])
    education_dict.update({"age_start": 6.0})
    return education_dict, education_data


def process_contra_use_pars():
    raw_pars = pd.read_csv(thisdir / 'data' / 'contra_coef.csv')
    pars = sc.objdict()
    for var_dict in raw_pars.to_dict('records'):
        var_name = var_dict['rhs'].replace('_0', '').replace('(', '').replace(')', '').lower()
        pars[var_name] = var_dict['Estimate']
    return pars


def process_contra_use(which):
    """
    Process cotraceptive use parameters.
    Args:
        which: either 'simple' or 'mid'
    """

    # Read in data
    alldfs = [
        pd.read_csv(thisdir / 'data' / f'contra_coef_{which}.csv'),
        pd.read_csv(thisdir / 'data' / f'contra_coef_{which}_pp1.csv'),
        pd.read_csv(thisdir / 'data' / f'contra_coef_{which}_pp6.csv'),
    ]

    contra_use_pars = dict()

    for di, df in enumerate(alldfs):
        if which == 'mid':
            contra_use_pars[di] = sc.objdict(
                intercept=df[df['rhs'].str.contains('Intercept')].Estimate.values[0],
                age_factors=df[df['rhs'].str.contains('age') & ~df['rhs'].str.contains('fp_ever_user')].Estimate.values,
                ever_used_contra=df[df['rhs'].str.contains('fp_ever_user') & ~df['rhs'].str.contains('age')].Estimate.values[0],
                edu_attainment=df[df['rhs'].str.contains('edu_attainment')].Estimate.values[0],
                parity=df[df['rhs'].str.contains('parity')].Estimate.values[0],
                urban=df[df['rhs'].str.contains('urban')].Estimate.values[0],
                wealthquintile=df[df['rhs'].str.contains('wealthquintile')].Estimate.values[0],
                age_ever_user_factors=df[df['rhs'].str.contains('age') & df['rhs'].str.contains('fp_ever_user')].Estimate.values,
            )

        elif which == 'simple':
            contra_use_pars[di] = sc.objdict(
                intercept=df[df['rhs'].str.contains('Intercept')].Estimate.values[0],
                age_factors=df[df['rhs'].str.match('age') & ~df['rhs'].str.contains('fp_ever_user')].Estimate.values,
                fp_ever_user=df[df['rhs'].str.contains('fp_ever_user') & ~df['rhs'].str.contains('age')].Estimate.values[0],
                age_ever_user_factors=df[df['rhs'].str.match('age') & df['rhs'].str.contains('fp_ever_user')].Estimate.values,
            )

    return contra_use_pars


def process_markovian_method_choice(methods, df=None):
    """ Choice of method is age and previous method """
    if df is None:
        df = pd.read_csv(thisdir / 'data' / 'method_mix_matrix_switch.csv', keep_default_na=False, na_values=['NaN'])
    csv_map = {method.csv_name: method.name for method in methods.values()}
    idx_map = {method.csv_name: method.idx for method in methods.values()}
    idx_df = {}
    for col in df.columns:
        if col in csv_map.keys():
            idx_df[col] = idx_map[col]

    mc = dict()  # This one is a dict because it will be keyed with numbers
    init_dist = sc.objdict()  # Initial distribution of method choice

    # Get end index
    ei = 4+len(methods)-1

    for pp in df.postpartum.unique():
        mc[pp] = sc.objdict()
        mc[pp].method_idx = np.array(list(idx_df.values()))
        for akey in df.age_grp.unique():
            mc[pp][akey] = sc.objdict()
            thisdf = df.loc[(df.age_grp == akey) & (df.postpartum == pp)]
            if pp == 1:  # Different logic for immediately postpartum
                mc[pp][akey] = thisdf.values[0][4:ei].astype(float)  # If this is going to be standard practice, should make it more robust
            else:
                from_methods = thisdf.From.unique()
                for from_method in from_methods:
                    from_mname = csv_map[from_method]
                    row = thisdf.loc[thisdf.From == from_method]
                    mc[pp][akey][from_mname] = row.values[0][4:ei].astype(float)

            # Set initial distributions by age
            if pp == 0:
                init_dist[akey] = thisdf.loc[thisdf.From == 'None'].values[0][4:ei].astype(float)
                init_dist.method_idx = np.array(list(idx_df.values()))

    return mc, init_dist


def process_dur_use(methods, df=None):
    """ Process duration of use parameters"""
    if df is None:
        df = pd.read_csv(thisdir / 'data' / 'method_time_coefficients.csv', keep_default_na=False, na_values=['NaN'])
    for method in methods.values():
        if method.name == 'btl':
            method.dur_use = dict(dist='lognormal', par1=100, par2=1)
        else:
            mlabel = method.csv_name

            thisdf = df.loc[df.method == mlabel]
            dist = thisdf.functionform.iloc[0]
            method.dur_use = dict()
            age_ind = sc.findfirst(thisdf.coef.values, 'age_grp_fact(0,18]')
            method.dur_use['age_factors'] = np.append(thisdf.estimate.values[age_ind:], 0)

            if dist in ['lognormal', 'lnorm']:
                method.dur_use['dist'] = 'lognormal'
                method.dur_use['par1'] = thisdf.estimate[thisdf.coef == 'meanlog'].values[0]
                method.dur_use['par2'] = thisdf.estimate[thisdf.coef == 'sdlog'].values[0]
            elif dist in ['gamma']:
                method.dur_use['dist'] = dist
                method.dur_use['par1'] = thisdf.estimate[thisdf.coef == 'shape'].values[0]
                method.dur_use['par2'] = thisdf.estimate[thisdf.coef == 'rate'].values[0]
            elif dist == 'llogis':
                method.dur_use['dist'] = dist
                method.dur_use['par1'] = thisdf.estimate[thisdf.coef == 'shape'].values[0]
                method.dur_use['par2'] = thisdf.estimate[thisdf.coef == 'scale'].values[0]
            elif dist == 'weibull':
                method.dur_use['dist'] = dist
                method.dur_use['par1'] = thisdf.estimate[thisdf.coef == 'shape'].values[0]
                method.dur_use['par2'] = thisdf.estimate[thisdf.coef == 'scale'].values[0]
            elif dist == 'exponential':
                method.dur_use['dist'] = dist
                method.dur_use['par1'] = thisdf.estimate[thisdf.coef == 'rate'].values[0]
                method.dur_use['par2'] = None
            else:
                errormsg = f"Duration of use distribution {dist} not recognized"
                raise ValueError(errormsg)

    return methods


def mcpr():

    mcpr = {}
    cpr_data = pd.read_csv(thisdir / 'data' / 'cpr.csv')
    mcpr['mcpr_years'] = cpr_data['year'].to_numpy()
    mcpr['mcpr_rates'] = cpr_data['cpr'].to_numpy() / 100

    return mcpr


def barriers_region():
    '''
    Returns reasons for nonuse by region
    '''
    reasons_region = pd.read_csv(thisdir / 'ethiopia' / 'subnational' / 'barriers_region.csv')
    reasons_region_dict = {}
    reasons_region_dict['region'] = reasons_region['region'] # Return region names
    reasons_region_dict['barrier'] = reasons_region['barrier'] # Return the reason for nonuse
    reasons_region_dict['perc'] = reasons_region['perc'] # Return retuned the percentage

    return reasons_region_dict


def urban_proportion():
    """Load information about the proportion of people who live in an urban setting"""
    urban_data = pd.read_csv(thisdir / 'data' / 'urban.csv')
    return urban_data["mean"][0]  # Return this value as a float

# %% Make and validate parameters

def make_pars(seed=None, use_subnational=None):
    '''
    Take all parameters and construct into a dictionary
    '''

    # Scalar parameters and filenames
    pars = scalar_pars()
    pars['filenames'] = filenames()

    # Demographics and pregnancy outcome
    pars['age_pyramid'] = age_pyramid()
    pars['age_mortality'] = age_mortality()
    pars['urban_prop'] = urban_proportion()
    pars['maternal_mortality'] = maternal_mortality()
    pars['infant_mortality'] = infant_mortality()
    pars['miscarriage_rates'] = miscarriage()
    pars['stillbirth_rate'] = stillbirth()

    # Fecundity
    pars['age_fecundity'] = female_age_fecundity()
    pars['fecundity_ratio_nullip'] = fecundity_ratio_nullip()
    pars['lactational_amenorrhea'] = lactational_amenorrhea()

    # Pregnancy exposure
    pars['sexual_activity'] = sexual_activity()
    pars['sexual_activity_pp'] = sexual_activity_pp()
    pars['debut_age'] = debut_age()
    pars['exposure_age'] = exposure_age()
    pars['exposure_parity'] = exposure_parity()
    pars['spacing_pref'] = birth_spacing_pref()

    # Contraceptive methods
    pars['barriers'] = barriers()
    pars['mcpr'] = mcpr()

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
