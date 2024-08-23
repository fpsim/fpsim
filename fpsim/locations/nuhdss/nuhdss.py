'''
Set the parameters for FPsim, specifically for NUHDSS.
'''

import numpy as np
import pandas as pd
import sciris as sc
from scipy import interpolate as si
from fpsim import defaults as fpd
from fpsim import utils as fpu

# %% Housekeeping

thisdir = sc.thispath(__file__)  # For loading CSV files


def scalar_pars():
    scalar_pars = {
        'location':             'nuhdss',
        'postpartum_dur':       23, #We used the median #We do  have. Askwhat to calculate!
        'breastfeeding_dur_mu': 11.4261936291137,   # Location parameter of gumbel distribution. Requires children's recode DHS file, see data_processing/breastfeedin_stats.R/  Not available
        'breastfeeding_dur_beta': 7.5435309020483,  # Location parameter of gumbel distribution. Requires children's recode DHS file, see data_processing/breastfeedin_stats.R/   NA
        'abortion_prob':        0.0010235414534,              # From https://bmcpregnancychildbirth.biomedcentral.com/articles/10.1186/s12884-015-0621-1, % of all pregnancies calculated #We calculated from the data
        'twins_prob':           0.0145530145530,              # From https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0025239 #Calculated from the data
        'high_parity_nonuse':   1,                  # TODO: check whether it's correct that this should be different to the other locations
        'mcpr_norm_year':       2015,          #Taken the most recent year in our data     # Year to normalize MCPR trend to 1
       
    }
    return scalar_pars


def data2interp(data, ages, normalize=False):
    ''' Convert unevenly spaced data into an even spline interpolation '''
    model = si.interp1d(data[0], data[1])
    interp = model(ages)
    if normalize:
        interp = np.minimum(1, np.maximum(0, interp))
    return interp


# TODO- these need to be changed for Kenya calibration and commented with their data source
def filenames():
    ''' Data files for use with calibration, etc -- not needed for running a sim '''
    files = {}
    files['base'] = sc.thisdir(aspath=True) / 'data'
    files['basic_dhs'] = 'basic_dhs.yaml' # From World Bank https://data.worldbank.org/indicator/SH.STA.MMRT?locations=KE
    files['popsize'] = 'popsize.csv' # Downloaded from World Bank: https://data.worldbank.org/indicator/SP.POP.TOTL?locations=KE  #How do we include rounds #Look at subcounty doata
    files['mcpr'] = 'cpr.csv'  # From UN Population Division Data Portal, married women 1970-1986, all women 1990-2030
    files['tfr'] = 'tfr.csv'   # From World Bank https://data.worldbank.org/indicator/SP.DYN.TFRT.IN?locations=KE
    files['asfr'] = 'asfr.csv' # From UN World Population Prospects 2022: https://population.un.org/wpp/Download/Standard/Fertility/
    files['ageparity'] = 'ageparity.csv' # Choose from either DHS 2014 or PMA 2022
    files['spacing'] = 'birth_spacing_dhs.csv'
    files['methods'] = 'mix.csv'
    files['afb'] = 'afb.table.csv'
    files['use'] = 'use.csv'
    return files


# %% Demographics and pregnancy outcome

def age_pyramid():
    '''
    Starting age bin, male population, female population
    Data are from World Population Prospects
    https://population.un.org/wpp/Download/Standard/Population/
     '''
    #Use this as its from Kenya. Change it in future from the NUHDSS combining both male and female data
    pyramid = np.array([[0, 801895, 800503],  # Kenya 1960
                        [5, 620524, 625424],
                        [10, 463547, 464020],
                        [15, 333241, 331921],
                        [20, 307544, 309057],
                        [25, 292141, 287621],
                        [30, 247826, 236200],
                        [35, 208416, 190234],
                        [40, 177914, 162057],
                        [45, 156771, 138943],
                        [50, 135912, 123979],
                        [55, 108653, 111939],
                        [60, 85407, 94582],
                        [65, 61664, 71912],
                        [70, 40797, 49512],
                        [75, 22023, 29298],
                        [80, 11025, 17580],
                        ], dtype=float)

    return pyramid


def urban_proportion():
    """Load information about the proportion of people who live in an urban setting""" #100% urban mean 1, se 0
    urban_data = pd.read_csv(thisdir / 'data' / 'urban.csv')
    return urban_data["mean"][0]  # Return this value as a float


def age_mortality():
    '''
    Age-dependent mortality rates taken from UN World Population Prospects 2022.  From probability of dying each year.
    https://population.un.org/wpp/
    Used CSV WPP2022_Life_Table_Complete_Medium_Female_1950-2021, Kenya, 2010
    Used CSV WPP2022_Life_Table_Complete_Medium_Male_1950-2021, Kenya, 2010
    Mortality rate trend from crude death rate per 1000 people, also from UN Data Portal, 1950-2030:
    https://population.un.org/dataportal/data/indicators/59/locations/404/start/1950/end/2030/table/pivotbylocation
    Projections go out until 2030, but the csv file can be manually adjusted to remove any projections and stop at your desired year
    #Leave as it is.....
    '''
    data_year = 2010
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
    https://data.worldbank.org/indicator/SH.STA.MMRT?locations=KE
    '''

    data = np.array([
        [2000, 708],
        [2001, 702],
        [2002, 692],
        [2003, 678],
        [2004, 653],
        [2005, 618],
        [2006, 583],
        [2007, 545],
        [2008, 513],
        [2009, 472],
        [2010, 432],
        [2011, 398],
        [2012, 373],
        [2013, 364],
        [2014, 358],
        [2015, 353],
        [2016, 346],
        [2017, 342],

    ])

    maternal_mortality = {}
    maternal_mortality['year'] = data[:, 0]
    maternal_mortality['probs'] = data[:, 1] / 100000  # ratio per 100,000 live births
    # maternal_mortality['ages'] = np.array([16, 17,   19, 22,   25, 50])
    # maternal_mortality['age_probs'] = np.array([2.28, 1.63, 1.3, 1.12, 1.0, 1.0]) #need to be added

    return maternal_mortality


def infant_mortality():
    '''
    From World Bank indicators for infant mortality (< 1 year) for Kenya, per 1000 live births
    From API_SP.DYN.IMRT.IN_DS2_en_excel_v2_1495452.numbers
    Adolescent increased risk of infant mortality gradient taken
    from Noori et al for Sub-Saharan African from 2014-2018.  Odds ratios with age 23-25 as reference group:
    https://www.medrxiv.org/content/10.1101/2021.06.10.21258227v1
    '''

    data = np.array([
        [1960, 118.1],
        [1961, 113.7],
        [1962, 109.8],
        [1963, 106.5],
        [1964, 103.8],
        [1965, 101.6],
        [1966, 99.8],
        [1967, 98.0],
        [1968, 96.3],
        [1969, 94.5],
        [1970, 92.6],
        [1971, 90.8],
        [1972, 88.8],
        [1973, 86.7],
        [1974, 84.5],
        [1975, 82.2],
        [1976, 79.8],
        [1977, 77.4],
        [1978, 75.0],
        [1979, 72.7],
        [1980, 70.5],
        [1981, 68.5],
        [1982, 66.7],
        [1983, 65.1],
        [1984, 63.8],
        [1985, 62.9],
        [1986, 62.4],
        [1987, 62.4],
        [1988, 62.8],
        [1989, 63.7],
        [1990, 64.8],
        [1991, 66.1],
        [1992, 67.2],
        [1993, 67.9],
        [1994, 68.0],
        [1995, 67.6],
        [1996, 66.5],
        [1997, 65.1],
        [1998, 63.5],
        [1999, 61.7],
        [2000, 59.7],
        [2001, 57.6],
        [2002, 55.4],
        [2003, 53.1],
        [2004, 50.7],
        [2005, 48.1],
        [2006, 45.8],
        [2007, 43.8],
        [2008, 41.4],
        [2009, 40.3],
        [2010, 39.4],
        [2011, 38.6],
        [2012, 38.2],
        [2013, 37.5],
        [2014, 36.5],
        [2015, 35.3],
        [2016, 34.5],
        [2017, 33.9],
        [2018, 32.8],
        [2019, 31.9]
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
    miscarriage_rates = np.array([[0, 5, 10, 15, 20, 25, 30, 35, 40, 50],
                                  [1, 1, 0.00358938, 0.0096776, 0.01051225, 0.01264532, 0.01018676, 0.01417004, 0.00826446, 0.00826446]])
    miscarriage_interp = data2interp(miscarriage_rates, fpd.spline_preg_ages)
    return miscarriage_interp


def stillbirth():
    '''
    From Report of the UN Inter-agency Group for Child Mortality Estimation, 2020
    https://childmortality.org/wp-content/uploads/2020/10/UN-IGME-2020-Stillbirth-Report.pdf

    Age adjustments come from an extension of Noori et al., which were conducted June 2022.
    '''
    #To calculate from our data  (Per Year of pregnancy termination ie end of pregnancy)

    data = np.array([
        [2002, 7.95228628230616],
        [2003, 8.64055299539171],
        [2004, 11.8216012896292],
        [2005, 10.4046242774566],
        [2006, 6.8241469816273],
        [2007, 11.2359550561798],
        [2008, 4.70219435736677],
        [2009, 6.68576886341929],
        [2010, 6.34765625],
        [2011, 12.8388017118402],
        [2012, 12.0988953182536],
        [2013, 14.8632580261593],
        [2014, 14.8720999405116],
        [2015, 6.43086816720257],
        [2016, 4.21686746987952],
        [2017, 4.09556313993174],
        [2018, 5.06512301013025],
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
    From DHS Kenya 2014 calendar data

    #Calculate consult
    '''
    data = np.array([
        [0, 0.9557236],
        [1, 0.8889493],
        [2, 0.7040052],
        [3, 0.5332317],
        [4, 0.4115276],
        [5, 0.2668908],
        [6, 0.1364079],
        [7, 0.0571638],
        [8, 0.0025502],
        [9, 0.0259570],
        [10, 0.0072750],
        [11, 0.0046938],
    ])

    lactational_amenorrhea = {}
    lactational_amenorrhea['month'] = data[:, 0]
    lactational_amenorrhea['rate'] = data[:, 1]

    return lactational_amenorrhea


# %% Pregnancy exposure

def sexual_activity(): 

    #Calculate from sex debut (yes), Married and has sex partner
    '''
    Returns a linear interpolation of rates of female sexual activity, defined as
    percentage women who have had sex within the last four weeks.
    From STAT Compiler DHS https://www.statcompiler.com/en/
    Using indicator "Timing of sexual intercourse"
    Includes women who have had sex "within the last four weeks"
    Excludes women who answer "never had sex", probabilities are only applied to agents who have sexually debuted
    Data taken from 2018 DHS, no trend over years for now
    Onset of sexual activity probabilities assumed to be linear from age 10 to first data point at age 15
    '''

    sexually_active = np.array([[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
                                [0, 0, 0, 31.4, 55.0, 64.4, 69.6, 65.3, 60.7, 57.4, 57.4]])

    sexually_active[1] /= 100  # Convert from percent to rate per woman
    activity_ages = sexually_active[0]
    activity_interp_model = si.interp1d(x=activity_ages, y=sexually_active[1])
    activity_interp = activity_interp_model(fpd.spline_preg_ages)  # Evaluate interpolation along resolution of ages

    return activity_interp


def sexual_activity_pp():
    '''
    #Use the data provided here
    Returns an array of monthly likelihood of having resumed sexual activity within 0-35 months postpartum
    Uses 2014 Kenya DHS individual recode (postpartum (v222), months since last birth, and sexual activity within 30 days.
    Data is weighted.
    Limited to 23 months postpartum (can use any limit you want 0-23 max)
    Postpartum month 0 refers to the first month after delivery
    TODO-- Add code for processing this for other countries to data_processing
    '''

    postpartum_sex = np.array([
        [0, 0.08453],
        [1, 0.08870],
        [2, 0.40634],
        [3, 0.58030],
        [4, 0.52688],
        [5, 0.60641],
        [6, 0.58103],
        [7, 0.72973],
        [8, 0.62647],
        [9, 0.73497],
        [10, 0.60254],
        [11, 0.75723],
        [12, 0.73159],
        [13, 0.68409],
        [14, 0.74925],
        [15, 0.74059],
        [16, 0.70051],
        [17, 0.78479],
        [18, 0.74965],
        [19, 0.79351],
        [20, 0.77338],
        [21, 0.70340],
        [22, 0.72395],
        [23, 0.72202]
    ])



    postpartum_activity = {}
    postpartum_activity['month'] = postpartum_sex[:, 0]
    postpartum_activity['percent_active'] = postpartum_sex[:, 1]

    return postpartum_activity


def debut_age():
    '''
    Returns an array of weighted probabilities of sexual debut by a certain age 10-45.
    Data taken from DHS variable v531 (imputed age of sexual debut, imputed with data from age at first union)
    Use sexual_debut_age_probs.py under locations/data_processing to output for other DHS countries
    

    sexual_debut = np.array([
        [10.0, 0.008404629256166524],
        [11.0, 0.006795048697926663],
        [12.0, 0.026330525753311643],
        [13.0, 0.04440278185223372],
        [14.0, 0.08283157906888061],
        [15.0, 0.14377365580688461],
        [16.0, 0.13271744734209995],
        [17.0, 0.11915611658325072],
        [18.0, 0.13735481818469894],
        [19.0, 0.0841039265081519],
        [20.0, 0.07725867074164659],
        [21.0, 0.03982337306065369],
        [22.0, 0.031195559243867545],
        [23.0, 0.020750304422300126],
        [24.0, 0.014468030815585422],
        [25.0, 0.010870195645684769],
        [26.0, 0.007574195696769944],
        [27.0, 0.0034378402773621282],
        [28.0, 0.0031344552061394622],
        [29.0, 0.0018168079578966389],
        [30.0, 0.001385356426809007],
        [31.0, 0.0004912818135032509],
        [32.0, 0.00045904179812542576],
        [33.0, 0.0005049625590548578],
        [34.0, 0.000165858204720886],
        [35.0, 0.00019259487032758347],
        [36.0, 0.0002126920535675137],
        [37.0, 8.84428869703282e-05],
        [38.0, 5.07209448615522e-05],
        [39.0, 6.555458199225806e-05],
        [41.0, 0.00013980442816424654],
        [44.0, 4.372731039149624e-05]])
        '''
    sexual_debut = pd.read_csv(thisdir / 'data' / 'sex_debut.csv')
    debut_age = {}
    debut_age['ages'] = sexual_debut.Age_at_first_sex.values
    debut_age['probs'] = sexual_debut.prob.values

    return debut_age


def exposure_age():
    '''
    #Ask
    Returns an array of experimental factors to be applied to account for
    residual exposure to either pregnancy or live birth by age.  Exposure to pregnancy will
    increase factor number and residual likelihood of avoiding live birth (mostly abortion,
    also miscarriage), will decrease factor number
    '''
    exposure_correction_age = np.array([[0, 5, 10, 12.5, 15, 18, 20, 25, 30, 35, 40, 45, 50],
                                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    exposure_age_interp = data2interp(exposure_correction_age, fpd.spline_preg_ages)

    return exposure_age_interp


def exposure_parity(): #Consult
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
    '''
    Names, indices, modern/traditional flag, and efficacies of contraceptive methods -- see also parameters.py
    Efficacy from Guttmacher, fp_prerelease/docs/gates_review/contraceptive-failure-rates-in-developing-world_1.pdf
    BTL failure rate from general published data
    Pooled efficacy rates for all women in this study: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4970461/
    '''

    # Define method data
    data = {  # Index, modern, long-term efficacy
        'None': [0, False, 0.000],
        'Withdrawal': [1, False, 0.866],
        'Other traditional': [2, False, 0.861],
        # 1/2 periodic abstinence, 1/2 other traditional approx.  Using rate from periodic abstinence
        'Condoms': [3, True, 0.946],
        'Pill': [4, True, 0.945],
        'Injectables': [5, True, 0.983],
        'Implants': [6, True, 0.994],
        'IUDs': [7, True, 0.986],
        'BTL': [8, True, 0.995],
       # 'Diaphragm': [9, True, 0.81],
        #'Sterilization': [10, True, 0.99],
        #'LAM': [11, True, 0.98],
       # 'Rhythm': [12, True, 0.75],
        'Other modern': [9, True, 0.880],
        # SDM makes up about 1/2 of this, perfect use is 95% and typical is 88%.  EC also included here, efficacy around 85% https : //www.aafp.org/afp/2004/0815/p707.html
    }

    keys = data.keys()
    methods = {}
    methods['map'] = {k: data[k][0] for k in keys}
    methods['modern'] = {k: data[k][1] for k in keys}
    methods['eff'] = {k: data[k][2] for k in keys}

    # Age bins for different method switching matrices -- duplicated in defaults.py
    methods['age_map'] = {
        '<18': [0, 18],
        '18-20': [18, 20],
        '21-25': [20, 25],
        '26-35': [25, 35],
        '>35': [35, fpd.max_age + 1],  # +1 since we're using < rather than <=
    }

    # Data on trend in CPR over time in from Kenya, in %.
    # Taken from UN Population Division Data Portal, married women 1970-1986, all women 1990-2030
    # https://population.un.org/dataportal/data/indicators/1/locations/404/start/1950/end/2040/table/pivotbylocation
    # Projections go out until 2030, but the csv file can be manually adjusted to remove any projections and stop at your desired year
    cpr_data = pd.read_csv(thisdir / 'data' / 'cpr.csv')
    methods['mcpr_years'] = cpr_data['year'].to_numpy()
    methods['mcpr_rates'] = cpr_data['cpr'].to_numpy() / 100  # convert from percent to rate

    return methods



'''
For reference
def method_probs_senegal():
    
    It does leave Senegal matrices in place in the Kenya file for now. 
    We may want to test with these as we work through scenarios and calibration. 
    
    Define "raw" (un-normalized, un-trended) matrices to give transitional probabilities
    from 2018 DHS Senegal contraceptive calendar data.

    Probabilities in this function are annual probabilities of initiating (top row), discontinuing (first column),
    continuing (diagonal), or switching methods (all other entries).

    Probabilities at postpartum month 1 are 1 month transitional probabilities
    for starting a method after delivery.

    Probabilities at postpartum month 6 are 5 month transitional probabilities
    for starting or changing methods over the first 6 months postpartum.

    Data from Senegal DHS contraceptive calendars, 2017 and 2018 combined
    

    raw = {

        # Main switching matrix: all non-postpartum women
        'annual': {
            '<18': np.array([
                [0.9953, 0., 0.0002, 0.0012, 0.0002, 0.0017, 0.0014, 0.0001, 0., 0.],
                [0., 1.0000, 0., 0., 0., 0., 0., 0., 0., 0.],
                [0.0525, 0., 0.9475, 0., 0., 0., 0., 0., 0., 0.],
                [0.307, 0., 0., 0.693, 0., 0., 0., 0., 0., 0.],
                [0.5358, 0., 0., 0., 0.3957, 0.0685, 0., 0., 0., 0.],
                [0.3779, 0., 0., 0., 0.0358, 0.5647, 0.0216, 0., 0., 0.],
                [0.2003, 0., 0., 0., 0., 0., 0.7997, 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 1.0000, 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 1.0000, 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.0000]]),
            '18-20': np.array([
                [0.9774, 0., 0.0014, 0.0027, 0.0027, 0.0104, 0.0048, 0.0003, 0., 0.0003],
                [0., 1.0000, 0., 0., 0., 0., 0., 0., 0., 0.],
                [0.3216, 0., 0.6784, 0., 0., 0., 0., 0., 0., 0.],
                [0.182, 0., 0., 0.818, 0., 0., 0., 0., 0., 0.],
                [0.4549, 0., 0., 0., 0.4754, 0.0463, 0.0234, 0., 0., 0.],
                [0.4389, 0., 0.0049, 0.0099, 0.0196, 0.5218, 0.0049, 0., 0., 0.],
                [0.17, 0., 0., 0., 0., 0.0196, 0.8103, 0., 0., 0.],
                [0.1607, 0., 0., 0., 0., 0., 0., 0.8393, 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 1.0000, 0.],
                [0.4773, 0., 0., 0.4773, 0., 0., 0., 0., 0., 0.0453]]),
            '21-25': np.array([
                [0.9581, 0.0001, 0.0011, 0.0024, 0.0081, 0.0184, 0.0108, 0.0006, 0., 0.0004],
                [0.4472, 0.5528, 0., 0., 0., 0., 0., 0., 0., 0.],
                [0.2376, 0., 0.7624, 0., 0., 0., 0., 0., 0., 0.],
                [0.1896, 0., 0.0094, 0.754, 0.0094, 0., 0.0188, 0., 0., 0.0188],
                [0.3715, 0.003, 0.003, 0., 0.5703, 0.0435, 0.0088, 0., 0., 0.],
                [0.3777, 0., 0.0036, 0.0036, 0.0258, 0.5835, 0.0036, 0.0024, 0., 0.],
                [0.137, 0., 0., 0.003, 0.0045, 0.0045, 0.848, 0.003, 0., 0.],
                [0.1079, 0., 0., 0., 0.0445, 0., 0.0225, 0.8251, 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 1.0000, 0.],
                [0.3342, 0., 0., 0.1826, 0., 0., 0., 0., 0., 0.4831]]),
            '26-35': np.array([
                [0.9462, 0.0001, 0.0018, 0.0013, 0.0124, 0.0209, 0.0139, 0.003, 0.0001, 0.0002],
                [0.0939, 0.8581, 0., 0., 0., 0.048, 0., 0., 0., 0.],
                [0.1061, 0., 0.8762, 0.0051, 0.0025, 0.0025, 0.0051, 0.0025, 0., 0.],
                [0.1549, 0., 0., 0.8077, 0.0042, 0.0125, 0.0083, 0.0083, 0., 0.0042],
                [0.3031, 0.0016, 0.0021, 0.0021, 0.6589, 0.0211, 0.0053, 0.0053, 0., 0.0005],
                [0.2746, 0., 0.0028, 0.002, 0.0173, 0.691, 0.0073, 0.0048, 0., 0.0003],
                [0.1115, 0.0003, 0.0009, 0.0003, 0.0059, 0.0068, 0.8714, 0.0025, 0.0003, 0.],
                [0.0775, 0., 0.0015, 0., 0.0058, 0.0044, 0.0044, 0.905, 0., 0.0015],
                [0., 0., 0., 0., 0., 0., 0., 0., 1.0000, 0.],
                [0.1581, 0., 0.0121, 0., 0., 0., 0., 0., 0., 0.8297]]),
            '>35': np.array([
                [0.9462, 0.0001, 0.0018, 0.0013, 0.0124, 0.0209, 0.0139, 0.003, 0.0001, 0.0002],
                [0.0939, 0.8581, 0., 0., 0., 0.048, 0., 0., 0., 0.],
                [0.1061, 0., 0.8762, 0.0051, 0.0025, 0.0025, 0.0051, 0.0025, 0., 0.],
                [0.1549, 0., 0., 0.8077, 0.0042, 0.0125, 0.0083, 0.0083, 0., 0.0042],
                [0.3031, 0.0016, 0.0021, 0.0021, 0.6589, 0.0211, 0.0053, 0.0053, 0., 0.0005],
                [0.2746, 0., 0.0028, 0.002, 0.0173, 0.691, 0.0073, 0.0048, 0., 0.0003],
                [0.1115, 0.0003, 0.0009, 0.0003, 0.0059, 0.0068, 0.8714, 0.0025, 0.0003, 0.],
                [0.0775, 0., 0.0015, 0., 0.0058, 0.0044, 0.0044, 0.905, 0., 0.0015],
                [0., 0., 0., 0., 0., 0., 0., 0., 1.0000, 0.],
                [0.1581, 0., 0.0121, 0., 0., 0., 0., 0., 0., 0.8297]])
        },

        # Postpartum switching matrix, 1 to 6 months
        'pp1to6': {
            '<18': np.array([
                [0.9014, 0., 0.0063, 0.001, 0.0126, 0.051, 0.0272, 0.0005, 0., 0.],
                [0., 0.5, 0., 0., 0., 0., 0.5, 0., 0., 0.],
                [0., 0., 1.0000, 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 1.0000, 0., 0., 0., 0., 0., 0.],
                [0.4, 0., 0., 0., 0.6, 0., 0., 0., 0., 0.],
                [0.0714, 0., 0., 0., 0., 0.9286, 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 1.0000, 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 1.0000, 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 1.0000, 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.0000]]),
            '18-20': np.array([
                [0.8775, 0.0007, 0.0026, 0.0033, 0.0191, 0.0586, 0.0329, 0.0046, 0., 0.0007],
                [0., 1.0000, 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 1.0000, 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 1.0000, 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.75, 0.25, 0., 0., 0., 0.],
                [0.0278, 0., 0., 0., 0., 0.9722, 0., 0., 0., 0.],
                [0.0312, 0., 0., 0., 0., 0., 0.9688, 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 1.0000, 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 1.0000, 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.0000]]),
            '21-25': np.array([
                [0.8538, 0.0004, 0.0055, 0.0037, 0.0279, 0.0721, 0.0343, 0.0022, 0., 0.],
                [0., 1.0000, 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0.9583, 0., 0., 0.0417, 0., 0., 0., 0.],
                [0., 0., 0., 0.5, 0.25, 0.25, 0., 0., 0., 0.],
                [0.0244, 0., 0., 0., 0.9512, 0.0244, 0., 0., 0., 0.],
                [0.0672, 0., 0., 0., 0., 0.9328, 0., 0., 0., 0.],
                [0.0247, 0., 0., 0., 0., 0.0123, 0.963, 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 1.0000, 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 1.0000, 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.0000]]),
            '26-35': np.array([
                [0.8433, 0.0008, 0.0065, 0.004, 0.029, 0.0692, 0.039, 0.0071, 0.0001, 0.001],
                [0., 0.5, 0., 0., 0., 0., 0.5, 0., 0., 0.],
                [0.027, 0., 0.9189, 0., 0., 0.027, 0.027, 0., 0., 0.],
                [0.1667, 0., 0., 0.6667, 0., 0., 0.1667, 0., 0., 0.],
                [0.0673, 0., 0., 0., 0.8654, 0.0288, 0.0385, 0., 0., 0.],
                [0.0272, 0., 0.0039, 0., 0.0078, 0.9533, 0.0078, 0., 0., 0.],
                [0.0109, 0., 0., 0., 0.0036, 0., 0.9855, 0., 0., 0.],
                [0.0256, 0., 0., 0., 0., 0.0256, 0., 0.9487, 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 1.0000, 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.0000]]),
            '>35': np.array([
                [0.8433, 0.0008, 0.0065, 0.004, 0.029, 0.0692, 0.039, 0.0071, 0.0001, 0.001],
                [0., 0.5, 0., 0., 0., 0., 0.5, 0., 0., 0.],
                [0.027, 0., 0.9189, 0., 0., 0.027, 0.027, 0., 0., 0.],
                [0.1667, 0., 0., 0.6667, 0., 0., 0.1667, 0., 0., 0.],
                [0.0673, 0., 0., 0., 0.8654, 0.0288, 0.0385, 0., 0., 0.],
                [0.0272, 0., 0.0039, 0., 0.0078, 0.9533, 0.0078, 0., 0., 0.],
                [0.0109, 0., 0., 0., 0.0036, 0., 0.9855, 0., 0., 0.],
                [0.0256, 0., 0., 0., 0., 0.0256, 0., 0.9487, 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 1.0000, 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.0000]])
        },

        # Postpartum initiation vectors, 0 to 1 month
        'pp0to1': {
            '<18': np.array([0.9607, 0.0009, 0.0017, 0.0009, 0.0021, 0.0128, 0.0205, 0.0004, 0., 0.]),
            '18-20': np.array([0.9525, 0.0006, 0.0017, 0.0006, 0.0028, 0.0215, 0.0198, 0.0006, 0., 0.]),
            '21-25': np.array([0.9379, 0., 0.0053, 0.0009, 0.0083, 0.0285, 0.0177, 0.0013, 0., 0.]),
            '26-35': np.array([0.9254, 0.0002, 0.0036, 0.0007, 0.0102, 0.0265, 0.0268, 0.004, 0.0022, 0.0004]),
            '>35': np.array([0.9254, 0.0002, 0.0036, 0.0007, 0.0102, 0.0265, 0.0268, 0.004, 0.0022, 0.0004]),
        }
    }
    return raw
    '''

def method_probs():
    '''
    Define "raw" (un-normalized, un-trended) matrices to give transitional probabilities
    from PMA Kenya contraceptive calendar data.

    Probabilities in this function are annual probabilities of initiating (top row), discontinuing (first column),
    continuing (diagonal), or switching methods (all other entries).

    Probabilities at postpartum month 1 are 1 month transitional probabilities
    for starting a method after delivery.

    Probabilities at postpartum month 6 are 5 month transitional probabilities
    for starting or changing methods over the first 6 months postpartum.

    Data from Kenya PMA contraceptive calendars, 2019-2020
    Processed from matrices_kenya_pma_2019_20.csv using process_matrices.py
    '''

    raw = {

        # Main switching matrix: all non-postpartum women
        'annual': {
            '<18': np.array([
                [0.958729287364908,0.00324869118999745,0.00309214156807592,0.0144334920054295,0.00384889000870717,0.0128951049797922,0.00375239288308918,0,0,0],
                [0,1,0,0,0,0,0,0,0,0],
                [0.322387593493318,0.000530487467058028,0.670628770320453,0.00289056432547088,0.00066487814312008,0.0022474530887343,0.000650253161844578,0,0,0],
                [0.722941506172913,0.00137621666546124,0.00138865841639427,0.265123074912138,0.00171281603795851,0.00578325257225382,0.00167447522288061,0,0,0],
                [0.278636555720339,0.000453457551751284,0.000461663668107394,0.0024756062478374,0.715494327962749,0.00192225454396113,0.000556134305255896,0,0,0],
                [0.316025258380343,0.000519164584203876,0.000528285386181899,0.00282968002500736,0.000650736573749687,0.678810449636611,0.000636425413903754,0,0,0],
                [0.292075743179317,0.000476915542552866,0.000485456933358768,0.00260216103218453,0.000597949186599755,0.00202133263345711,0.70174044149253,0,0,0],
                [0,0,0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,0,0,1,0],
                [0,0,0,0,0,0,0,0,0,1]]),
            '18-20': np.array([
                [0.907934887456798,0.000755532866520863,0.000310292400005308,0.0170459977370897,0.0098151247244949,0.0616153199380119,0.00252284487707888,0,0,0],
                [0.0896025279046224,0.906069277244235,9.16173874749515e-06,0.000818461404407199,0.000457827019366535,0.00293024601411494,0.000112498674507398,0,0,0],
                [0.442300409113862,0.000120161440610459,0.534311878189141,0.00440558033225827,0.00246950473356716,0.015783709678019,0.000608756512540551,0,0,0],
                [0.325296744363331,8.39634186324787e-05,3.60814516306625e-05,0.661167249357349,0.00175576758345548,0.0112279133800738,0.000432280445526518,0,0,0],
                [0.211249858950407,5.22368559625942e-05,2.24671369900669e-05,0.00198020585836461,0.77933166826031,0.00709095637834108,0.000272606559623795,0,0,0],
                [0.262485323449647,0.0206347139179738,0.007680855285443,0.00249419270208828,0.00139649876038998,0.704964769013903,0.000343646870554426,0,0,0],
                [0,0,0,0,0,0,1,0,0,0],
                [0,0,0,0,0,0,0,1,0,0 ],
                [0,0,0,0,0,0,0,0,1,0],
                [0,0,0,0,0,0,0,0,0,1]]),
            '21-25': np.array([
                [0.863465086802549,0.000103923786407093,0.00818222283658148,0.00265775804737037,0.0156233709637681,0.100267258370611,0.00970037919271196,0,0,0],
                [0.161065087198627,0.00121382005952542,0.000617265051612585,0.109643274336321,0.0123718537317179,0.704826851839696,0.010261847782501,0,0,0],
                [0.00978318789002996,0.000100731467673275,0.881568489936328,0.000131696543328328,0.00095500819432268,0.106705761462745,0.000755124505572637,0,0,0],
                [0.256363114646434,0.00295888079480554,0.00115079759120401,0.70160294766116,0.00245386516587306,0.0340546396889093,0.00141575445161341,0,0,0],
                [0.336571349770617,3.78657102593889e-05,0.00155585113782625,0.000541754150544982,0.594478872982046,0.0401807412615186,0.0266335649871887,0,0,0],
                [0.1659816355524,0.000869065631832306,0.000725204608623943,0.00235188025058452,0.0156314085368555,0.801042154596249,0.0133986508234547,0,0,0],
                [0.0298250055832434,1.42316144698319e-06,0.000126751820401564,4.20809153599464e-05,0.000253907523776732,0.00157934654350837,0.968171484452263,0,0,0],
                [0,0,0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,0,0,1,0],
                [0,0,0,0,0,0,0,0,0,1]]),
            '26-35': np.array([
                [0.846525320704276,0.000177378555225855,0.00137853904652593,0.00873277709104264,0.00904282394020776,0.0998129433197505,0.0327688445640532,0.00132379805285494,0.000102856301705054,0.000134718424358144],
                [0.221315910924807,0.75999731668268,0.000168645241048074,0.00105428304422766,0.0010788030729217,0.0123207229136821,0.00388581531661182,0.000161005946327919,7.62219101889231e-06,9.87466667557743e-06 ],
                [0.116142562198115,9.88314796886902e-06,0.766785294761073,0.000809141565654651,0.00159357556704469,0.111518457557909,0.00293726208176848,8.1781979488651e-05,0.000114518286231512,7.52285474616802e-06],
                [0.101551365947752,5.85802989471756e-06,7.5687906476313e-05,0.860515693235418,0.000483945135512671,0.0055303711505502,0.00174299613029065,7.22517005593082e-05,3.3827215210951e-06,0.0300184480420256],
                [0.0706442802776139,0.000393255199627473,0.0122944200132138,0.000512100509544559,0.766704663190217,0.0781574697147834,0.0711601504551628,4.8820013442272e-05,8.01078704526028e-05,4.7327559404868e-06],
                [0.14822084041827,9.91469512983706e-05,0.000235584685107878,0.00525683600872368,0.0177737931420622,0.807183885491734,0.0191439681578475,0.000106512923639564,0.00190103212239024,7.84000989268449e-05],
                [0.0442780542482289,0.0106093339131479,3.22586809292117e-05,0.000217188160016457,0.000270988883091367,0.00929597879671564,0.935255244140265,3.05129077491948e-05,8.45799933575981e-06,1.98227052018309e-06],
                [0.0888032465754913,8.12973582668051e-06,0.000230041521882835,0.000411880000909113,0.0234356725959714,0.00578441484768781,0.00241388709383154,0.878905402801961,3.52933666607764e-06,3.79548977044643e-06],
                [0,0,0,0,0,0,0,0,1,0],
                [0,0,0,0,0,0,0,0,0,1]]),
            '>35': np.array([
                [0.932815867236076,7.70827382809539e-06,0.00715020371657686,0.00542313676708199,0.0170322487958652,0.0227685122392103,0.00937788678563001,0,0.00542443618573064,0],
                [0.889513359005766,0.0687234429638204,0.00439617205822415,0.00339094268835794,0.0106045471449342,0.0142439740411192,0.00578911985056123,0,0.00333844224721508,0],
                [0,0,1,0,0,0,0,0,0,0],
                [0.139192237626711,3.90898443399341e-07,0.000487819570251894,0.856531090233107,0.00118271893555453,0.0015916985325708,0.000643444312027906,0,0.000370599891332359,0],
                [0.0741031007856288,2.04260137063122e-07,0.000256162219056281,0.000198985898326513,0.924072988472624,0.000836043348455075,0.000337903883085203,0,0.000194611132686087,0 ],
                [0.0617111355957674,0.000470857675731716,0.00955481002648199,0.0116657108302525,0.00932704204311633,0.901491733693755,0.000279592881886988,0,0.00549911725300783,0],
                [0,0,0,0,0,0,1,0,0,0],
                [0.0854849961261186,2.3638418025458e-07,0.000296200062868986,0.000230077353312114,0.000718248744249048,0.000966673449130586,0.000390713814971124,0.911687826024887,0.000225028040282834,0],
                [0,0,0,0,0,0,0,0,1,0],
                [0,0,0,0,0,0,0,0,0,1]])
        },


        # Postpartum switching matrix, 1 to 6 months
        'pp1to6': {
            '<18': np.array([
                [0.653508054004204,0,0,0,0.0178519437987631,0.297001696036839,0.031638306160194,0,0,0],
                [0,1,0,0,0,0,0,0,0,0],
                [0,0,1,0,0,0,0,0,0,0],
                [0,0,0,1,0,0,0,0,0,0],
                [0,0,0,0,1,0,0,0,0,0],
                [0,0,0,0,0,1,0,0,0,0],
                [0,0,0,0,0,0,1,0,0,0],
                [0,0,0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,0,0,1,0],
                [0,0,0,0,0,0,0,0,0,1]]),
            '18-20': np.array([
                [0.593508634912798,0.00668776932733301,0,0,0.029008722352493,0.279592496762608,0.0912023766447683,0,0,0],
                [0,1,0,0,0,0,0,0,0,0],
                [0,0,1,0,0,0,0,0,0,0],
                [0,0,0,1,0,0,0,0,0,0],
                [0,0,0,0,1,0,0,0,0,0],
                [0,0,0,0,0,1,0,0,0,0],
                [0,0,0,0,0,0,1,0,0,0],
                [0,0,0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,0,0,1,0],
                [0,0,0,0,0,0,0,0,0,1]]),
            '21-25': np.array([
                [0.672012512089745,0,0.00183222147778614,0.00432939112932932,0.0445485047706886,0.24717217137509,0.0301051991573608,0,0,0],
                [0,1,0,0,0,0,0,0,0,0],
                [0,0,0.549061385936689,0,0,0.450938614063311,0,0,0,0],
                [0,0,0,1,0,0,0,0,0,0],
                [0,0,0,0,1,0,0,0,0,0],
                [0,0,0,0,0.115413172849361,0.786502932545886,0.098083894604753,0,0,0],
                [0,0,0,0,0,0,1,0,0,0],
                [0,0,0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,0,0,1,0],
                [0,0,0,0,0,0,0,0,0,1]]),
            '26-35': np.array([
                [0.698179575142731,0.00416188189204647,0,0.0185189429975234,0.0482600508253817,0.179030615299206,0.0329459574339988,0,0.00191802367914793,0.0169849527299649],
                [0,1,0,0,0,0,0,0,0,0],
                [0,0,1,0,0,0,0,0,0,0],
                [0,0,0,1,0,0,0,0,0,0],
                [0,0,0,0,0.597763288554891,0.402236711445109,0,0,0,0],
                [0.0689067952792333,0,0,0,0,0.876974476531936,0.0541187281888307,0,0,0],
                [0,0,0,0,0,0,1,0,0,0],
                [0,0,0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,0,0,1,0],
                [0,0,0,0,0,0,0,0,0,1]]),
            '>35': np.array([
               [0.813797986351526,0,0.0137708858022805,0,0,0.143505910270293,0,0.0258483697204492,0.00307684785545134,0],
                [0,1,0,0,0,0,0,0,0,0],
                [0,0,1,0,0,0,0,0,0,0],
                [0,0,0,1,0,0,0,0,0,0],
                [0,0,0,0,1,0,0,0,0,0],
                [0,0,0,0,0,1,0,0,0,0],
                [0,0,0,0,0,0,1,0,0,0],
                [0,0,0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,0,0,1,0],
                [0,0,0,0,0,0,0,0,0,1]])
        },

        # Postpartum initiation vectors, 0 to 1 month
        'pp0to1': {
            '<18': np.array([0.884444729389054,0,0,0,0.0290461560346503,0.0711529517614218,0.0153561628148743,0,0,0]),
            '18-20': np.array([0.936948801298604,0,0,0,0.00706678217945454,0.0493202071999504,0,0.00666420932199142,0,0]),
            '21-25': np.array([0.842695299342924,0,0.00816374556710577,0.000939092022713957,0.0226149324403939,0.0844226470639215,0.0386270651886316,0.00253721837430921,0,0]),
            '26-35': np.array([0.897670627668072,0,0.00331648113376799,0.0126807618859443,0.0201141031582065,0.0446924052159183,0,0.00476862367696488,0.0167569972611259,0]),
            '>35': np.array([0.986355185886955,0,0.0136448141130445,0,0,0,0,0,0,0]),
           
        }
    }

    return raw


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


def empowerment_sexual_autonomy(ages, regression_fun, regression_pars=None):
    """
    Interpolate data from DHS and extrapolate to cover the full range of ages

    NOTE: this is a temporary implementation to illustrate different parameterisation to
    interpolate/extrapolate DHS data.
    """
    arr = regression_fun(ages, *regression_pars)
    if regression_fun.__name__ == "piecewise_linear":  # piecewise linear interpolation
        # Set the metric to zero for ages < 5
        arr[ages < 5] = 0.0
        # Set the metric to zero if it goes below 0
        arr[arr < 0] = 0.0
        # Set metric to 1 if it goes above with this parameterisation
        arr[arr > 1] = 1.0
    return arr


def empowerment_decision_wages(ages, regression_fun, regression_pars=None):
    """
    Interpolate data from DHS and extrapolate to cover the full range of ages

    NOTE: this is a temporary implementation to illustrate different parameterisation to
    interpolate/extrapolate DHS data.
    """
    arr = regression_fun(ages, *regression_pars)
    if regression_fun.__name__ == "piecewise_linear":  # piecewise linear interpolation
        # Set the metric to zero for ages < 5
        arr[ages < 5] = 0.0
        # Set other metric to zero if it goes below 0
        arr[arr < 0] = 0.0
        # Set metric to 1 if it goes above with this parameterisation
        arr[arr > 1] = 1.0
    return arr


def empowerment_decision_health(ages, regression_fun, regression_pars=None):
    """
    Interpolate data from DHS and extrapolate to cover the full range of ages

    NOTE: this is a temporary implementation to illustrate different parameterisation to
    interpolate/extrapolate DHS data.
    """
    arr = regression_fun(ages, *regression_pars)
    if regression_fun.__name__ == "piecewise_linear":  # piecewise linear interpolation
        # Set other metric to zero if it goes below 0
        arr[arr < 0] = 0.0
        # Set metric to 1 if it goes above with this parameterisation
        arr[arr > 1] = 1.0
    return arr


def empowerment_paid_employment(ages, regression_fun, regression_pars=None):
    """
    Interpolate data from DHS and extrapolate to cover the full range of ages

    NOTE: this is a temporary implementation to illustrate different parameterisation to
    interpolate/extrapolate DHS data.

    """
    arr = regression_fun(ages, *regression_pars)

    if regression_fun.__name__ == "piecewise_linear": # piecewise linear interpolation
        inflection_age, inflection_prob, m1, m2 = regression_pars
        # Set other probabilities to zero in the range 5 <= age < 15
        arr[arr < 0] = 0.0
        # Decline probability of having paid wages above 60 -- age of retirement in Kenya
        inflection_age_2 = 60
        if m2 > 0:
            m3 = -m2 # NOTE: assumption
        else:
            m3 = m2
        age_inds = sc.findinds(ages >= inflection_age_2 - 5)
        arr[age_inds] = regression_fun(ages[age_inds], inflection_age_2, arr[inflection_age_2], m2, m3)
        arr[ages < 5] = 0.0
    return arr


def empowerment_regression_pars(regression_type='logistic'):
    """
    Return initial guesses of parameters for the corresponding regression function.
    These parameters have been estimated from the mean estimates of each metric over the range 15-49 years old
    """
    if regression_type == "pwlin":
    # Parameters for two-part piecewise lienar interpolation, p0: age, p1: val at age, p2: slope < age,  p: slope >= age
        regression_pars = {"paid_employment": [25.0, 0.6198487     , 6.216042e-02  ,  0.0008010242],
                           "decision_wages":  [28.0, 0.5287573     , 4.644537e-02  , -0.001145422],
                           "decision_health": [16.0, 9.90297066e-01, 6.26846208e-02,  1.44754082e-04],
                           "sexual_autonomy": [25.0, 0.8292142     , 0.025677      , -0.003916498]}
        regression_fun = fpu.piecewise_linear
    elif regression_type == 'logistic':
        # Parameters for product of sigmoids
        regression_pars = {"paid_employment": [-6.33459372e-01, -2.07598104e-03,  1.02375876e+01,  5.03843348e-01],
                           "decision_wages":  [-6.33459372e-01, -2.07598104e-03,  1.02375876e+01,  5.03843348e-01],
                           "decision_health": [-4.36694812e+00, 1.72072493e-02 ,  2.92858981e+02,  1.97195136e+01],
                           "sexual_autonomy": [ 6.25030509    , 0.43789906     , -1.83553293    , -0.015715291]}
        regression_fun = fpu.sigmoid_product
    else:
        mssg = f"Not implemented or unknown regression type [{regression_type}]."
        raise NotImplementedError(mssg)

    return regression_pars, regression_fun


# Empowerment metrics
def empowerment_distributions(seed=None, regression_type='logistic'):
    """Intial distributions of empowerment attributes based on latest DHS data <YYYY>
    TODO: perhaps split into single functions, one per attribute?
    TODO: update docstring for empowerment_distributions
    NOTE: DHS data covers the age group from 15 to 49 (inclusive). In this function we
    interpolate data to reduce noise and extrapolate to cover the age range (0, 100).
    Interpolation is done using a piecewise linear approximation with an inflexion point
    on

    Paid employment (https://github.com/fpsim/fpsim/issues/185)
    0.6198487 at age 25
    slope <25, 6.216042e-02 (SE 2.062729e-03)
    slope >25, 0.0008010242 (SE 0.0592966648)

    Control over wages (https://github.com/fpsim/fpsim/issues/187)
    Parameterization:
    0.9434381 at age 20
    slope <20, 2.548961e-02 (SE 5.243655e-03)
    slope >20, 0.0008366125 (SE 0.0194093421)

    Sexual autonomy (https://github.com/fpsim/fpsim/issues/188)
    Parameterization:
    0.8292142 at age 25
    slope <25, 0.025677 (SE 0.003474)
    slope>25, -0.003916498 (SE 0.026119389)
    """
    from scipy import optimize

    # Load empirical data
    empowerment_data = pd.read_csv(thisdir / 'kenya' / 'empowerment.csv')
    mean_cols = {col: col + '.mean' for col in empowerment_data.columns if not col.endswith('.se') and not col == "age"}
    empowerment_data.rename(columns=mean_cols, inplace=True)
    empowerment_dict = {}

    # TODO: Think of a better way to initialize this?
    # Set seed
    if seed is None:
        seed = 42
    fpu.set_seed(seed)

    # TODO: parametrise so the users can decide which function to use?
    regression_pars, regression_fun = empowerment_regression_pars(regression_type=regression_type)

    data_points = {"paid_employment": [], "decision_wages":  [], "decision_health": [], "sexual_autonomy": []}
    cols = ["paid_employment", "decision_wages", "decision_health", "sexual_autonomy"]
    ages_interp = empowerment_data["age"].to_numpy()
    for col in cols:
        loc   = empowerment_data[f"{col}.mean"]
        scale = empowerment_data[f"{col}.se"]
        # Use the standard error to capture the uncertainty in the mean eastimates of each metric
        data = np.random.normal(loc=loc, scale=scale)
        data_points[col] = data
        # Optimise regression parameters
        fit_pars, fit_err = optimize.curve_fit(regression_fun, ages_interp, data, p0=regression_pars[col])
        # Update regression parameters
        regression_pars[col]  = fit_pars

    # Create vector of ages 0, 99 (inclusive) to extrapolate data
    ages = np.arange(100.0)

    # Interpolate and extrapolate data for different empowerment metrics
    empowerment_dict["age"] = ages
    empowerment_dict["paid_employment"] = empowerment_paid_employment(ages, regression_fun, regression_pars=regression_pars["paid_employment"])
    empowerment_dict["decision_wages"]  = empowerment_decision_wages(ages, regression_fun, regression_pars=regression_pars["decision_wages"])
    empowerment_dict["decision_health"] = empowerment_decision_health(ages, regression_fun, regression_pars=regression_pars["decision_health"])
    empowerment_dict["sexual_autonomy"] = empowerment_sexual_autonomy(ages, regression_fun, regression_pars=regression_pars["sexual_autonomy"])
    # Store the estimates of each metric and the optimised regression parameters
    empowerment_dict["regression_pars"] = regression_pars
    empowerment_dict["sampled_points"] = data_points

    return empowerment_dict, empowerment_data


def age_partnership():
    """ Probabilities of being partnered at age X"""
    age_partnership_data = pd.read_csv(thisdir / 'data' / 'age_partnership.csv')
    partnership_dict = {}
    partnership_dict["age"] = age_partnership_data["age_partner"].to_numpy()
    partnership_dict["partnership_probs"] = age_partnership_data["percent"].to_numpy()
    return  partnership_dict


def education_objective(df):
    """
    Convert education objective data to necesary numeric types and into a numpy array
    NOTE: These values are based on the distribution of education for women over age 20 with no children,
    stratified by urban/rural from DHS.
    """
    # This df has columns
    # edu: years education, urban: geographic setting, percent:
    # transformed to a 2d array of proportions with dimensions (n_urban, n_edu_years)
    arr = df["percent"].to_numpy().reshape(df["urban"].nunique(), df["edu"].nunique())
    return arr


def education_attainment(df):
    """
    Convert education attainment data to necessary numeric types and into a numpy array
    These data are the mean years of education of a woman aged X years from DHS.

    NOTE: The data in education_initialization.csv have been extrapolated. Here we only
    interpolate data for the group 15-49 (inclusive range).
    """
    # This df has columns
    # age:age in years and edu: mean years of education
    df.sort_values(by="age", ascending=True, inplace=True)
    ages = df["age"].to_numpy()
    arr  = df["edu"].to_numpy()

    # We interpolate data from 15-49 years
    # Get indices of those ages
    inds = np.array(sc.findinds(ages >= 15, ages <= 55))
    from scipy import interpolate
    # TODO: parameterise interpolation, or provide interpolated data in csv file
    f_interp = interpolate.interp1d(ages[inds[::4]], arr[inds[::4]], kind="quadratic")
    arr[inds] = f_interp(ages[inds])
    return arr, ages


def education_dropout_probs(df):
    """
    Convert education dropout probability to necessary numeric types and data structure

    NOTE: This df contains PMA data:
    - Of women with a first birth before age 18, 12.6% stopped education within 1 year of that birth.
    - Of women who had a subsequent (not first) birth before age 18, 14.1% stopped school within 1 year of that birth.

    The probabilities in this df represents the prob of stopping/droppping out of education within 1 year of that birth.
    """
    data = {}
    for k in df["parity"].unique():
        data[k] = {"age": None, "percent": None}
        data[k]["age"] = df["age"].unique()
        data[k]["percent"] = df["percent"][df["parity"] == k].to_numpy()
    return data


def education_distributions():
    # Load empirical data
    education_data = {"edu_objective": pd.read_csv(thisdir / 'kenya' / 'edu_objective.csv'),
                      "edu_attainment": pd.read_csv(thisdir / 'kenya' / 'edu_initialization.csv'),
                      "edu_dropout_probs": pd.read_csv(thisdir / 'kenya' / 'edu_stop.csv')}

    attainment, age = education_attainment(education_data["edu_attainment"])
    education_dict = {"age": age,
                      "age_start": 6.0,
                      "edu_objective": education_objective(education_data["edu_objective"]),
                      "edu_attainment": attainment,
                      "edu_dropout_probs": education_dropout_probs(education_data["edu_dropout_probs"]),
                      }

    return education_dict, education_data


# %% Make and validate parameters

def make_pars(use_empowerment=None, use_education=None, use_partnership=None, use_subnational=None, seed=None):
    """
    Take all parameters and construct into a dictionary
    """

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
    pars['methods'] = methods()
    pars['methods']['raw'] = method_probs()
    pars['barriers'] = barriers()

    # Empowerment metrics
    if use_empowerment:
        empowerment_dict, _ = empowerment_distributions(seed=seed)  # This function returns extrapolated and raw data
        pars['empowerment'] = empowerment_dict
    if use_education:
        education_dict, _ = education_distributions() # This function returns extrapolated and raw data
        pars['education'] = education_dict
    if use_partnership:
        pars['age_partnership'] = age_partnership()

    kwargs = locals()
    not_implemented_args = ['use_subnational']
    true_args = [key for key in not_implemented_args if kwargs[key] is True]
    if true_args:
        errmsg = f"{true_args} not implemented yet for {pars['location']}"
        raise NotImplementedError(errmsg)

    return pars
