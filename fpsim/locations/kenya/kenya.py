'''
Set the parameters for FPsim, specifically for Kenya.
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
        'location':             'kenya',
        'postpartum_dur':       23,
        'breastfeeding_dur_mu': 11.4261936291137,   # Location parameter of gumbel distribution. Requires children's recode DHS file, see data_processing/breastfeedin_stats.R
        'breastfeeding_dur_beta': 7.5435309020483,  # Location parameter of gumbel distribution. Requires children's recode DHS file, see data_processing/breastfeedin_stats.R
        'abortion_prob':        0.201,              # From https://bmcpregnancychildbirth.biomedcentral.com/articles/10.1186/s12884-015-0621-1, % of all pregnancies calculated
        'twins_prob':           0.016,              # From https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0025239
        'high_parity_nonuse':   1,                  # TODO: check whether it's correct that this should be different to the other locations
        'mcpr_norm_year':       2020,               # Year to normalize MCPR trend to 1
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
    files['popsize'] = 'popsize.csv' # Downloaded from World Bank: https://data.worldbank.org/indicator/SP.POP.TOTL?locations=KE
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
    """Load information about the proportion of people who live in an urban setting"""
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
        [2000, 22.5],
        [2010, 20.6],
        [2019, 19.7],
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
    '''

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

    debut_age = {}
    debut_age['ages'] = sexual_debut[:, 0]
    debut_age['probs'] = sexual_debut[:, 1]

    return debut_age


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


def _check_age_endpoints(df):
    """
    Add 'age endpoints' 0 and fpd.max_age+1 if they are not present in the
    add rows of the dataframe. This is needed for extrapolation if we want to
    have a complete representation of the probabilities across all possible
    ages.

    TODO:PSL: remove after we decide whether we want to have full arrays of
    data, meaning arrays that have data/probs/metric every age represented
    in the range [0-fpd.max_age],
    as opposed to arrays that have a subset of ages represented. The
    difference between the two approaches is how to correctly index
    into the data array.

    Pseudo-code:
    A: with age extrapolation
    age = ppl.age
    empwr_data[int(age)][empwr_metric]

    B: without age extrapolation, we don't need this function _check_age_endpoints()
    age = ppl.age
    age_cutofs = np.hstack(empwr_min_age, empwr_max_age) ## assumes contignuous age range between min, max age
    empwr_age_index = np.digitize(age, age_cuttofs)
    empwr_data[empwr_age_index][empwr_metric]

    A: can be faster than B as we don't have to digitize the agents ages each
    time we want to access the data, we only need to cast ages into integers.
    continous ages - > integer ages > index empowerment data

    B: removes the need for extrapolation and choosing the extrapolation function.
    But we won't be able to directly index using an integer version of the agent ages.

    continuous ages -> binned ages > bin indices -> index empowerment probs

    """

    rows_to_add = []
    for age in [0, fpd.max_age+1]:
        if age not in df['age'].values:
            new_row = pd.Series([0.0] * len(df.columns), index=df.columns)
            new_row['age'] = age
            rows_to_add.append(new_row)

    df = pd.concat([df, pd.DataFrame(rows_to_add)], ignore_index=True)
    df.sort_values('age', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def get_empowerment_loadings():
    """
    Loading coefficients to estimate composite measures for:
     - (1) decision making and
     - (2) financial autonomy

    Bilbiographic ref:
    https://www.tandfonline.com/doi/full/10.1080/13545701.2024.2339979?src=


    """

    df = pd.read_csv(thisdir / 'data' / 'empower_cfa_loadings.csv')
    loadings = df.set_index('empowerment_var')['loading_coef'].to_dict()
    return loadings


# Empowerment metrics
def make_empowerment_pars(seed=None, return_data=False):
    """
    Produce intial distributions of empowerment attributes
    based on latest DHS-8 IR 2022 dataset and PMA Household/Female surveys
    from 2019, 2020 and 2022.

    Because DHS data covers the age group from 15 to 49 (inclusive range),
    in this function we:
    - (1) interpolate data to reduce noise; and
    - (2) extrapolate to cover the age range (0, 100) needed
          to populate the corresponding attributes of each individual agent
          at the start of a simulation.


    By default, interpolation and extrapolation use
    a nonlinear logistic-based function (a product of sigmoids).

    Arguments:
        seed (int): random seed used to generate a new 'empirical' dataset
        using the standard error of the mean (se) specified in the csv files.

        regression_type (str): specifies the function to use to peform
        regression on the empirical data, usually done to extrapolate ages beyond
        the 15-49 range.

    Returns:
        empowerment_dict (dict): A dictionary with the processed data ready
            for fpsim simulations.
        empowerment_data (dict): A dictionary with the preprocessed data loaded
            from the csv files in thisdir/data
    """
    from scipy import optimize

    # Load empirical data
    empowerment_data = pd.read_csv(thisdir / 'data' / 'empowerment.csv')
    empwr_cols = [col for col in empowerment_data.columns if not col.endswith('.se') and not col == "age"]
    mean_cols = {col: col + '.mean' for col in empwr_cols}
    empowerment_data.rename(columns=mean_cols, inplace=True)
    # Output dict
    empowerment_pars = {}

    if seed is None:
        seed = 42
    fpu.set_seed(seed)

    data_points = {col: [] for col in empwr_cols}
    ages = empowerment_data["age"].to_numpy()
    for col in empwr_cols:
        loc   = empowerment_data[f"{col}.mean"]
        scale = empowerment_data[f"{col}.se"]
        # Use the standard error to capture the uncertainty in the mean eastimates of each metric for each age
        empowerment_pars[col] = np.random.normal(loc=loc, scale=scale)

    empowerment_pars["avail_ages"] = ages           # available ages in empowerment data
    empowerment_pars["avail_metrics"] = empwr_cols  # empowerment metrics available in the csv file, does not include the composite metrics

    # Estimate composites for decision making (dm) and financial autonomy (fa)
    # and add them to the empowerment pars
    empowerment_pars["loadings"] = get_empowerment_loadings()
    if return_data:
        return empowerment_pars, empowerment_data
    return empowerment_pars


def fertility_intent_dist():
    """Add additional metrics from PMA Household/Female surveys"""
    df = pd.read_csv(thisdir / 'data' / 'fertility_intent.csv')

    data = {}
    for row in df.itertuples(index=False):
        intent = row.fertility_intent
        age = row.age
        freq = row.freq
        data.setdefault(age, {})[intent] = freq
    return data


def empowerment_update_pars():
    raw_pars = pd.read_csv(thisdir / 'data' / 'empower_coef.csv')
    pars = sc.objdict()
    metrics = raw_pars.lhs.unique()
    for metric in metrics:
        pars[metric] = sc.objdict()
        thisdf = raw_pars.loc[raw_pars.lhs == metric]
        for var_dict in thisdf.to_dict('records'):
            var_name = var_dict['rhs'].replace('_0', '').replace('(', '').replace(')', '').lower()
            if var_name == 'contraception': var_name = 'on_contra'
            pars[metric][var_name] = var_dict['Estimate']
    return pars


def age_partnership():
    """ Probabilities of being partnered at age X"""
    age_partnership_data = pd.read_csv(thisdir / 'data' / 'age_partnership.csv')
    partnership_dict = {}
    partnership_dict["age"] = age_partnership_data["age_partner"].to_numpy()
    partnership_dict["partnership_probs"] = age_partnership_data["percent"].to_numpy()
    return  partnership_dict


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


def process_contra_use_simple():
    # Read in data
    alldfs = [
        pd.read_csv(thisdir / 'data' / 'contra_coef_simple.csv'),
        pd.read_csv(thisdir / 'data' / 'contra_coef_simple_pp1.csv'),
        pd.read_csv(thisdir / 'data' / 'contra_coef_simple_pp6.csv'),
    ]

    contra_use_pars = dict()
    for di, df in enumerate(alldfs):
        contra_use_pars[di] = sc.objdict(
            intercept=df[df['rhs'].str.contains('Intercept')].Estimate.values[0],
            age_factors=df[df['rhs'].str.match('age')].Estimate.values,
            fp_ever_user=df[df['rhs'].str.contains('fp_ever_user')].Estimate.values[0],
        )
    return contra_use_pars


def mcpr():

    mcpr = {}
    cpr_data = pd.read_csv(thisdir / 'data' / 'cpr.csv')
    mcpr['mcpr_years'] = cpr_data['year'].to_numpy()
    mcpr['mcpr_rates'] = cpr_data['cpr'].to_numpy() / 100

    return mcpr


def process_simple_method_pars(methods):
    """ Choice of method is based on age and parity """
    df = pd.read_csv(thisdir / 'data' / 'method_mix.csv')
    # Awful code to speed pandas up
    dd = dict()
    for akey in df.age_grp.unique():
        dd[akey] = dict()
        for pkey in df.parity.unique():
            dd[akey][pkey] = dict()
            for mlabel in df.method.unique():
                val = df.loc[(df.age_grp == akey) & (df.parity == pkey) & (df.method == mlabel)].percent.values[0]
                if mlabel != 'Abstinence':
                    mname = [m.name for m in methods.values() if m.csv_name == mlabel][0]
                    dd[akey][pkey][mname] = val
                else:
                    abstinence_val = sc.dcp(val)
            dd[akey][pkey]['othtrad'] += abstinence_val

    return dd


def process_markovian_method_choice(methods):
    """ Choice of method is age and previous method """
    df = pd.read_csv(thisdir / 'data' / 'method_mix_matrix_switch.csv', keep_default_na=False, na_values=['NaN'])
    csv_map = {method.csv_name: method.name for method in methods.values()}
    idx_map = {method.csv_name: method.idx for method in methods.values()}
    idx_df = {}
    for col in df.columns:
        if col in csv_map.keys():
            idx_df[col] = idx_map[col]

    mc = dict()  # This one is a dict because it will be keyed with numbers
    init_dist = sc.objdict()  # Initial distribution of method choice

    for pp in df.postpartum.unique():
        mc[pp] = sc.objdict()
        mc[pp].method_idx = idx_df.values()
        for akey in df.age_grp.unique():
            mc[pp][akey] = sc.objdict()
            thisdf = df.loc[(df.age_grp == akey) & (df.postpartum == pp)]
            if pp == 1:  # Different logic for immediately postpartum
                mc[pp][akey] = thisdf.values[0][4:13].astype(float)  # If this is going to be standard practice, should make it more robust
            else:
                from_methods = thisdf.From.unique()
                for from_method in from_methods:
                    from_mname = csv_map[from_method]
                    row = thisdf.loc[thisdf.From == from_method]
                    mc[pp][akey][from_mname] = row.values[0][4:13].astype(float)

            # Set initial distributions by age
            if pp == 0:
                init_dist[akey] = thisdf.loc[thisdf.From == 'None'].values[0][4:13].astype(float)
                init_dist.method_idx = idx_df.values()

    return mc, init_dist


def process_dur_use(methods):
    """ Process duration of use parameters"""
    df = pd.read_csv(thisdir / 'data' / 'method_time_coefficients.csv', keep_default_na=False, na_values=['NaN'])
    for method in methods.values():
        if method.name == 'btl':
            method.dur_use = dict(dist='lognormal', par1=100, par2=1, age_factors=[0, 0, 0, 0, 0, 0])
        else:
            mlabel = method.csv_name

            thisdf = df.loc[df.method == mlabel]
            dist = thisdf.functionform.iloc[0]
            method.dur_use = dict()
            method.dur_use['age_factors'] = np.append(thisdf.coef.values[2:], 0)

            if dist == 'lognormal':
                method.dur_use['dist'] = dist
                method.dur_use['par1'] = thisdf.coef[thisdf.estimate == 'meanlog'].values[0]
                method.dur_use['par2'] = thisdf.coef[thisdf.estimate == 'sdlog'].values[0]
            elif dist in ['gamma']:
                method.dur_use['dist'] = dist
                method.dur_use['par1'] = thisdf.coef[thisdf.estimate == 'shape'].values[0]
                method.dur_use['par2'] = thisdf.coef[thisdf.estimate == 'rate'].values[0]
            elif dist == 'llogis':
                method.dur_use['dist'] = dist
                method.dur_use['par1'] = thisdf.coef[thisdf.estimate == 'shape'].values[0]
                method.dur_use['par2'] = thisdf.coef[thisdf.estimate == 'scale'].values[0]

    return methods


# %% Make and validate parameters

def make_pars(seed=None, use_subnational=None):
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
    pars['barriers'] = barriers()
    pars['mcpr'] = mcpr()

    # Demographics: geography and partnership status
    pars['urban_prop'] = urban_proportion()
    pars['age_partnership'] = age_partnership()
    pars['fertility_intent'] = fertility_intent_dist()

    kwargs = locals()
    not_implemented_args = ['use_subnational']
    true_args = [key for key in not_implemented_args if kwargs[key] is True]
    if true_args:
        errmsg = f"{true_args} not implemented yet for {pars['location']}"
        raise NotImplementedError(errmsg)

    return pars
