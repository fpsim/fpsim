'''
Set the parameters for FPsim, specifically for Senegal.
'''

import numpy as np
import sciris as sc
from scipy import interpolate as si
from .. import defaults as fpd

#%% Housekeeping

def scalar_pars():
    scalar_pars = {
        # Basic parameters
        'location'               : 'senegal',
        'n_agents'               : 10_000, # Number of agents
        'scaled_pop'             : None, # Scaled population / total population size
        'start_year'             : 1960, # Start year of simulation
        'end_year'               : 2020, # End year of simulation
        'timestep'               : 1, # The simulation timestep in months
        'method_timestep'        : 1, # How many simulation timesteps to go for every method update step
        'seed'                   : 1, # Random seed
        'verbose'                : 1, # How much detail to print during the simulation
        'track_switching'        : 0, # Whether to track method switching

        # Age limits (in years)
        'method_age'             : 15,
        'age_limit_fecundity'    : 50,
        'max_age'                : 99,

        # Durations (in months)
        'switch_frequency'       : 12, # How frequently to check for changes to contraception
        'end_first_tri'          : 3,
        'preg_dur_low'           : 9,
        'preg_dur_high'          : 9,
        'postpartum_dur'         : 35,
        'breastfeeding_dur_mu'   : 19.66828, # For a Gumbel distribution, drawn from DHS
        'breastfeeding_dur_beta' : 7.2585,

        # Pregnancy outcomes
        'abortion_prob'             : 0.08, # From https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4712915/
        'twins_prob'                : 0.015, # From https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0025239
        'LAM_efficacy'              : 0.98, # From Cochrane review: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6823189/
        'maternal_mortality_factor' : 1,

         # Fecundity and exposure
        'fecundity_var_low'      : 0.7,
        'fecundity_var_high'     : 1.1,
        'high_parity'            : 4,
        'high_parity_nonuse'     : 0.6,
        'primary_infertility'    : 0.05,
        'exposure_factor'        : 1.0, # Overall exposure correction factor

        # MCPR
        'mcpr_growth_rate'       : 0.02, # The year-on-year change in MCPR after the end of the data
        'mcpr_max'               : 0.90, # Do not allow MCPR to increase beyond this
        'mcpr_norm_year'         : 2018, # Year to normalize MCPR trend to 1
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
    files['base'] = sc.thisdir(aspath=True) / 'senegal'
    files['basic_dhs']        = 'senegal-basic-dhs.yaml'
    files['pregnancy_parity'] = 'SNIR80FL.obj'
    files['popsize']          = 'senegal-popsize.csv'
    files['mcpr']             = 'mcpr_senegal.csv'
    files['tfr']              = 'senegal-tfr.csv'
    files['skyscrapers']      = 'Skyscrapers-All-DHS.csv'
    files['spacing']          = 'BirthSpacing.obj'
    files['methods']          = 'Method_v312.csv'
    return files


#%% Demographics and pregnancy outcome

def age_pyramid():
    ''' Starting age bin, male population, female population '''
    pyramid = np.array([
         [0,  318225,  314011], # Senegal 1962
         [5,  249054,  244271],
        [10,  191209,  190998],
        [15,  157800,  159536],
        [20,  141480,  141717],
        [25,  125002,  124293],
        [30,  109339,  107802],
        [35,  93359,   92119],
        [40,  77605,   78231],
        [45,  63650,   66117],
        [50,  51038,   54934],
        [55,  39715,   44202],
        [60,  29401,   33497],
        [65,  19522,   23019],
        [70,  11686,   14167],
        [75,  5985,    7390],
        [80,  2875,    3554],
    ], dtype=float)

    return pyramid


def age_mortality():
    '''
    Age-dependent mortality rates, Senegal specific from 1990-1995 -- see age_dependent_mortality.py in the fp_analyses repository
    Mortality rate trend from crude mortality rate per 1000 people: https://data.worldbank.org/indicator/SP.DYN.CDRT.IN?locations=SN
    '''
    mortality = {
        'bins': np.array([ 0.,  5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]),
        'm': np.array([0.03075891, 0.00266326, 0.00164035, 0.00247776, 0.00376541,0.00377009, 0.00433534, 0.00501743, 0.00656144, 0.00862479, 0.01224844, 0.01757291, 0.02655129, 0.0403916 , 0.06604032,0.10924413, 0.17495116, 0.26531436, 0.36505174, 0.43979833]),
        'f': np.array([0.02768283, 0.00262118, 0.00161414, 0.0023998 , 0.00311697, 0.00354105, 0.00376715, 0.00429043, 0.00503436, 0.00602394, 0.00840777, 0.01193858, 0.01954465, 0.03220238, 0.05614077, 0.0957751 , 0.15973906, 0.24231313, 0.33755308, 0.41632442])
    }

    mortality['year']  = np.array([1950., 1955, 1960, 1965, 1970, 1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025, 2030]) # Starting year bin
    mortality['probs'] = np.array([28,    27,    26.023, 25.605, 24.687, 20.995, 16.9, 13.531, 11.335, 11.11, 10.752, 9.137, 7.305, 6.141, 5.7, 5.7, 5.7]) # First 2 estimated, last 3 are projected
    mortality['probs'] /= mortality['probs'][8]  # Normalize around 2000 for trending # CK: TODO: shouldn't be hardcoded

    m_mortality_spline_model = si.splrep(x=mortality['bins'], y=mortality['m'])  # Create a spline of mortality along known age bins
    f_mortality_spline_model = si.splrep(x=mortality['bins'], y=mortality['f'])
    m_mortality_spline = si.splev(fpd.spline_ages, m_mortality_spline_model)  # Evaluate the spline along the range of ages in the model with resolution
    f_mortality_spline = si.splev(fpd.spline_ages, f_mortality_spline_model)
    m_mortality_spline = np.minimum(1, np.maximum(0, m_mortality_spline)) # Normalize
    f_mortality_spline = np.minimum(1, np.maximum(0, f_mortality_spline))

    mortality['m_spline'] = m_mortality_spline
    mortality['f_spline'] = f_mortality_spline

    return mortality


def maternal_mortality():
    '''
    Risk of maternal death assessed at each pregnancy. Data from Huchon et al. (2013) prospective study on risk of maternal death in Senegal and Mali.
    Maternal deaths: The annual number of female deaths from any cause related to or aggravated by pregnancy
    or its management (excluding accidental or incidental causes) during pregnancy and childbirth or within
    42 days of termination of pregnancy, irrespective of the duration and site of the pregnancy,
    expressed per 100,000 live births, for a specified time period.
    '''

    data = np.array([
        [1990, 0.00278, 0.00319, 0.00364],
        [2000, 0.00268, 0.00309, 0.00354],
        [2001, 0.00263, 0.00304, 0.00350],
        [2002, 0.00259, 0.00300, 0.00346],
        [2003, 0.00255, 0.00296, 0.00341],
        [2004, 0.00252, 0.00293, 0.00338],
        [2005, 0.00249, 0.00290, 0.00335],
        [2006, 0.00245, 0.00286, 0.00331],
        [2007, 0.00242, 0.00283, 0.00329],
        [2008, 0.00237, 0.00278, 0.00323],
        [2009, 0.00230, 0.00271, 0.00317],
        [2010, 0.00220, 0.00261, 0.00306],
        [2011, 0.00207, 0.00248, 0.00293],
        [2012, 0.00194, 0.00235, 0.00280],
        [2013, 0.00182, 0.002228327, 0.00268338],
        [2014, 0.00172, 0.00213, 0.00258],
        [2015, 0.00161, 0.00202, 0.00248],
        [2016, 0.00152, 0.00193, 0.00239],
        [2017, 0.00143, 0.00184, 0.00230],
        [2018, 0.00135, 0.00176, 0.00222],
        [2019, 0.00128, 0.00169, 0.00214]
    ])

    maternal_mortality = {}
    maternal_mortality['year'] = data[:,0]
    maternal_mortality['probs'] = data[:,3]

    return maternal_mortality


def infant_mortality():
    '''
    From World Bank indicators for infant morality (< 1 year) for Senegal, per 1000 live births
    From API_SP.DYN.IMRT.IN_DS2_en_excel_v2_1495452.numbers
    Adolescent increased risk of infant mortality gradient taken
    from Noori et al for Sub-Saharan African from 2014-2018.  Odds ratios with age 23-25 as reference group:
    https://www.medrxiv.org/content/10.1101/2021.06.10.21258227v1
    '''

    data = np.array([
        [1960, 128.3],
        [1961, 128.2],
        [1962, 128.3],
        [1963, 128.6],
        [1964, 128.8],
        [1965, 128.9],
        [1966, 128.6],
        [1967, 128.2],
        [1968, 127.7],
        [1969, 126.7],
        [1970, 125.4],
        [1971, 123.6],
        [1972, 121.4],
        [1973, 118.7],
        [1974, 115.6],
        [1975, 112.2],
        [1976, 108.4],
        [1977, 104.6],
        [1978, 101],
        [1979, 97.7],
        [1980, 94.9],
        [1981, 92.5],
        [1982, 90.4],
        [1983, 88.2],
        [1984, 85.7],
        [1985, 82.9],
        [1986, 79.9],
        [1987, 77],
        [1988, 74.4],
        [1989, 72.4],
        [1990, 71],
        [1991, 70.3],
        [1992, 70.1],
        [1993, 70.3],
        [1994, 70.6],
        [1995, 71],
        [1996, 71.2],
        [1997, 71.1],
        [1998, 70.6],
        [1999, 69.4],
        [2000, 67.5],
        [2001, 65.1],
        [2002, 62.2],
        [2003, 59.2],
        [2004, 56.1],
        [2005, 53.2],
        [2006, 50.6],
        [2007, 48.2],
        [2008, 46.2],
        [2009, 44.3],
        [2010, 42.6],
        [2011, 41.1],
        [2012, 39.8],
        [2013, 38.6],
        [2014, 37.5],
        [2015, 36.4],
        [2016, 35.4],
        [2017, 34.4],
        [2018, 33.6],
        [2019, 32.7],
    ])

    infant_mortality = {}
    infant_mortality['year'] = data[:,0]
    infant_mortality['probs'] = data[:,1]/1000   # Rate per 1000 live births, used after stillbirth is filtered out
    infant_mortality['ages'] = np.array([16, 17,   19, 22,   25, 50])
    infant_mortality['age_probs'] = np.array([2.28, 1.63, 1.3, 1.12, 1.0, 1.0])

    return infant_mortality


def miscarriage():
    '''
    Returns a linear interpolation of the likelihood of a miscarriage
    by age, taken from data from Magnus et al BMJ 2019: https://pubmed.ncbi.nlm.nih.gov/30894356/
    Data to be fed into likelihood of continuing a pregnancy once initialized in model
    Age 0 and 5 set at 100% likelihood.  Age 10 imputed to be symmetrical with probability at age 45 for a parabolic curve
    '''
    miscarriage_rates = np.array([[0, 5,    10,     15,     20,    25,    30,    35,    40,    45,    50],
                                  [1, 1, 0.569,  0.167,  0.112, 0.097, 0.108, 0.167, 0.332, 0.569, 0.569]])
    miscarriage_interp = data2interp(miscarriage_rates, fpd.spline_preg_ages)
    return miscarriage_interp


def stillbirth():
    '''
    From Report of the UN Inter-agency Group for Child Mortality Estimation, 2020
    https://childmortality.org/wp-content/uploads/2020/10/UN-IGME-2020-Stillbirth-Report.pdf
    '''

    data = np.array([
        [2000, 25.3],
        [2010, 22.6],
        [2019, 19.7],
        ])

    stillbirth_rate = {}
    stillbirth_rate['year'] = data[:,0]
    stillbirth_rate['probs'] = data[:,1]/1000    # Rate per 1000 total births

    return stillbirth_rate



#%% Fecundity

def female_age_fecundity():
    '''
    Use fecundity rates from PRESTO study: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5712257/
    Fecundity rate assumed to be approximately linear from onset of fecundity around age 10 (average age of menses 12.5) to first data point at age 20
    45-50 age bin estimated at 0.10 of fecundity of 25-27 yr olds, based on fertility rates from Senegal
    '''
    fecundity = {
        'bins': np.array([0., 5, 10, 15, 20,     25,   28,  31,   34,   37,  40,   45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99]),
        'f': np.array([0.,    0,  0, 65, 70.8, 79.3,  77.9, 76.6, 74.8, 67.4, 55.5, 7.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])}
    fecundity['f'] /= 100  # Conceptions per hundred to conceptions per woman over 12 menstrual cycles of trying to conceive

    fecundity_interp_model = si.interp1d(x=fecundity['bins'], y=fecundity['f'])
    fecundity_interp = fecundity_interp_model(fpd.spline_preg_ages)
    fecundity_interp = np.minimum(1, np.maximum(0, fecundity_interp)) # Normalize to avoid negative or >1 values

    return fecundity_interp


def fecundity_ratio_nullip():
    '''
    Returns an array of fecundity ratios for a nulliparous woman vs a gravid woman
    from PRESTO study: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5712257/
    Approximates primary infertility and its increasing likelihood if a woman has never conceived by age
    '''
    fecundity_ratio_nullip = np.array([[ 0,  5, 10, 12.5,  15,  18,  20,   25,   30,   34,   37,   40,   45,   50],
                                        [1,  1,  1,    1,   1,   1,   1, 0.96, 0.95, 0.71, 0.73, 0.42, 0.42, 0.42]])
    fecundity_nullip_interp = data2interp(fecundity_ratio_nullip, fpd.spline_preg_ages)

    return fecundity_nullip_interp


def lactational_amenorrhea():
    '''
    Returns an array of the percent of breastfeeding women by month postpartum 0-11 months who meet criteria for LAM:
    Exclusively breastfeeding (bf + water alone), menses have not returned.  Extended out 5-11 months to better match data
    as those women continue to be postpartum insusceptible.
    From DHS Senegal calendar data
    '''
    data = np.array([
        [0, 0.903125],
        [1, 0.868794326],
        [2, 0.746478873],
        [3, 0.648854962],
        [4, 0.563573883],
        [5, 0.457564576],
        [6, 0.254966887],
        [7, 0.2],
        [8, 0.146341463],
        [9, 0.10982659],
        [10, 0.10982659],
        [11, 0.101796407],
    ])

    lactational_amenorrhea = {}
    lactational_amenorrhea['month'] = data[:, 0]
    lactational_amenorrhea['rate'] = data[:, 1]

    return lactational_amenorrhea



#%% Pregnancy exposure

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

    sexually_active = np.array([[0, 5, 10,    15,   20,   25,   30,   35,   40,   45,   50],
                                [0, 0,  0,  50.4, 55.9, 57.3, 60.8, 66.4, 67.5, 68.2, 68.2]])

    sexually_active[1] /= 100 # Convert from percent to rate per woman
    activity_ages = sexually_active[0]
    activity_interp_model = si.interp1d(x=activity_ages, y=sexually_active[1])
    activity_interp = activity_interp_model(fpd.spline_preg_ages)  # Evaluate interpolation along resolution of ages

    return activity_interp


def sexual_activity_pp():
    '''
    Returns an array of monthly likelihood of having resumed sexual activity within 0-35 months postpartum
    Uses DHS Senegal 2018 individual recode (postpartum (v222), months since last birth, and sexual activity within 30 days.
    Limited to 35 months postpartum (can use any limit you want 0-35 max)
    Postpartum month 0 refers to the first month after delivery
    '''

    postpartum_sex = np.array([
        [0,  0.104166667],
        [1,  0.300000000],
        [2,  0.383177570],
        [3,  0.461538462],
        [4,  0.476635514],
        [5,  0.500000000],
        [6,  0.565217391],
        [7,  0.541666667],
        [8,  0.547368421],
        [9,  0.617391304],
        [10, 0.578947368],
        [11, 0.637254902],
        [12, 0.608247423],
        [13, 0.582278481],
        [14, 0.542553191],
        [15, 0.678260870],
        [16, 0.600000000],
        [17, 0.605042017],
        [18, 0.562500000],
        [19, 0.529411765],
        [20, 0.674698795],
        [21, 0.548780488],
        [22, 0.616161616],
        [23, 0.709401709],
        [24, 0.651376147],
        [25, 0.780219780],
        [26, 0.717647059],
        [27, 0.716417910],
        [28, 0.683544304],
        [29, 0.716417910],
        [30, 0.640625000],
        [31, 0.650000000],
        [32, 0.676470588],
        [33, 0.645161290],
        [34, 0.606557377],
        [35, 0.644736842],
    ])

    postpartum_activity = {}
    postpartum_activity['month'] = postpartum_sex[:, 0]
    postpartum_activity['percent_active'] = postpartum_sex[:, 1]

    return postpartum_activity


def debut_age():
    '''
    Returns an array of weighted probabilities of sexual debut by a certain age 10-45.
    Data taken from DHS variable v531 (imputed age of sexual debut, imputed with data from age at first union)
    Use sexual_debut_age_probs.py under fp_analyses/data to output for other DHS countries
    '''

    sexual_debut = np.array([
        [10, 0.004362494588533180],
        [11, 0.005887267309386780],
        [12, 0.016249279181639800],
        [13, 0.0299019826473517],
        [14, 0.055785658051997],
        [15, 0.09813952463469960],
        [16, 0.112872333807184],
        [17, 0.11953800217275100],
        [18, 0.10881048442282400],
        [19, 0.08688267743864320],
        [20, 0.0781062086093285],
        [21, 0.055562127900473800],
        [22, 0.047649966917757800],
        [23, 0.03670233295320280],
        [24, 0.02962171655627400],
        [25, 0.03071900157389080],
        [26, 0.020088166028125700],
        [27, 0.012959307423989900],
        [28, 0.009789125590573670],
        [29, 0.010992698492904500],
        [30, 0.0064009386756690000],
        [31, 0.00531499008144595],
        [32, 0.004500210075413140],
        [33, 0.004643541107103950],
        [34, 0.0015287248836055500],
        [35, 0.0012933308143284600],
        [36, 0.0008169702220519970],
        [37, 0.0005138447212362420],
        [38, 0.0030994890039629400],
        [39, 0.0007583698086919300],
        [40, 0.0001470674087999730],
        [42, 0.00018238823303343100],
        [43, 0.0000620676775406016],
        [45, 0.0001177109855848480]])

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
    exposure_correction_age = np.array([[0, 5, 10, 12.5,  15,  18,  20,  25,  30,  35,  40,  45,  50],
                                        [1, 1,  1,    1,   1,   1,   1,   1,   1, 1, 1, 1, 1]])
    exposure_age_interp = data2interp(exposure_correction_age, fpd.spline_preg_ages)

    return exposure_age_interp


def exposure_parity():
    '''
    Returns an array of experimental factors to be applied to account for residual exposure to either pregnancy
    or live birth by parity.
    '''
    exposure_correction_parity = np.array([[0, 1, 2, 3, 4, 5, 6,   7,   8,   9,   10,   11,    12,  20],
                                           [1, 1, 1, 1, 1, 1, 1, 0.8, 0.5, 0.3, 0.15, 0.10,  0.05, 0.01]])
    exposure_parity_interp = data2interp(exposure_correction_parity, fpd.spline_parities)

    return exposure_parity_interp


def birth_spacing_pref():
    '''
    Returns an array of birth spacing preferences by closest postpartum month.
    Applied to postpartum pregnancy likelihoods.

    NOTE: spacing bins must be uniform!
    '''
    postpartum_spacing = np.array([
        [0,  0.5],
        [3,  0.5],
        [6,  0.5],
        [9,  0.5],
        [12, 0.8],
        [15, 1.2],
        [18, 5.0],
        [21, 5.0],
        [24, 9.0],
        [27, 9.0],
        [30, 9.0],
        [33, 9.0],
        [36, 5.0],
    ])

    # Calculate the intervals and check they're all the same
    intervals = np.diff(postpartum_spacing[:, 0])
    interval = intervals[0]
    assert np.all(intervals == interval), f'In order to be computed in an array, birth spacing preference bins must be equal width, not {intervals}'
    pref_spacing = {}
    pref_spacing['interval'] = interval # Store the interval (which we've just checked is always the same)
    pref_spacing['n_bins'] = len(intervals) # Actually n_bins - 1, but we're counting 0 so it's OK
    pref_spacing['months'] = postpartum_spacing[:,0]
    pref_spacing['preference'] = postpartum_spacing[:, 1] # Store the actual birth spacing data

    return pref_spacing



#%% Contraceptive methods

def methods():
    '''
    Matrices to give transitional probabilities from 2018 DHS Senegal contraceptive calendar data
    Probabilities in this function are annual probabilities of initiating, discontinuing, continuing
    or switching methods.

    Probabilities at postpartum month 1 are 1 month transitional probabilities for starting a method after delivery.

    Probabilities at postpartum month 6 are 5 month transitional probabilities for starting or changing methods over
    the first 6 months postpartum.

    Data from Senegal DHS contraceptive calendars, 2017 and 2018 combined
    '''
    methods = {}

    # Names and indices of contraceptive methods -- see also defaults.py
    methods['map'] = {
        'None'              : 0,
        'Pill'              : 1,
        'IUDs'              : 2,
        'Injectables'       : 3,
        'Condoms'           : 4,
        'BTL'               : 5,
        'Withdrawal'        : 6,
        'Implants'          : 7,
        'Other traditional' : 8,
        'Other modern'      : 9,
    }

    # Age bins for different method switching matrices -- duplicated in defaults.py
    methods['age_map'] = {
        '<18':   [ 0, 18],
        '18-20': [18, 20],
        '21-25': [20, 25],
        '>25':   [25, fpd.max_age+1], # +1 since we're using < rather than <=
    }

    methods['raw'] = {}
    methods['raw']['annual'] = {
        '<18': np.array([
            [0.9953132643, 0.0001774295, 0.0000506971, 0.0016717604, 0.0011907588, 0.0000000000, 0.0000000000, 0.0013933117, 0.0001774295, 0.0000253488],
            [0.5357744265, 0.3956817610, 0.0000000000, 0.0685438124, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000],
            [0.0000000000, 0.0000000000, 1.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000],
            [0.3779420208, 0.0357636560, 0.0000000000, 0.5646930063, 0.0000000000, 0.0000000000, 0.0000000000, 0.0216013169, 0.0000000000, 0.0000000000],
            [0.3069791534, 0.0000000000, 0.0000000000, 0.0000000000, 0.6930208466, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000],
            [0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000],
            [0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000, 0.0000000000, 0.0000000000, 0.0000000000],
            [0.2002612324, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.7997387676, 0.0000000000, 0.0000000000],
            [0.0525041055, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.9474958945, 0.0000000000],
            [0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000]]),
        '18-20': np.array([
            [0.9773550478, 0.0027107826, 0.0002856633, 0.0103783776, 0.0027107826, 0.0000000000, 0.0000000000, 0.0048461143, 0.0014275685, 0.0002856633],
            [0.4549090811, 0.4753537374, 0.0000000000, 0.0463236223, 0.0000000000, 0.0000000000, 0.0000000000, 0.0234135593, 0.0000000000, 0.0000000000],
            [0.1606987622, 0.0000000000, 0.8393012378, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000],
            [0.4388765348, 0.0196152254, 0.0000000000, 0.5217816111, 0.0098521372, 0.0000000000, 0.0000000000, 0.0049372457, 0.0049372457, 0.0000000000],
            [0.1820343429, 0.0000000000, 0.0000000000, 0.0000000000, 0.8179656571, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000],
            [0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000],
            [0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000, 0.0000000000, 0.0000000000, 0.0000000000],
            [0.1700250992, 0.0000000000, 0.0000000000, 0.0196339461, 0.0000000000, 0.0000000000, 0.0000000000, 0.8103409547, 0.0000000000, 0.0000000000],
            [0.3216043236, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.6783956764, 0.0000000000],
            [0.4773308708, 0.0000000000, 0.0000000000, 0.0000000000, 0.4773308708, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0453382584]]),
        '21-25': np.array([
            [0.9580844982, 0.0080857514, 0.0005744238, 0.0183734782, 0.0024392127, 0.0000000000, 0.0001436343, 0.0107913587, 0.0010767964, 0.0004308462],
            [0.3714750489, 0.5703031529, 0.0000000000, 0.0434765586, 0.0000000000, 0.0000000000, 0.0029538440, 0.0088375516, 0.0029538440, 0.0000000000],
            [0.1079007053, 0.0445194312, 0.8250878450, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0224920185, 0.0000000000, 0.0000000000],
            [0.3776516335, 0.0257873675, 0.0023699011, 0.5835323377, 0.0035529200, 0.0000000000, 0.0000000000, 0.0035529200, 0.0035529200, 0.0000000000],
            [0.1895749834, 0.0094301857, 0.0000000000, 0.0000000000, 0.7540070663, 0.0000000000, 0.0000000000, 0.0187787895, 0.0094301857, 0.0187787895],
            [0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000],
            [0.4471746024, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.5528253976, 0.0000000000, 0.0000000000, 0.0000000000],
            [0.1370415572, 0.0044750998, 0.0029854437, 0.0044750998, 0.0029854437, 0.0000000000, 0.0000000000, 0.8480373557, 0.0000000000, 0.0000000000],
            [0.2375959813, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.7624040187, 0.0000000000],
            [0.3342350643, 0.0000000000, 0.0000000000, 0.0000000000, 0.1826479897, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.4831169460]]),
        '>25': np.array([
            [0.9462131116, 0.0123838433, 0.0029627055, 0.0209275401, 0.0013426008, 0.0001399314, 0.0001399314, 0.0139046297, 0.0017618246, 0.0002238816],
            [0.3030708445, 0.6588856456, 0.0053038550, 0.0210610879, 0.0021246451, 0.0000000000, 0.0015938721, 0.0053038550, 0.0021246451, 0.0005315497],
            [0.0774565806, 0.0058280554, 0.9050475430, 0.0043739679, 0.0000000000, 0.0000000000, 0.0000000000, 0.0043739679, 0.0014599426, 0.0014599426],
            [0.2745544442, 0.0173002840, 0.0047711954, 0.6910279110, 0.0019671434, 0.0000000000, 0.0000000000, 0.0072886657, 0.0028091184, 0.0002812379],
            [0.1548546967, 0.0041688319, 0.0083217274, 0.0124587419, 0.8077054428, 0.0000000000, 0.0000000000, 0.0083217274, 0.0000000000, 0.0041688319],
            [0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000],
            [0.0939227324, 0.0000000000, 0.0000000000, 0.0480226276, 0.0000000000, 0.0000000000, 0.8580546400, 0.0000000000, 0.0000000000, 0.0000000000],
            [0.1114787085, 0.0059115428, 0.0024929872, 0.0068420105, 0.0003119354, 0.0003119354, 0.0003119354, 0.8714034061, 0.0009355387, 0.0000000000],
            [0.1060688610, 0.0025335150, 0.0025335150, 0.0025335150, 0.0050611449, 0.0000000000, 0.0000000000, 0.0050611449, 0.8762083044, 0.0000000000],
            [0.1581337279, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0121394565, 0.8297268156]])
    }

    # Postpartum probabilities, 0-1 month
    methods['raw']['pp0to1'] = {
        '<18':   np.array([0.9606501283, 0.0021385800, 0.0004277160, 0.0128314799, 0.0008554320, 0.0000000000, 0.0008554320, 0.0205303678, 0.0017108640, 0.0000000000]),
        '18-20': np.array([0.9524886878, 0.0028280543, 0.0005656109, 0.0214932127, 0.0005656109, 0.0000000000, 0.0005656109, 0.0197963801, 0.0016968326, 0.0000000000]),
        '21-25': np.array([0.9379245283, 0.0083018868, 0.0013207547, 0.0284905660, 0.0009433962, 0.0000000000, 0.0000000000, 0.0177358491, 0.0052830189, 0.0000000000]),
        '>25':   np.array([0.9253704535, 0.0102379883, 0.0040413112, 0.0264930400, 0.0007184553, 0.0022451729, 0.0001796138, 0.0267624607, 0.0035922766, 0.0003592277])
    }

    # Postpartum probabilities, 1-6 months
    methods['raw']['pp1to6'] = {
        '<18': np.array([
            [0.9013605442, 0.0126336249, 0.0004859086, 0.0510204082, 0.0009718173, 0.0000000000, 0.0000000000, 0.0272108844, 0.0063168124, 0.0000000000],
            [0.4000000000, 0.6000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000],
            [0.0000000000, 0.0000000000, 1.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000],
            [0.0714285714, 0.0000000000, 0.0000000000, 0.9285714286, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000],
            [0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000],
            [0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000],
            [0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.5000000000, 0.5000000000, 0.0000000000, 0.0000000000],
            [0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000, 0.0000000000, 0.0000000000],
            [0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000, 0.0000000000],
            [0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000]]),
        '18-20': np.array([
            [0.8774703557, 0.0191040843, 0.0046113307, 0.0586297760, 0.0032938076, 0.0000000000, 0.0006587615, 0.0329380764, 0.0026350461, 0.0006587615],
            [0.0000000000, 0.7500000000, 0.0000000000, 0.2500000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000],
            [0.0000000000, 0.0000000000, 1.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000],
            [0.0277777778, 0.0000000000, 0.0000000000, 0.9722222222, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000],
            [0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000],
            [0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000],
            [0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000, 0.0000000000, 0.0000000000, 0.0000000000],
            [0.0312500000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.9687500000, 0.0000000000, 0.0000000000],
            [0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000, 0.0000000000],
            [0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000]]),
        '21-25': np.array([
            [0.8538140251, 0.0279182238, 0.0021982853, 0.0721037591, 0.0037370851, 0.0000000000, 0.0004396571, 0.0342932513, 0.0054957133, 0.0000000000],
            [0.0243902439, 0.9512195122, 0.0000000000, 0.0243902439, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000],
            [0.0000000000, 0.0000000000, 1.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000],
            [0.0671641791, 0.0000000000, 0.0000000000, 0.9328358209, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000],
            [0.0000000000, 0.2500000000, 0.0000000000, 0.2500000000, 0.5000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000],
            [0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000],
            [0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000, 0.0000000000, 0.0000000000, 0.0000000000],
            [0.0246913580, 0.0000000000, 0.0000000000, 0.0123456790, 0.0000000000, 0.0000000000, 0.0000000000, 0.9629629630, 0.0000000000, 0.0000000000],
            [0.0000000000, 0.0000000000, 0.0000000000, 0.0416666667, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.9583333333, 0.0000000000],
            [0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000]]),
        '>25': np.array([
            [0.8433467315, 0.0289824413, 0.0070869473, 0.0691770679, 0.0040194627, 0.0001057753, 0.0008462027, 0.0390310979, 0.0064522953, 0.0009519780],
            [0.0673076923, 0.8653846154, 0.0000000000, 0.0288461538, 0.0000000000, 0.0000000000, 0.0000000000, 0.0384615385, 0.0000000000, 0.0000000000],
            [0.0256410256, 0.0000000000, 0.9487179487, 0.0256410256, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000],
            [0.0272373541, 0.0077821012, 0.0000000000, 0.9533073930, 0.0000000000, 0.0000000000, 0.0000000000, 0.0077821012, 0.0038910506, 0.0000000000],
            [0.1666666667, 0.0000000000, 0.0000000000, 0.0000000000, 0.6666666667, 0.0000000000, 0.0000000000, 0.1666666667, 0.0000000000, 0.0000000000],
            [0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000],
            [0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.5000000000, 0.5000000000, 0.0000000000, 0.0000000000],
            [0.0109090909, 0.0036363636, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.9854545455, 0.0000000000, 0.0000000000],
            [0.0270270270, 0.0000000000, 0.0000000000, 0.0270270270, 0.0000000000, 0.0000000000, 0.0000000000, 0.0270270270, 0.9189189189, 0.0000000000],
            [0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000]])
    }

    # Data on trend in MCPR in Senegal over time, in % # CK: TODO: find source
    methods['mcpr_years'] = np.array([1950, 1980, 1986, 1992, 1997, 2005, 2010, 2012, 2014, 2015, 2016, 2017, 2018, 2019, 2020])
    methods['mcpr_rates'] = np.array([0.50,  1.0, 2.65, 4.53, 7.01, 7.62, 8.85, 11.3, 14.7, 15.3, 16.5, 18.8,   19,   20, 20.4])/100

    return methods


def efficacy(disable=False):
    '''
    From Guttmacher, fp/docs/gates_review/contraceptive-failure-rates-in-developing-world_1.pdf
    BTL failure rate from general published data
    Pooled efficacy rates for all women in this study: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4970461/
    '''
    method_efficacy = sc.odict({
        'None'              :  0.0,
        'Pill'              : 94.5,
        'IUDs'              : 98.6,
        'Injectable'        : 98.3,
        'Condoms'           : 94.6,
        'BTL'               : 99.5,
        'Withdrawal'        : 86.6,
        'Implants'          : 99.4,
        'Other traditional' : 86.1, # 1/2 periodic abstinence, 1/2 other traditional approx.  Using rate from periodic abstinence
        'Other modern'      : 88.0, # SDM makes up about 1/2 of this, perfect use is 95% and typical is 88%.  EC also included here, efficacy around 85% https : //www.aafp.org/afp/2004/0815/p707.html
    })

    if disable:
        method_efficacy[:] = 100 # To disable contraception

    method_efficacy = method_efficacy[:]/100

    return method_efficacy


def efficacy25(disable=False):
    '''
    From Guttmacher, fp/docs/gates_review/contraceptive-failure-rates-in-developing-world_1.pdf
    BTL failure rate from general published data
    Pooled efficacy rates for women ages 25+
    '''

    method_efficacy25 = sc.odict({
        'None'       :  0.0,
        'Pill'       : 91.7,
        'IUDs'       : 96.8,
        'Injectable' : 96.5,
        'Condoms'    : 91.1,
        'BTL'        : 99.5,
        'Rhythm'     : 75.4,
        'Withdrawal' : 77.3,
        'Implants'   : 99.4,
        'Other'      : 94.5,
    })

    if disable:
        method_efficacy25[:] = 100 # To disable contraception

    method_efficacy25 = method_efficacy25[:]/100

    return method_efficacy25


def barriers():
    ''' Reasons for nonuse -- taken from DHS '''

    barriers = sc.odict({
      'No need'   :  54.2,
      'Opposition':  30.5,
      'Knowledge' :   1.7,
      'Access'    :   4.5,
      'Health'    :  12.9,
    })

    barriers[:] /= barriers[:].sum() # Ensure it adds to 1
    return barriers



#%% Make and validate parameters

def make_pars():
    '''
    Take all parameters and construct into a dictionary
    '''

    # Scalar parameters and filenames
    pars = scalar_pars()
    pars['filenames'] = filenames()

    # Demographics and pregnancy outcome
    pars['age_pyramid']        = age_pyramid()
    pars['age_mortality']      = age_mortality()
    pars['maternal_mortality'] = maternal_mortality()
    pars['infant_mortality']   = infant_mortality()
    pars['miscarriage_rates']  = miscarriage()
    pars['stillbirth_rate']    = stillbirth()

    # Fecundity
    pars['age_fecundity']          = female_age_fecundity()
    pars['fecundity_ratio_nullip'] = fecundity_ratio_nullip()
    pars['lactational_amenorrhea'] = lactational_amenorrhea()

    # Pregnancy exposure
    pars['sexual_activity']    = sexual_activity()
    pars['sexual_activity_pp'] = sexual_activity_pp()
    pars['debut_age']          = debut_age()
    pars['exposure_age']       = exposure_age()
    pars['exposure_parity']    = exposure_parity()
    pars['spacing_pref']       = birth_spacing_pref()

    # Contraceptive methods
    pars['method_efficacy']   = efficacy()
    pars['method_efficacy25'] = efficacy25()
    pars['methods']           = methods()
    pars['barriers']          = barriers()

    # Perform validation
    validate_pars(pars)

    return pars


def validate_pars(pars):
    ''' Perform internal validation checks and other housekeeping '''

    # Validate method matrices
    method_map = pars['methods']['map']
    method_age_map = pars['methods']['age_map']
    n = len(method_map)
    raw = pars['methods']['raw']
    age_keys = set(method_age_map.keys())
    for mkey in ['annual', 'pp0to1', 'pp1to6']:
        m_age_keys = set(raw[mkey].keys())
        assert age_keys == m_age_keys, f'Matrix "{mkey}" has inconsistent keys: "{sc.strjoin(age_keys)}" ≠ "{sc.strjoin(m_age_keys)}"'
    for k in age_keys:
        shape = raw['pp0to1'][k].shape
        assert shape == (n,), f'Postpartum method initiation matrix for ages {k} has unexpected shape: should be ({n},), not {shape}'
        for mkey in ['annual', 'pp1to6']:
            shape = raw[mkey][k].shape
            assert shape == (n,n), f'Method matrix {mkey} for ages {k} has unexpected shape: should be ({n},{n}), not {shape}'

    # Copy to defaults, preserving original object ID
    for k,v in method_map.items():
        fpd.method_map[k] = v
    for k,v in method_age_map.items():
        fpd.method_age_map[k] = v

    return