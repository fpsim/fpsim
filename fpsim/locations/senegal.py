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
        'n_agents'               : 1_000, # Number of agents
        'scaled_pop'             : None, # Scaled population / total population size
        'start_year'             : 1960, # Start year of simulation
        'end_year'               : 2020, # End year of simulation
        'timestep'               : 1, # The simulation timestep in months
        'method_timestep'        : 1, # How many simulation timesteps to go for every method update step
        'seed'                   : 1, # Random seed
        'verbose'                : 1, # How much detail to print during the simulation
        'track_switching'        : 0, # Whether to track method switching
        'track_as'               : 0, # Whether to track age-specific channels

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
        'breastfeeding_dur_mu'   : 19.66828, # Location parameter of gumbel distribution. Requires children's recode DHS file, see data_processing/breastfeedin_stats.R 
        'breastfeeding_dur_beta' : 7.2585, # Scale parameter of gumbel distribution. Requires children's recode DHS file, see data_processing/breastfeedin_stats.R 
        'max_lam_dur'            : 5, # Duration of lactational amenorrhea

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
    files['popsize']          = 'senegal-popsize.csv'
    files['mcpr']             = 'senegal-mcpr.csv'
    files['tfr']              = 'senegal-tfr.csv'
    files['asfr']             = 'senegal-asfr.csv'
    files['skyscrapers']      = 'Skyscrapers-All-DHS.csv'
    files['pregnancy_parity'] = 'SNIR80FL.obj'
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
    
    Age adjustments come from an extension of Noori et al., which were conducted June 2022.
    '''

    data = np.array([
        [2000, 25.3],
        [2010, 22.6],
        [2019, 19.7],
        ])

    stillbirth_rate = {}
    stillbirth_rate['year'] = data[:,0]
    stillbirth_rate['probs'] = data[:,1]/1000    # Rate per 1000 total births
    stillbirth_rate['ages']      = np.array([15,   16,   17,   19,   20,   28,  31,  36,   50])
    stillbirth_rate['age_probs'] = np.array([3.27, 1.64, 1.85, 1.39, 0.89, 1.0, 1.5, 1.55, 1.78]) #odds ratios 

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
    Names, indices, modern/traditional flag, and efficacies of contraceptive methods -- see also parameters.py
    Efficacy from Guttmacher, fp_prerelease/docs/gates_review/contraceptive-failure-rates-in-developing-world_1.pdf
    BTL failure rate from general published data
    Pooled efficacy rates for all women in this study: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4970461/
    '''

    # Define method data
    data = {                # Index, modern, efficacy
        'None'              : [0, False, 0.000],
        'Withdrawal'        : [1, False, 0.866],
        'Other traditional' : [2, False, 0.861], # 1/2 periodic abstinence, 1/2 other traditional approx.  Using rate from periodic abstinence
        'Condoms'           : [3, True,  0.946],
        'Pill'              : [4, True,  0.945],
        'Injectables'       : [5, True,  0.983],
        'Implants'          : [6, True,  0.994],
        'IUDs'              : [7, True,  0.986],
        'BTL'               : [8, True,  0.995],
        'Other modern'      : [9, True,  0.880], # SDM makes up about 1/2 of this, perfect use is 95% and typical is 88%.  EC also included here, efficacy around 85% https : //www.aafp.org/afp/2004/0815/p707.html
    }

    keys = data.keys()
    methods = {}
    methods['map']    = {k:data[k][0] for k in keys}
    methods['modern'] = {k:data[k][1] for k in keys}
    methods['eff']    = {k:data[k][2] for k in keys}

    # Age bins for different method switching matrices -- duplicated in defaults.py
    methods['age_map'] = {
        '<18'   : [ 0, 18],
        '18-20' : [18, 20],
        '21-25' : [20, 25],
        '26-35' : [25, 35],
        '>35'   : [35, fpd.max_age+1], # +1 since we're using < rather than <=
    }

    # Data on trend in MCPR in Senegal over time, in % # CK: TODO: find source
    methods['mcpr_years'] = np.array([1950, 1980, 1986, 1992, 1997, 2005, 2010, 2012, 2014, 2015, 2016, 2017, 2018, 2019, 2020])
    methods['mcpr_rates'] = np.array([0.50,  1.0, 2.65, 4.53, 7.01, 7.62, 8.85, 11.3, 14.7, 15.3, 16.5, 18.8,   19,   20, 20.4])/100

    return methods


def method_probs():
    '''
    Define "raw" (un-normalized, un-trended) matrices to give transitional probabilities
    from 2018 DHS Senegal contraceptive calendar data.

    Probabilities in this function are annual probabilities of initiating (top row), discontinuing (first column),
    continuing (diagonal), or switching methods (all other entries).

    Probabilities at postpartum month 1 are 1 month transitional probabilities
    for starting a method after delivery.

    Probabilities at postpartum month 6 are 5 month transitional probabilities
    for starting or changing methods over the first 6 months postpartum.

    Data from Senegal DHS contraceptive calendars, 2017 and 2018 combined
    '''

    raw = {

    # Main switching matrix: all non-postpartum women
    'annual': {
        '<18': np.array([
           [0.9953, 0.    , 0.0002, 0.0012, 0.0002, 0.0017, 0.0014, 0.0001, 0.    , 0.    ],
           [0.    , 1.0000, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ],
           [0.0525, 0.    , 0.9475, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ],
           [0.307 , 0.    , 0.    , 0.693 , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ],
           [0.5358, 0.    , 0.    , 0.    , 0.3957, 0.0685, 0.    , 0.    , 0.    , 0.    ],
           [0.3779, 0.    , 0.    , 0.    , 0.0358, 0.5647, 0.0216, 0.    , 0.    , 0.    ],
           [0.2003, 0.    , 0.    , 0.    , 0.    , 0.    , 0.7997, 0.    , 0.    , 0.    ],
           [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 1.0000, 0.    , 0.    ],
           [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 1.0000, 0.    ],
           [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 1.0000]]),
        '18-20': np.array([
           [0.9774, 0.    , 0.0014, 0.0027, 0.0027, 0.0104, 0.0048, 0.0003, 0.    , 0.0003],
           [0.    , 1.0000, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ],
           [0.3216, 0.    , 0.6784, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ],
           [0.182 , 0.    , 0.    , 0.818 , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ],
           [0.4549, 0.    , 0.    , 0.    , 0.4754, 0.0463, 0.0234, 0.    , 0.    , 0.    ],
           [0.4389, 0.    , 0.0049, 0.0099, 0.0196, 0.5218, 0.0049, 0.    , 0.    , 0.    ],
           [0.17  , 0.    , 0.    , 0.    , 0.    , 0.0196, 0.8103, 0.    , 0.    , 0.    ],
           [0.1607, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.8393, 0.    , 0.    ],
           [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 1.0000, 0.    ],
           [0.4773, 0.    , 0.    , 0.4773, 0.    , 0.    , 0.    , 0.    , 0.    , 0.0453]]),
        '21-25': np.array([
           [0.9581, 0.0001, 0.0011, 0.0024, 0.0081, 0.0184, 0.0108, 0.0006, 0.    , 0.0004],
           [0.4472, 0.5528, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ],
           [0.2376, 0.    , 0.7624, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ],
           [0.1896, 0.    , 0.0094, 0.754 , 0.0094, 0.    , 0.0188, 0.    , 0.    , 0.0188],
           [0.3715, 0.003 , 0.003 , 0.    , 0.5703, 0.0435, 0.0088, 0.    , 0.    , 0.    ],
           [0.3777, 0.    , 0.0036, 0.0036, 0.0258, 0.5835, 0.0036, 0.0024, 0.    , 0.    ],
           [0.137 , 0.    , 0.    , 0.003 , 0.0045, 0.0045, 0.848 , 0.003 , 0.    , 0.    ],
           [0.1079, 0.    , 0.    , 0.    , 0.0445, 0.    , 0.0225, 0.8251, 0.    , 0.    ],
           [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 1.0000, 0.    ],
           [0.3342, 0.    , 0.    , 0.1826, 0.    , 0.    , 0.    , 0.    , 0.    , 0.4831]]),
        '26-35': np.array([
           [0.9462, 0.0001, 0.0018, 0.0013, 0.0124, 0.0209, 0.0139, 0.003 , 0.0001, 0.0002],
           [0.0939, 0.8581, 0.    , 0.    , 0.    , 0.048 , 0.    , 0.    , 0.    , 0.    ],
           [0.1061, 0.    , 0.8762, 0.0051, 0.0025, 0.0025, 0.0051, 0.0025, 0.    , 0.    ],
           [0.1549, 0.    , 0.    , 0.8077, 0.0042, 0.0125, 0.0083, 0.0083, 0.    , 0.0042],
           [0.3031, 0.0016, 0.0021, 0.0021, 0.6589, 0.0211, 0.0053, 0.0053, 0.    , 0.0005],
           [0.2746, 0.    , 0.0028, 0.002 , 0.0173, 0.691 , 0.0073, 0.0048, 0.    , 0.0003],
           [0.1115, 0.0003, 0.0009, 0.0003, 0.0059, 0.0068, 0.8714, 0.0025, 0.0003, 0.    ],
           [0.0775, 0.    , 0.0015, 0.    , 0.0058, 0.0044, 0.0044, 0.905 , 0.    , 0.0015],
           [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 1.0000, 0.    ],
           [0.1581, 0.    , 0.0121, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.8297]]),
        '>35': np.array([
           [0.9462, 0.0001, 0.0018, 0.0013, 0.0124, 0.0209, 0.0139, 0.003 , 0.0001, 0.0002],
           [0.0939, 0.8581, 0.    , 0.    , 0.    , 0.048 , 0.    , 0.    , 0.    , 0.    ],
           [0.1061, 0.    , 0.8762, 0.0051, 0.0025, 0.0025, 0.0051, 0.0025, 0.    , 0.    ],
           [0.1549, 0.    , 0.    , 0.8077, 0.0042, 0.0125, 0.0083, 0.0083, 0.    , 0.0042],
           [0.3031, 0.0016, 0.0021, 0.0021, 0.6589, 0.0211, 0.0053, 0.0053, 0.    , 0.0005],
           [0.2746, 0.    , 0.0028, 0.002 , 0.0173, 0.691 , 0.0073, 0.0048, 0.    , 0.0003],
           [0.1115, 0.0003, 0.0009, 0.0003, 0.0059, 0.0068, 0.8714, 0.0025, 0.0003, 0.    ],
           [0.0775, 0.    , 0.0015, 0.    , 0.0058, 0.0044, 0.0044, 0.905 , 0.    , 0.0015],
           [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 1.0000, 0.    ],
           [0.1581, 0.    , 0.0121, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.8297]])
        },

    # Postpartum switching matrix, 1 to 6 months
    'pp1to6': {
        '<18': np.array([
           [0.9014, 0.    , 0.0063, 0.001 , 0.0126, 0.051 , 0.0272, 0.0005, 0.    , 0.    ],
           [0.    , 0.5   , 0.    , 0.    , 0.    , 0.    , 0.5   , 0.    , 0.    , 0.    ],
           [0.    , 0.    , 1.0000, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ],
           [0.    , 0.    , 0.    , 1.0000, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ],
           [0.4   , 0.    , 0.    , 0.    , 0.6   , 0.    , 0.    , 0.    , 0.    , 0.    ],
           [0.0714, 0.    , 0.    , 0.    , 0.    , 0.9286, 0.    , 0.    , 0.    , 0.    ],
           [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 1.0000, 0.    , 0.    , 0.    ],
           [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 1.0000, 0.    , 0.    ],
           [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 1.0000, 0.    ],
           [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 1.0000]]),
        '18-20': np.array([
           [0.8775, 0.0007, 0.0026, 0.0033, 0.0191, 0.0586, 0.0329, 0.0046, 0.    , 0.0007],
           [0.    , 1.0000, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ],
           [0.    , 0.    , 1.0000, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ],
           [0.    , 0.    , 0.    , 1.0000, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ],
           [0.    , 0.    , 0.    , 0.    , 0.75  , 0.25  , 0.    , 0.    , 0.    , 0.    ],
           [0.0278, 0.    , 0.    , 0.    , 0.    , 0.9722, 0.    , 0.    , 0.    , 0.    ],
           [0.0312, 0.    , 0.    , 0.    , 0.    , 0.    , 0.9688, 0.    , 0.    , 0.    ],
           [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 1.0000, 0.    , 0.    ],
           [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 1.0000, 0.    ],
           [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 1.0000]]),
        '21-25': np.array([
           [0.8538, 0.0004, 0.0055, 0.0037, 0.0279, 0.0721, 0.0343, 0.0022, 0.    , 0.    ],
           [0.    , 1.0000, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ],
           [0.    , 0.    , 0.9583, 0.    , 0.    , 0.0417, 0.    , 0.    , 0.    , 0.    ],
           [0.    , 0.    , 0.    , 0.5   , 0.25  , 0.25  , 0.    , 0.    , 0.    , 0.    ],
           [0.0244, 0.    , 0.    , 0.    , 0.9512, 0.0244, 0.    , 0.    , 0.    , 0.    ],
           [0.0672, 0.    , 0.    , 0.    , 0.    , 0.9328, 0.    , 0.    , 0.    , 0.    ],
           [0.0247, 0.    , 0.    , 0.    , 0.    , 0.0123, 0.963 , 0.    , 0.    , 0.    ],
           [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 1.0000, 0.    , 0.    ],
           [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 1.0000, 0.    ],
           [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 1.0000]]),
        '26-35': np.array([
           [0.8433, 0.0008, 0.0065, 0.004 , 0.029 , 0.0692, 0.039 , 0.0071, 0.0001, 0.001 ],
           [0.    , 0.5   , 0.    , 0.    , 0.    , 0.    , 0.5   , 0.    , 0.    , 0.    ],
           [0.027 , 0.    , 0.9189, 0.    , 0.    , 0.027 , 0.027 , 0.    , 0.    , 0.    ],
           [0.1667, 0.    , 0.    , 0.6667, 0.    , 0.    , 0.1667, 0.    , 0.    , 0.    ],
           [0.0673, 0.    , 0.    , 0.    , 0.8654, 0.0288, 0.0385, 0.    , 0.    , 0.    ],
           [0.0272, 0.    , 0.0039, 0.    , 0.0078, 0.9533, 0.0078, 0.    , 0.    , 0.    ],
           [0.0109, 0.    , 0.    , 0.    , 0.0036, 0.    , 0.9855, 0.    , 0.    , 0.    ],
           [0.0256, 0.    , 0.    , 0.    , 0.    , 0.0256, 0.    , 0.9487, 0.    , 0.    ],
           [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 1.0000, 0.    ],
           [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 1.0000]]),
        '>35': np.array([
           [0.8433, 0.0008, 0.0065, 0.004 , 0.029 , 0.0692, 0.039 , 0.0071, 0.0001, 0.001 ],
           [0.    , 0.5   , 0.    , 0.    , 0.    , 0.    , 0.5   , 0.    , 0.    , 0.    ],
           [0.027 , 0.    , 0.9189, 0.    , 0.    , 0.027 , 0.027 , 0.    , 0.    , 0.    ],
           [0.1667, 0.    , 0.    , 0.6667, 0.    , 0.    , 0.1667, 0.    , 0.    , 0.    ],
           [0.0673, 0.    , 0.    , 0.    , 0.8654, 0.0288, 0.0385, 0.    , 0.    , 0.    ],
           [0.0272, 0.    , 0.0039, 0.    , 0.0078, 0.9533, 0.0078, 0.    , 0.    , 0.    ],
           [0.0109, 0.    , 0.    , 0.    , 0.0036, 0.    , 0.9855, 0.    , 0.    , 0.    ],
           [0.0256, 0.    , 0.    , 0.    , 0.    , 0.0256, 0.    , 0.9487, 0.    , 0.    ],
           [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 1.0000, 0.    ],
           [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 1.0000]])
        },

    # Postpartum initiation vectors, 0 to 1 month
    'pp0to1': {
        '<18':   np.array([0.9607, 0.0009, 0.0017, 0.0009, 0.0021, 0.0128, 0.0205, 0.0004, 0.    , 0.    ]),
        '18-20': np.array([0.9525, 0.0006, 0.0017, 0.0006, 0.0028, 0.0215, 0.0198, 0.0006, 0.    , 0.    ]),
        '21-25': np.array([0.9379, 0.    , 0.0053, 0.0009, 0.0083, 0.0285, 0.0177, 0.0013, 0.    , 0.    ]),
        '26-35': np.array([0.9254, 0.0002, 0.0036, 0.0007, 0.0102, 0.0265, 0.0268, 0.004 , 0.0022, 0.0004]),
        '>35':   np.array([0.9254, 0.0002, 0.0036, 0.0007, 0.0102, 0.0265, 0.0268, 0.004 , 0.0022, 0.0004]),
        }
    }
    return raw


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
    pars['methods']        = methods()
    pars['methods']['raw'] = method_probs()
    pars['barriers']       = barriers()

    return pars