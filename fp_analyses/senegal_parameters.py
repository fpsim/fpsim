'''
Set the parameters for FPsim, specifically for Senegal.
'''

import os
import numpy as np
import sciris as sc
from scipy import interpolate as si
import fpsim.defaults as fpd

# Define default user-tunable parameters and values
defaults = {
  'name'                          : 'Default',
  'n'                             : 100000,
  'start_year'                    : 1960,
  'end_year'                      : 2019,
  'timestep'                      : 1,
  'verbose'                       : 1,
  'seed'                          : 1,
  'fecundity_variation_low'       : 0.7,
  'fecundity_variation_high'      : 1.1,
  'method_age'                    : 15,
  'max_age'                       : 99,
  'preg_dur_low'                  : 9,
  'preg_dur_high'                 : 9,
  'switch_frequency'              : 12,
  'breastfeeding_dur_low'         : 1,
  'breastfeeding_dur_high'        : 24,
  'age_limit_fecundity'           : 50,
  'postpartum_length'             : 35,
  'end_first_tri'                 : 3,
  'abortion_prob'                 : 0.08, # From https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4712915/
  'twins_prob'                    : 0.015, # From https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0025239
  'LAM_efficacy'                  : 0.98, # From Cochrane review: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6823189/
  'maternal_mortality_multiplier' : 1,
  'high_parity'                   : 4,
  'high_parity_nonuse_correction' : 0.6
}


#%% Helper function

def abspath(path, *args):
    '''
    Turn a relative path into an absolute path. Accepts a
    list of arguments and joins them into a path.

    Example:

        import senegal_parameters as sp
        figpath = sp.abspath('figs', 'myfig.png')
    '''
    cwd = os.path.abspath(os.path.dirname(__file__))
    output = os.path.join(cwd, path, *args)
    return output


# %% Set parameters for the simulation

def default_age_pyramid():
    ''' Starting age bin, male population, female population '''
    pyramid = np.array([ [0,  318225,  314011], # Senegal 1962
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

# def default_wealth_index():
#     '''
#     Relative wealth as represented by the DHS 2019 household wealth index
#     '''
#
# 2018 "Poorest" 22.7
#     "Poorer" 22.8
#     "Middle" 21.0
#     "Richer" 18.0
#     "Richest" 15.5


def default_age_mortality(bound):
    ''' Age-dependent mortality rates, Senegal specific from 1990-1995 -- see age_dependent_mortality.py in the fp_analyses repository
    Mortality rate trend from crude mortality rate per 1000 people: https://data.worldbank.org/indicator/SP.DYN.CDRT.IN?locations=SN
    '''
    mortality = {
            'bins': np.array([ 0.,  5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]),
            'm': np.array([0.03075891, 0.00266326, 0.00164035, 0.00247776, 0.00376541,0.00377009, 0.00433534, 0.00501743, 0.00656144, 0.00862479, 0.01224844, 0.01757291, 0.02655129, 0.0403916 , 0.06604032,0.10924413, 0.17495116, 0.26531436, 0.36505174, 0.43979833]),
            'f': np.array([0.02768283, 0.00262118, 0.00161414, 0.0023998 , 0.00311697, 0.00354105, 0.00376715, 0.00429043, 0.00503436, 0.00602394, 0.00840777, 0.01193858, 0.01954465, 0.03220238, 0.05614077, 0.0957751 , 0.15973906, 0.24231313, 0.33755308, 0.41632442])
            }

    mortality['years'] = np.array([1950., 1955, 1960, 1965, 1970, 1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025, 2030]) # Starting year bin
    mortality['trend'] = np.array([28,    27,    26.023, 25.605, 24.687, 20.995, 16.9, 13.531, 11.335, 11.11, 10.752, 9.137, 7.305, 6.141, 5.7, 5.7, 5.7]) # First 2 estimated, last 3 are projected
    mortality['trend'] /= mortality['trend'][8]  # Normalize around 2000 for trending

    m_mortality_spline_model = si.splrep(x=mortality['bins'], y=mortality['m'])  # Create a spline of mortality along known age bins
    f_mortality_spline_model = si.splrep(x=mortality['bins'], y=mortality['f'])
    m_mortality_spline = si.splev(fpd.spline_ages, m_mortality_spline_model)  # Evaluate the spline along the range of ages in the model with resolution
    f_mortality_spline = si.splev(fpd.spline_ages, f_mortality_spline_model)
    if bound:
        m_mortality_spline = np.minimum(1, np.maximum(0, m_mortality_spline))
        f_mortality_spline = np.minimum(1, np.maximum(0, f_mortality_spline))

    mortality['m_spline'] = m_mortality_spline
    mortality['f_spline'] = f_mortality_spline

    return mortality


def default_female_age_fecundity(bound):
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
    fecundity_interp = fecundity_interp_model(fpd.spline_ages)
    if bound:
        fecundity_interp = np.minimum(1, np.maximum(0, fecundity_interp))

    return fecundity_interp


def default_maternal_mortality():
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


def default_infant_mortality():
    '''
    From World Bank indicators for infant morality (< 1 year) for Senegal, per 1000 live births
    From API_SP.DYN.IMRT.IN_DS2_en_excel_v2_1495452.numbers
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
    infant_mortality['ages'] = np.array([0, 15, 16.5,   18.5, 21,   24,    28,  33,   38,   45, 50])
    infant_mortality['age_probs'] = np.array([3.00, 2.32, 1.76, 1.40, 1.22, 1.08, 1.00, 1.09, 1.07, 1.03, 1.03])
    im_interp_model = si.interp1d(x=infant_mortality['ages'], y=infant_mortality['age_probs'])
    im_interp = im_interp_model(fpd.spline_preg_ages)  # Evaluate interpolation along resolution of ages
    infant_mortality['age_gradient'] = im_interp

    return infant_mortality

def default_stillbirth():
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


''''
OLD INITIATION, DISCONTINUATION, AND SWITCHING CONTRACEPTIVE MATRICES.  LEAVING HERE IN CASE USEFUL.
CAN'T COMMENT ON HOW THESE PROBABILITIES WERE CALCULATED
def default_methods():
    methods = {}

    methods['map'] = {'None':0,
                    'Lactation':1,
                    'Implants':2,
                    'Injectables':3,
                    'IUDs':4,
                    'Pill':5,
                    'Condoms':6,
                    'Other':7,
                    'Traditional':8} # Add 'Novel'?
    methods['names'] = list(methods['map'].keys())

    methods['matrix'] = np.array([
       [8.81230657e-01, 0.00000000e+00, 9.56761433e-04, 1.86518124e-03,        1.44017774e-04, 8.45978530e-04, 1.80273996e-04, 1.61138768e-05,        1.46032008e-04],
       [2.21565806e-05, 1.52074712e-04, 2.01423460e-06, 3.02135189e-06,        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,        0.00000000e+00],
       [3.45441233e-04, 0.00000000e+00, 3.29206502e-02, 1.61138768e-05,        5.03558649e-06, 1.40996422e-05, 2.01423460e-06, 0.00000000e+00,        1.00711730e-06],
       [1.22767599e-03, 0.00000000e+00, 3.02135189e-05, 4.28810403e-02,        1.40996422e-05, 6.94910936e-05, 8.05693838e-06, 0.00000000e+00,        6.04270379e-06],
       [3.52491054e-05, 0.00000000e+00, 2.01423460e-06, 3.02135189e-06,        6.10715929e-03, 5.03558649e-06, 0.00000000e+00, 0.00000000e+00,        0.00000000e+00],
       [6.33476780e-04, 0.00000000e+00, 1.30925249e-05, 5.03558649e-05,        6.04270379e-06, 1.97092855e-02, 4.02846919e-06, 1.00711730e-06,        6.04270379e-06],
       [8.35907357e-05, 0.00000000e+00, 5.03558649e-06, 6.04270379e-06,        2.01423460e-06, 3.02135189e-06, 3.94689269e-03, 2.01423460e-06,        1.00711730e-06],
       [1.20854076e-05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,        0.00000000e+00, 0.00000000e+00, 3.02135189e-06, 1.74432716e-03,        1.00711730e-06],
       [4.93487476e-05, 0.00000000e+00, 2.01423460e-06, 4.02846919e-06,        0.00000000e+00, 2.01423460e-06, 0.00000000e+00, 0.00000000e+00,        4.45145846e-03]])

    methods['matrix'][0,0] *= 0.53 # Correct for 2015

    methods['mcpr_years'] = np.array([1950, 1980, 1986, 1992, 1997, 2005, 2010, 2012, 2014, 2015, 2016, 2017])

    mcpr_rates = np.array([0.50, 1.0, 2.65, 4.53, 7.01, 7.62, 8.85, 11.3, 14.7, 15.3, 16.5, 18.8])
    # mcpr_rates /= 100
    # methods['mcpr_multipliers'] = (1-mcpr_rates)**2.0

    methods['trend'] = mcpr_rates[5]/mcpr_rates # normalize trend around 2005 so "no method to no method" matrix entry will increase or decrease based on mcpr that year

    methods['mcpr_multipliers'] = 10/mcpr_rates # No idea why it should be this...

    return methods
'''


def default_methods():
    '''Matrices to give transitional probabilities from 2018 DHS Senegal contraceptive calendar data
    Probabilities in this function are annual probabilities of initiating, discontinuing, continuing
    or switching methods'''
    methods = {}

    methods['map'] = {'None': 0,
                      'Pill': 1,
                      'IUDs': 2,
                      'Injectables': 3,
                      'Condoms': 4,
                      'BTL': 5,
                      'Withdrawal': 6,
                      'Implants': 7,
                      'Other traditional': 8,
                      'Other modern': 9}

    methods['names'] = list(methods['map'].keys())

    methods['probs_matrix'] = {
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
            [0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000],
            [0.9462131116, 0.0123838433, 0.0029627055, 0.0209275401, 0.0013426008, 0.0001399314, 0.0001399314, 0.0139046297, 0.0017618246, 0.0002238816]]),
        '18-20': np.array([
            [0.9773550478, 0.0027107826, 0.0002856633, 0.0103783776, 0.0027107826, 0.0000000000, 0.0000000000, 0.0048461143, 0.0014275685, 0.0002856633],
            [0.4549090811, 0.4753537374, 0.0000000000, 0.0463236223, 0.0000000000, 0.0000000000, 0.0000000000, 0.0234135593, 0.0000000000, 0.0000000000],
            [0.1606987622, 0.0000000000, 0.8393012378, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000],
            [0.4388765348, 0.0196152254, 0.0000000000, 0.5217816111, 0.0098521372, 0.0000000000, 0.0000000000, 0.0049372457, 0.0049372457, 0.0000000000],
            [0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000],
            [0.1820343429, 0.0000000000, 0.0000000000, 0.0000000000, 0.8179656571, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000],
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

    '''
    Not age-stratified, if wanting to revert to these
    methods['probs_matrix'] = np.array([
        [0.9675668047,   0.0067289610,   0.0013039766,   0.0131728314,   0.0016425499,   0.0000513672,  0.0005032939, 0.0000719133, 0.0081880486, 0.0007702535],
        [0.3175086974,   0.6396387998,   0.0042690889,   0.0262002878,   0.0017096448,   0.0000000000,  0.0008551576, 0.0017096448,  0.0059720416, 0.0021366372],
        [0.0785790764,   0.0079928528,   0.9014183760,   0.0040037760,   0.0000000000,   0.0000000000,  0.00267081978, 0.0000000000, 0.0053350991, 0.0000000000],
        [0.3039655976,   0.0195387097,   0.0040997551,   0.6582634673,   0.0026665967,   0.0000000000,   0.0008211866, 0.0000000000, 0.0083879112, 0.0022567758],
        [0.1856456515,   0.0043699858,   0.0043699858,   0.0087224598,   0.7794338635,   0.0000000000,   0.0021871859, 0.0000000000, 0.0087224598, 0.0065484078],
        [0.0000000000,   0.0000000000,   0.0000000000,   0.0000000000,   0.0000000000,   1.0000000000,   0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000],
        [0.0758317932,   0.0032701974,  0.0032701974,   0.0000000000,   0.0065305891,   0.0000000000,   0.9110972228, 0.0000000000, 0.0000000000, 0.0000000000],
        [0.1257421323,   0.0000000000,   0.0000000000,   0.0393606248,   0.0000000000Mat,   0.0000000000,   0.0000000000, 0.8348972429, 0.0000000000, 0.0000000000],
        [0.1203931080,   0.0051669358,   0.0023516490,   0.0065718465,   0.0007060277,   0.0002353934,   0.0004707359, 0.0002353934, 0.8636335171, 0.0002353934],
        [0.2027418204,   0.0000000000,   0.0000000000,   0.0032480686,   0.0097152118,   0.0000000000,   0.0032480686, 0.0000000000, 0.0064864638, 0.7745603668]])

    '''

    methods['mcpr_years'] = np.array([1950, 1980, 1986, 1992, 1997, 2005, 2010, 2012, 2014, 2015, 2016, 2017, 2018, 2019])

    mcpr_rates = np.array([0.50, 1.0, 2.65, 4.53, 7.01, 7.62, 8.85, 11.3, 14.7, 15.3, 16.5, 18.8, 19, 20])

    methods['trend'] = mcpr_rates[-2] / mcpr_rates  # normalize trend around 2018 so "no method to no method" matrix entry will increase or decrease based on mcpr that year, probs from 2018

    return methods


def default_methods_postpartum():
    '''Function to give probabilities postpartum.
    Probabilities at postpartum month 1 are 1 month transitional probabilities for starting a method after delivery.
    Probabilities at postpartum month 6 are 5 month transitional probabilities for starting or changing methods over
    the first 6 months postpartum.
    Data from Senegal DHS contraceptive calendars, 2017 and 2018 combined
    '''
    methods_postpartum = {}

    methods_postpartum['map'] = {'None': 0,
                      'Pill': 1,
                      'IUDs': 2,
                      'Injectables': 3,
                      'Condoms': 4,
                      'BTL': 5,
                      'Withdrawal': 6,
                      'Implants': 7,
                      'Other traditional' : 8,
                      'Other modern' : 9}

    methods_postpartum['names'] = list(methods_postpartum['map'].keys())

    methods_postpartum['probs_matrix_1'] = {
        '<18': np.array([0.9606501283, 0.0021385800, 0.0004277160, 0.0128314799, 0.0008554320, 0.0000000000, 0.0008554320, 0.0205303678, 0.0017108640, 0.0000000000]),
        '18-20': np.array([0.9524886878, 0.0028280543, 0.0005656109, 0.0214932127, 0.0005656109, 0.0000000000, 0.0005656109, 0.0197963801, 0.0016968326, 0.0000000000]),
        '21-25': np.array([0.9379245283, 0.0083018868, 0.0013207547, 0.0284905660, 0.0009433962, 0.0000000000, 0.0000000000, 0.0177358491, 0.0052830189, 0.0000000000]),
        '>25': np.array([0.9253704535, 0.0102379883, 0.0040413112, 0.0264930400, 0.0007184553, 0.0022451729, 0.0001796138, 0.0267624607, 0.0035922766, 0.0003592277])
    }

    methods_postpartum['probs_matrix_1-6'] = {
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

    methods_postpartum['mcpr_years'] = np.array(
        [1950, 1980, 1986, 1992, 1997, 2005, 2010, 2012, 2014, 2015, 2016, 2017, 2018, 2019])

    mcpr_rates = np.array(
        [0.50, 1.0, 2.65, 4.53, 7.01, 7.62, 8.85, 11.3, 14.7, 15.3, 16.5, 18.8, 19, 20])  # Combintion of DHS data and Track20 data

    methods_postpartum['trend'] = mcpr_rates[
                           -2] / mcpr_rates  # normalize trend around 2018 so "no method to no method" matrix entry will increase or decrease based on mcpr that year, probs from 2018

    return methods_postpartum


def default_efficacy():
    ''' From Guttmacher, fp/docs/gates_review/contraceptive-failure-rates-in-developing-world_1.pdf
    BTL failure rate from general published data
    Pooled efficacy rates for all women in this study: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4970461/
    '''


    method_efficacy = sc.odict({
            "None":        0.0,
            "Pill":        94.5,
            "IUDs":        98.6,
            "Injectable": 98.3,
            "Condoms":     94.6,
            "BTL":         99.5,
            "Withdrawal":  86.6,
            "Implants":    99.4,
            "Other traditional": 86.1, # 1/2 periodic abstinence, 1/2 other traditional approx.  Using rate from periodic abstinence
            "Other modern":   88, # SDM makes up about 1/2 of this, perfect use is 95% and typical is 88%.  EC also included here, efficacy around 85% https://www.aafp.org/afp/2004/0815/p707.html
            })

    # method_efficacy[:] = 100 # To disable contraception

    method_efficacy = method_efficacy[:]/100

    return method_efficacy

def default_efficacy25():
    ''' From Guttmacher, fp/docs/gates_review/contraceptive-failure-rates-in-developing-world_1.pdf
    BTL failure rate from general published data
    Pooled efficacy rates for women ages 25+
    '''


    method_efficacy25 = sc.odict({
            "None":        0.0,
            "Pill":        91.7,
            "IUDs":        96.8,
            "Injectable":  96.5,
            "Condoms":     91.1,
            "BTL":         99.5,
            "Rhythm":       75.4,
            "Withdrawal":   77.3,
            "Implants":     99.4,
            "Other":       94.5,
            })

    # method_efficacy25[:] = 100 # To disable contraception

    method_efficacy25 = method_efficacy25[:]/100

    return method_efficacy25


def default_barriers():
    barriers = sc.odict({
          'No need'   :  54.2,
          'Opposition':  30.5,
          'Knowledge' :   1.7,
          'Access'    :   4.5,
          'Health'    :  12.9,
        })
    barriers[:] /= barriers[:].sum() # Ensure it adds to 1
    return barriers


def default_sexual_activity():
    '''
    Returns a linear interpolation of rates of female sexual activity, defined as
    percentage women who have had sex within the last four weeks.
    From STAT Compiler DHS https://www.statcompiler.com/en/
    Using indicator "Timing of sexual intercourse"
    Includes women who have had sex "within the last four weeks"
    Data taken from 2018 DHS, no trend over years for now
    Onset of sexual activity assumed to be linear from age 10 to first data point at age 15
    '''

    sexually_active = np.array([[0, 5, 10, 15,  20,   25,   30,   35,   40,    45,   50],
                                [0, 0,  0,  11.5, 35.5, 49.6, 57.4, 64.4, 64.45, 64.5, 66.8]])

    sexually_active[1] /= 100 # Convert from percent to rate per woman
    activity_ages = sexually_active[0]
    activity_interp_model = si.interp1d(x=activity_ages, y=sexually_active[1])
    activity_interp = activity_interp_model(fpd.spline_preg_ages)  # Evaluate interpolation along resolution of ages

    return activity_interp


def default_birth_spacing_preference():
    '''
    Returns an array of birth spacing preferences by closest postpartum month.
    Applied to postpartum pregnancy likelihoods.

    NOTE: spacing bins must be uniform!
    '''
    postpartum_spacing = np.array([
        [0, 0.5],
        [3, 0.5],
        [6, 0.5],
        [9, 0.5],
        [12, 0.8],
        [15, 1.2],
        [18, 5.0],
        [21, 5.0],
        [24, 9.0],
        [27, 9.0],
        [30, 9.0],
        [33, 9.0],
        [36, 5.0]])

    # Calculate the intervals and check they're all the same
    intervals = np.diff(postpartum_spacing[:, 0])
    interval = intervals[0]
    assert np.all(intervals == interval), f'In order to be computed in an array, birth spacing preference bins must be equal width, not {intervals}'
    pref_spacing = {}
    pref_spacing['interval'] = interval # Store the interval (which we've just checked is always the same)
    pref_spacing['n_bins'] = len(intervals) # Actually n_bins - 1, but we're counting 0 so it's OK
    pref_spacing['preference'] = postpartum_spacing[:, 1] # Store the actual birth spacing data

    return pref_spacing


def default_sexual_activity_postpartum():
    '''
    Returns an array of monthly likelihood of having resumed sexual activity within 0-35 months postpartum
    Uses DHS Senegal 2018 individual recode (postpartum (v222), months since last birth, and sexual activity within 30 days.
    Limited to 35 months postpartum (can use any limit you want 0-35 max)
    Postpartum month 0 refers to the first month after delivery
    '''

    postpartum_sex = np.array([
        [0, 0.104166667],
        [1, 0.300000000],
        [2, 0.383177570],
        [3, 0.461538462],
        [4, 0.476635514],
        [5, 0.500000000],
        [6, 0.565217391],
        [7, 0.541666667],
        [8, 0.547368421],
        [9, 0.617391304],
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
        [35, 0.644736842]])

    postpartum_activity = {}
    postpartum_activity['month'] = postpartum_sex[:, 0]
    postpartum_activity['percent_active'] = postpartum_sex[:, 1]

    return postpartum_activity

def default_lactational_amenorrhea():
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


def data2interp(data, ages, normalize=False):
    ''' Convert unevently spaced data into an even spline interpolation '''
    model = si.interp1d(data[0], data[1])
    interp = model(ages)
    if normalize:
        interp = np.minimum(1, np.maximum(0, interp))
    return interp


def default_miscarriage_rates():
    '''
    Returns a linear interpolation of the likelihood of a miscarriage
    by age, taken from data from Magnus et al BMJ 2019: https://pubmed.ncbi.nlm.nih.gov/30894356/
    Data to be fed into likelihood of continuing a pregnancy once initialized in model
    Age 0 and 5 set at 100% likelihood.  Age 10 imputed to be symmetrical with probability at age 45 for a parabolic curve
    '''
    miscarriage_rates = np.array([[0,   5, 10,      15,     20,     25,    30,    35,    40,    45,    50],
                                  [1,   1, 0.569,  0.167,   0.112, 0.097,  0.108, 0.167, 0.332, 0.569, 0.569]])
    miscarriage_interp = data2interp(miscarriage_rates, fpd.spline_preg_ages)
    return miscarriage_interp

def default_fecundity_ratio_nullip():
    '''
    Returns an array of fecundity ratios for a nulliparous woman vs a gravid woman
    from PRESTO study: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5712257/
    Approximates primary infertility and its increasing likelihood if a woman has never conceived by age
    '''

    fecundity_ratio_nullip = np.array([[  0,  5,  10, 12.5,  15,  18,  20,   25,   30,   34,   37,   40,   45,   50],
                                        [1,    1,  1,   1,   1,   1,    1, 0.96, 0.95, 0.71, 0.73, 0.42, 0.42, 0.42]])
    fecundity_nullip_interp = data2interp(fecundity_ratio_nullip, fpd.spline_preg_ages)

    return fecundity_nullip_interp


def default_exposure_correction_age():
    '''
    Returns an array of experimental factors to be applied to account for
    residual exposure to either pregnancy or live birth by age.  Exposure to pregnancy will
    increase factor number and residual likelihood of avoiding live birth (mostly abortion,
    also miscarriage), will decrease factor number
    '''

    exposure_correction_age = np.array([[0, 5, 10, 12.5,  15,  18,  20,  25,  30,  35,  40,  45,  50],
                                        [1, 1, 1,   1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    exposure_age_interp = data2interp(exposure_correction_age, fpd.spline_preg_ages)

    return exposure_age_interp


def default_exposure_correction_parity():
    '''
    Returns an array of experimental factors to be applied to account for residual exposure to either pregnancy
    or live birth by parity.

    Michelle note: Thinking about this in terms of child preferences/ideal number of children
    '''
    exposure_correction_parity = np.array([[   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,   12,  20],
                                           [0.5, 1, 1, 1, 1, 1, 1, 0.8, 0.5, 0.3, 0.15, 0.10,  0.05, 0.01]])
    exposure_parity_interp = data2interp(exposure_correction_parity, fpd.spline_parities)
    #
    # exposure_correction_parity = np.array([[   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,   12,  20],
    #                                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1, 1]])
    # exposure_parity_interp = data2interp(exposure_correction_parity, fpd.spline_parities)

    return exposure_parity_interp


def make_pars(configuration_file=None, defaults_file=None):
    pars = sc.dcp(defaults)

    # Complicated parameters
    pars['methods']            = default_methods()
    pars['methods_postpartum'] = default_methods_postpartum()
    pars['methods_postpartum_switch'] = {}
    pars['age_pyramid']        = default_age_pyramid()
    pars['age_mortality']      = default_age_mortality(bound=True)
    pars['age_fecundity']      = default_female_age_fecundity(bound=True)  # Changed to age_fecundity for now from age_fertility for use with LEMOD
    pars['method_efficacy']    = default_efficacy()
    pars['method_efficacy25']  = default_efficacy25()
    pars['barriers']           = default_barriers()
    pars['maternal_mortality'] = default_maternal_mortality()
    pars['infant_mortality']   = default_infant_mortality()
    pars['stillbirth_rate']    = default_stillbirth()
    pars['sexual_activity']    = default_sexual_activity() # Returns linear interpolation of annual sexual activity based on age
    pars['pref_spacing']       = default_birth_spacing_preference()
    pars['sexual_activity_postpartum'] = default_sexual_activity_postpartum() # Returns array of likelihood of resuming sex per postpartum month
    pars['lactational_amenorrhea']     = default_lactational_amenorrhea()
    pars['miscarriage_rates']          = default_miscarriage_rates()
    pars['fecundity_ratio_nullip']     = default_fecundity_ratio_nullip()
    pars['exposure_correction_age']    = default_exposure_correction_age()
    pars['exposure_correction_parity'] = default_exposure_correction_parity()
    pars['exposure_correction'] = 1 # Overall exposure correction factor

    return pars
