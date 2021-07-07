'''
Set the parameters for FPsim, specifically for Senegal.
'''

import os
import numpy as np
import sciris as sc
from scipy import interpolate as si
import fpsim.defaults as fpd


DEFAULT_CONFIGURATION_FILE = os.path.join(os.path.dirname(__file__), 'config.json')
DEFAULTS_FILE = os.path.join(os.path.dirname(__file__), 'defaults.json')


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
    Fecundity rate in 12.5-20 age bins estimated equal to fecundity of 20-25 yr olds
    45-50 age bin estimated at 0.10 of fecundity of 25-27 yr olds, based on fertility rates from Senegal
    '''
    fecundity = {
        'bins': np.array([0., 5, 10, 12.5,  15,    20,     25,   28,  31,   34,   37,  40,   45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99]),
        'f': np.array([0.,    0,  0, 70.8, 70.8, 70.8, 79.3,  77.9, 76.6, 74.8, 67.4, 55.5, 7.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])}
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
    maternal_mortality['probs'] = data[:,2]

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
    infant_mortality['probs'] = data[:,1]/1000   # Rate per 1000 live births

    return infant_mortality


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

    methods['map'] = {
        'None': 0,
        'Pill': 1,
        'IUDs': 2,
        'Injectables': 3,
        'Condoms': 4,
        'BTL': 5,
        'Rhythm': 6,
        'Withdrawal': 7,
        'Implants': 8,
        'Other': 9,
    }

    methods['names'] = list(methods['map'].keys())

    methods['probs_matrix'] = {
        '<18': np.array([
            [0.9946721570, 0.0003284283, 0.0000757999, 0.0019690872, 0.0012121670, 0.0000000000, 0.0001263303, 0.0000000000, 0.0015402303, 0.0000757999],
            [0.5287578181, 0.4069217859, 0.0000000000, 0.0643203960, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000],
            [0.0000000000, 0.0000000000, 1.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000],
            [0.3784797068, 0.0337520371, 0.0000000000, 0.5673896826, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0203785735, 0.0000000000],
            [0.3062747325, 0.0000000000, 0.0000000000, 0.0000000000, 0.6937252675, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000],
            [0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000],
            [0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000, 0.0000000000, 0.0000000000, 0.0000000000],
            [0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000, 0.0000000000, 0.0000000000],
            [0.1975288747, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.8024711253, 0.0000000000],
            [0.0751985122, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.9248014878]]),
        '18-20': np.array([
            [0.9748965935, 0.0030914772, 0.0004220819, 0.0117567491, 0.0028107963, 0.0000000000, 0.0007033791, 0.0000000000, 0.0053343190, 0.0009846038],
            [0.4381271606, 0.4954792608, 0.0000000000, 0.0441103821, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0222831965, 0.0000000000],
            [0.1487728323, 0.0000000000, 0.8512271677, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000],
            [0.4335262316, 0.0186547239, 0.0000000000, 0.5290636135, 0.0093676132, 0.0000000000, 0.0000000000, 0.0000000000, 0.0046939089, 0.0046939089],
            [0.1805180504, 0.0000000000, 0.0000000000, 0.0000000000, 0.8194819496, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000],
            [0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000],
            [0.1075144246, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.8924855754, 0.0000000000, 0.0000000000, 0.0000000000],
            [0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000, 0.0000000000, 0.0000000000],
            [0.1640640890, 0.0000000000, 0.0000000000, 0.0188921157, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.8170437953, 0.0000000000],
            [0.4596399123, 0.0000000000, 0.0000000000, 0.0000000000, 0.0824258398, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.4579342479]]),
        '21-25': np.array([
            [0.9529715288, 0.0091621367, 0.0006319355, 0.0210060937, 0.0027357445, 0.0000000000, 0.0006319355, 0.0001404618, 0.0115969756, 0.0011231879],
            [0.3655280066, 0.5757339128, 0.0000000000, 0.0445607536, 0.0000000000, 0.0000000000, 0.0000000000, 0.0028398987, 0.0084975296, 0.0028398987],
            [0.1050694104, 0.0433144238, 0.8297391932, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0218769726, 0.0000000000],
            [0.3735520653, 0.0258666776, 0.0022740184, 0.5846761633, 0.0045432956, 0.0000000000, 0.0022740184, 0.0000000000, 0.0056761591, 0.0011376024],
            [0.1871644779, 0.0092986546, 0.0000000000, 0.0000000000, 0.7572022368, 0.0000000000, 0.0092986546, 0.0000000000, 0.0185179881, 0.0185179881],
            [0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000],
            [0.1654336539, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.8345663461, 0.0000000000, 0.0000000000, 0.0000000000],
            [0.4392192231, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.5607807769, 0.0000000000, 0.0000000000],
            [0.1347134235, 0.0043496579, 0.0029017027, 0.0043496579, 0.0029017027, 0.0000000000, 0.0000000000, 0.0000000000, 0.8507838553, 0.0000000000],
            [0.3153596058, 0.0000000000, 0.0000000000, 0.0000000000, 0.0430040247, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.6416363695]]),
        '>25': np.array([
            [0.9425899462, 0.0134002260, 0.0030712745, 0.0224482639, 0.0014544504, 0.0001372952, 0.0008235121, 0.0001372952, 0.0145929883, 0.0013447482],
            [0.3012134405, 0.6598026690, 0.0052187409, 0.0217514048, 0.0020905005, 0.0000000000, 0.0010457515, 0.0015682513, 0.0052187409, 0.0020905005],
            [0.0765319596, 0.0057560700, 0.9061903142, 0.0043199069, 0.0000000000, 0.0000000000, 0.0028818424, 0.0000000000, 0.0043199069, 0.0000000000],
            [0.2742752700, 0.0174787294, 0.0049444789, 0.6895757745, 0.0019255229, 0.0000000000, 0.0005504966, 0.0000000000, 0.0087746799, 0.0024750477],
            [0.1527406996, 0.0041074733, 0.0081994759, 0.0163372808, 0.8063081212, 0.0000000000, 0.0000000000, 0.0000000000, 0.0081994759, 0.0041074733],
            [0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000],
            [0.0626421728, 0.0040248143, 0.0040248143, 0.0000000000, 0.0080347744, 0.0000000000, 0.9212734241, 0.0000000000, 0.0000000000, 0.0000000000],
            [0.0928336994, 0.0000000000, 0.0000000000, 0.0474530355, 0.0000000000, 0.0000000000, 0.0000000000, 0.8597132651, 0.0000000000, 0.0000000000],
            [0.1116410740, 0.0058129737, 0.0024513547, 0.0067279749, 0.0003067210, 0.0003067210, 0.0006133558, 0.0003067210, 0.8715263828, 0.0003067210],
            [0.1704237847, 0.0000000000, 0.0000000000, 0.0042053557, 0.0000000000, 0.0000000000, 0.0042053557, 0.0000000000, 0.0083944944, 0.8127710096]])
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
    '''Function to give probabilities postpartum.  Probabilities are transitional probabilities
    over 3 month period'''
    methods_postpartum = {}

    methods_postpartum['map'] = {'None': 0,
                      'Pill': 1,
                      'IUDs': 2,
                      'Injectables': 3,
                      'Condoms': 4,
                      'BTL': 5,
                      'Rhythm': 6,
                      'Withdrawal': 7,
                      'Implants' : 8,
                      'Other' : 9,}

    methods_postpartum['names'] = list(methods_postpartum['map'].keys())

    methods_postpartum['probs_matrix_0-3'] = {
        '<18': np.array([0.9067060959, 0.0092128075, 0.0004863025, 0.0403021519, 0.0014584344, 0.0000000000, 0.0004863025, 0.0009724473, 0.0374600081, 0.0029154501]),
        '18-20': np.array([0.8667850364, 0.0160975636, 0.0032334530, 0.0633518946, 0.0025873208, 0.0000000000, 0.0019409095, 0.0019409095, 0.0421220031, 0.0019409095]),
        '21-25': np.array([0.8416930819, 0.0241524911, 0.0026066166, 0.0805660866, 0.0023895718, 0.0000000000, 0.0019553876, 0.0006520794, 0.0392601957, 0.0067244893]),
        '>25': np.array([0.8217363315, 0.0306566932, 0.0096104032, 0.0716553441, 0.0025619992, 0.0024596033, 0.0028691447, 0.0007178014, 0.0518977116, 0.0058349678])
    }
    methods_postpartum['probs_matrix_4-6'] = np.array([
        [0.9285354464, 0.0139201751, 0.0023609477, 0.0314036722, 0.0028709337,  0.0000638583,  0.0013404539, 0.0002554170, 0.0171432064, 0.0021058894],
        [0.0982865824, 0.8725373947, 0.0000000000, 0.0194295031, 0.0000000000,  0.0000000000,  0.0000000000, 0.0000000000, 0.0097465198, 0.0000000000],
        [0.0437254457, 0.0000000000, 0.9562745543, 0.0000000000, 0.0000000000,  0.0000000000,  0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000],
        [0.0621614125, 0.0085368283, 0.0007132667, 0.9221745182, 0.0007132667,  0.0000000000,  0.0014261942, 0.0000000000, 0.0042745133, 0.0000000000],
        [0.0749675483, 0.0191077585, 0.0191077585, 0.0191077585, 0.8486014178,  0.0000000000,  0.0000000000, 0.0000000000, 0.0191077585, 0.0000000000],
        [0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000,  1.0000000000,  0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000],
        [0.0204075272, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000,  0.0000000000,  0.9795924728, 0.0000000000, 0.0000000000, 0.0000000000],
        [0.0000000000, 0.0714005891, 0.0000000000, 0.0000000000, 0.0000000000,  0.0000000000,  0.0000000000, 0.8571988218, 0.0714005891, 0.0000000000],
        [0.0098647440, 0.0021978014, 0.0000000000, 0.0010993037, 0.0000000000,  0.0000000000,  0.0000000000, 0.0000000000, 0.9868381509, 0.0000000000],
        [0.0095845990, 0.0000000000, 0.0000000000, 0.0095845990, 0.0000000000,  0.0000000000,  0.0000000000, 0.0000000000, 0.0000000000, 0.9808308020]])

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
    Pooled efficacy rates for women ages 25+
    '''


    method_efficacy = sc.odict({
            "None":        0.0,
            "Pill":        95.6,
            "IUDs":        98.9,
            "Injectable": 98.4,
            "Condoms":     94.6,
            "BTL":         99.5,
            "Rhythm":       86.7,
            "Withdrawal":   88.3,
            "Implants":     99.4,
            "Other":       94.5,
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
    Age 12.5 adjusted in calibration to help model match age at first intercourse
    '''

    sexually_active = np.array([[0, 5, 10, 12.5, 15,   18,   20,   25,   30,   35,   40,    45,   50],
                                [0, 0,  0,  8,   11.5, 11.5, 35.5, 49.6, 57.4, 64.4, 64.45, 64.5, 66.8]])

    sexually_active[1] /= 100 # Convert from percent to rate per woman
    activity_ages = sexually_active[0]
    activity_interp_model = si.interp1d(x=activity_ages, y=sexually_active[1])
    activity_interp = activity_interp_model(fpd.spline_preg_ages)  # Evaluate interpolation along resolution of ages

    return activity_interp

def default_sexual_activity_postpartum():
    '''
    Returns an array of monthly likelihood of having resumed sexual activity within 0-36 months postpartum
    Replaces DHS Senegal 2018 calendar data with DHS Senegal 2018 individual recode (postpartum (v222) and this month's sexual activity report)
    Limited to 36 months postpartum (can use any limit you want)
    Changes postpartum_abstinent to postpartum_sex and removes 1- in calculation
    DHS data is reporting on sexual activity in this module (not abstinence)
    '''

    postpartum_sex = np.array([
        [0, 0.0],
        [1, 0.104166667],
        [2, 0.300000000],
        [3, 0.383177570],
        [4, 0.461538462],
        [5, 0.476635514],
        [6, 0.500000000],
        [7, 0.565217391],
        [8, 0.541666667],
        [9,0.547368421],
        [10,0.617391304],
        [11,0.578947368],
        [12,0.637254902],
        [13,0.608247423],
        [14,0.582278481],
        [15,0.542553191],
        [16,0.678260870],
        [17,0.600000000],
        [18,0.605042017],
        [19,0.562500000],
        [20,0.529411765],
        [21,0.674698795],
        [22,0.548780488],
        [23,0.616161616],
        [24,0.709401709],
        [25,0.651376147],
        [26,0.780219780],
        [27,0.717647059],
        [28,0.716417910],
        [29,0.683544304],
        [30,0.716417910],
        [31,0.640625000],
        [32,0.650000000],
        [33,0.676470588],
        [34,0.645161290],
        [35,0.606557377],
        [36,0.644736842]])

    postpartum_activity = {}
    postpartum_activity['month'] = postpartum_sex[:, 0]
    postpartum_activity['percent_active'] = postpartum_sex[:, 1]

    return postpartum_activity


def default_lactational_amenorrhea():
    '''
    Returns an array of the percent of women by month postpartum 0-11 months who meet criteria for LAM:
    Exclusively breastfeeding, menses have not returned.  Extended out 5-11 months to better match data
    as those women continue to be postpartum insusceptible.
    From DHS Senegal calendar data
    '''

    data = np.array([
        [0, 0.875757575757576],
        [1, 0.853658536585366],
        [2, 0.73356401384083],
        [3, 0.627306273062731],
        [4, 0.552188552188552],
        [5, 0.444444444444444],
        [6, 0.250814332247557],
        [7, 0.195046439628483],
        [8, 0.143835616438356],
        [9, 0.108571428571429],
        [10, 0.1],
        [11, 0.0870967741935484],
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
    '''
    miscarriage_rates = np.array([[0, 5, 10,   12.5,   15,     20,   25,    30,    35,    40,    45,    50],
                                  [0, 0,  0,  0.167, 0.167, 0.112, 0.97, 0.108, 0.167, 0.332, 0.569, 0.569]])
    miscarriage_interp = data2interp(miscarriage_rates, fpd.spline_preg_ages)
    return miscarriage_interp


# def default_fecundity_ratio_nullip():
#     '''
#     Returns an array of fecundity ratios for a nulliparous woman vs a gravid woman
#     Help correct for decreased likelihood of conceiving if never conceived
#     from PRESTO study: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5712257/
#     '''
#
#     fecundity_ratio_nullip = np.array([[  0,   5,  10, 12.5,  15,  18,  20,   25,   30,   34,   37,   40,   45,   50],
#                                        [1.0, 1.0, 1.0,  1.0, 1.0, 1.0, 1.0, 0.96, 0.95, 0.71, 0.73, 0.42, 0.42, 0.42]])
#     fecundity_nullip_interp = data2interp(fecundity_ratio_nullip, fpd.spline_preg_ages)
#
#     return fecundity_nullip_interp


def default_fecundity_ratio_nullip():
    '''
    Returns an array of fecundity ratios for a nulliparous woman vs a gravid woman
    Help correct for decreased likelihood of conceiving if never conceived
    Assumes higher rates (20%) of infecundability in LMICs than PRESTO (North America) due to environmental exposures
    Could be adjusted with condom use later (STIs linked to infecundability)
    '''

    fecundity_ratio_nullip = np.array([[  0,   5,  10, 12.5,  15,  18,  20,   25,   30,   34,   37,   40,   45,   50],
                                       [1, 1, 1, 1, 1, 1, 1, 0.768, 0.76, 0.568, 0.584, 0.336, 0.336, 0.336]])
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
                                           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.15, 0.10,  0.05, 0.01]])
    exposure_parity_interp = data2interp(exposure_correction_parity, fpd.spline_parities)
    #
    # exposure_correction_parity = np.array([[   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,   12,  20],
    #                                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1, 1]])
    # exposure_parity_interp = data2interp(exposure_correction_parity, fpd.spline_parities)

    return exposure_parity_interp


def load_configuration_file(configuration_file=None):
    if configuration_file is None:
        configuration_file = DEFAULT_CONFIGURATION_FILE
    try:
        parameters = sc.loadjson(configuration_file)
    except FileNotFoundError as e:
        e.args = (f'Required configuration file: {configuration_file} not found.',)
        raise e
    return parameters


def load_defaults_file(defaults_file=None):
    if defaults_file is None:
        defaults_file = DEFAULTS_FILE
    return load_configuration_file(configuration_file=defaults_file)


def set_fecundity_variation(input_parameters, all_parameters, defaults):
    # Multiplicative range of fertility factors, from confidence intervals from PRESTO study, adjusted from 0.9-1.1 to account for calibration to data
    low = get_parameter(parameters=input_parameters, parameter='fecundity_variation_low', defaults=defaults)
    high = get_parameter(parameters=input_parameters, parameter='fecundity_variation_high', defaults=defaults)
    all_parameters['fecundity_variation'] = [low, high]


def set_pregnancy_duration(input_parameters, all_parameters, defaults):
    # Duration of a pregnancy, in months
    low = get_parameter(parameters=input_parameters, parameter='preg_dur_low', defaults=defaults)
    high = get_parameter(parameters=input_parameters, parameter='preg_dur_high', defaults=defaults)
    all_parameters['preg_dur'] = [low, high]


def set_breastfeeding_duration(input_parameters, all_parameters, defaults):
    # range in duration of breastfeeding per pregnancy, in months
    low = get_parameter(parameters=input_parameters, parameter='breastfeeding_dur_low', defaults=defaults)
    high = get_parameter(parameters=input_parameters, parameter='breastfeeding_dur_high', defaults=defaults)
    all_parameters['breastfeeding_dur'] = [low, high]


def get_parameter(parameters, parameter, defaults):
    value = parameters.pop(parameter, None)
    try:
        default = defaults.pop(parameter)
    except KeyError as e:
        e.args = (f'Unknown input parameter: {parameter} in configuration file.',)
        raise e

    value = default if value is None else value
    return value

# leaving this here for now to keep the default-related comments
# DEFAULT_PARAMETERS = {
#     'name': 'Default',
#     'n': 5000,  # Number of people in the simulation -- for comparing data from Impact 2
#     'start_year': 1950,
#     'end_year': 2015,
#     'timestep': 1, # Timestep in months  DO NOT CHANGE
#     'verbose': True,
#
#     'fecundity_variation_low': 0.3,
#     'fecundity_variation_high': 1.3,
#     'method_age': 15,  # When people start choosing a method
#     'max_age': 99,
#     'preg_dur_low': 9,
#     'preg_dur_high': 9,
#     'switch_frequency': 3,  # Number of months that pass before an agent can select a new method
#     'breastfeeding_dur_low': 1,
#     'breastfeeding_dur_high': 24,
#     'age_limit_fecundity': 50,
#     'postpartum_length': 24,  # Extended postpartum period, for tracking
#     'postpartum_infecund_0-5': 0.65,  # Data from https://www.contraceptionjournal.org/action/showPdf?pii=S0010-7824%2815%2900101-8
#     'postpartum_infecund_6-11': 0.25,
#     'end_first_tri': 3,  # months at which first trimester ends, for miscarriage calculation
#     'abortion_prob': 0.1,
#     'seed': 1  # Random seed, if None, don't reset
# }


def make_pars(configuration_file=None, defaults_file=None):
    input_parameters = load_configuration_file(configuration_file=configuration_file)
    default_parameters = load_defaults_file(defaults_file=defaults_file)

    pars = {}

    #
    # User-tunable parameters
    #

    # parameters that require a bit of code to set
    set_fecundity_variation(input_parameters=input_parameters, all_parameters=pars, defaults=default_parameters)
    set_pregnancy_duration(input_parameters=input_parameters, all_parameters=pars, defaults=default_parameters)
    set_breastfeeding_duration(input_parameters=input_parameters, all_parameters=pars, defaults=default_parameters)

    ###
    # TODO: Finish porting these over to use input parameters
    ###

    # Complicated parameters
    pars['methods']            = default_methods()
    pars['methods_postpartum'] = default_methods_postpartum()
    pars['age_pyramid']        = default_age_pyramid()
    pars['age_mortality']      = default_age_mortality(bound=True)
    pars['age_fecundity']      = default_female_age_fecundity(bound=True)  # Changed to age_fecundity for now from age_fertility for use with LEMOD
    pars['method_efficacy']    = default_efficacy()
    pars['method_efficacy25'] = default_efficacy25()
    pars['barriers']           = default_barriers()
    pars['maternal_mortality'] = default_maternal_mortality(get_parameter(parameters=input_parameters, parameter="maternal_mortality_multiplier", defaults=default_parameters))
    pars['infant_mortality']   = default_infant_mortality()
    pars['sexual_activity']    = default_sexual_activity() # Returns linear interpolation of annual sexual activity based on age
    pars['sexual_activity_postpartum'] = default_sexual_activity_postpartum() # Returns array of likelihood of resuming sex per postpartum month
    pars['lactational_amenorrhea']     = default_lactational_amenorrhea()
    pars['miscarriage_rates']          = default_miscarriage_rates()
    pars['fecundity_ratio_nullip']     = default_fecundity_ratio_nullip()
    pars['exposure_correction_age']    = default_exposure_correction_age()
    pars['exposure_correction_parity'] = default_exposure_correction_parity()
    pars['exposure_correction'] = 1 # Overall exposure correction factor

    ###
    # END TODO
    ###

    # finish consuming all remaining input parameters
    for input_parameter in list(input_parameters.keys()):
        pars[input_parameter] = get_parameter(parameters=input_parameters, parameter=input_parameter, defaults=default_parameters)

    # now consume all remaining parameters NOT in the input parameters (use defaults, only)
    pars.update(default_parameters)

    return pars
