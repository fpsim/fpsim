'''
Set the parameters for LEMOD-FP.
'''

import os
import pylab as pl
import sciris as sc
import pandas as pd
from scipy import interpolate as si

resolution = 100
max_age = 99
max_age_preg = 50

#%% Helper function

def abspath(path):
    cwd = os.path.abspath(os.path.dirname(__file__))
    output = os.path.join(cwd, path)
    return output



#%% Set parameters for the simulation

def default_age_pyramid():
    ''' Starting age bin, male population, female population ''' 
    # Based on Senegal 1982
    # pyramid = pl.array([[0,  579035, 567499],
    #                     [5,  459255, 452873],
    #                     [10, 364432, 359925],
    #                     [15, 294589, 292235],
    #                     [20, 239825, 241363],
    #                     [25, 195652, 198326],
    #                     [30, 155765, 158950],
    #                     [35, 135953, 137097],
    #                     [40, 124615, 122950],
    #                     [45, 107622, 106116],
    #                     [50,  89533, 89654],
    #                     [55,  70781, 73290],
    #                     [60,  52495, 57330],
    #                     [65,  36048, 41585],
    #                     [70,  21727, 26383],
    #                     [75,  10626, 13542],
    #                     [80,   4766,  6424]])
    
    pyramid = pl.array([ [0,  318225,  314011], # Senegal 1962
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
    

def default_age_mortality():
    ''' Age-dependent mortality rates, Senegal specific from 1990-1995 -- see age_dependent_mortality.py in the fp_analyses repository
    Mortality rate trend from crude mortality rate per 1000 people: https://data.worldbank.org/indicator/SP.DYN.CDRT.IN?locations=SN
    '''
    mortality = {
            'bins': pl.array([ 0.,  5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]),
            'm': pl.array([0.03075891, 0.00266326, 0.00164035, 0.00247776, 0.00376541,0.00377009, 0.00433534, 0.00501743, 0.00656144, 0.00862479, 0.01224844, 0.01757291, 0.02655129, 0.0403916 , 0.06604032,0.10924413, 0.17495116, 0.26531436, 0.36505174, 0.43979833]),
            'f': pl.array([0.02768283, 0.00262118, 0.00161414, 0.0023998 , 0.00311697, 0.00354105, 0.00376715, 0.00429043, 0.00503436, 0.00602394, 0.00840777, 0.01193858, 0.01954465, 0.03220238, 0.05614077, 0.0957751 , 0.15973906, 0.24231313, 0.33755308, 0.41632442])
            }

    mortality['years'] = pl.array([1950., 1955, 1960, 1965, 1970, 1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025, 2030]) # Starting year bin
    mortality['trend'] = pl.array([28,    27,    26.023, 25.605, 24.687, 20.995, 16.9, 13.531, 11.335, 11.11, 10.752, 9.137, 7.305, 6.141, 5.7, 5.7, 5.7]) # First 2 estimated, last 3 are projected
    mortality['trend'] /= mortality['trend'][8]
    return mortality


# def default_age_year_fertility():
#     ''' From WPP2019_FERT_F07_AGE_SPECIFIC_FERTILITY.xlsx, filtered on Senegal '''
#     fertility = {}
#     fertility['ages'] = [0, 15, 20, 25, 30, 35, 40, 45, 50] # Starting age bin
#     fertility['years'] = [1950, 1955, 1960, 1965, 1970, 1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015], # Starting year bin
#     fertility['data'] = [
#         [0, 195.3,   297.1,   310.0,   260.9,   185.7,   78.1,    32.9, 0],
#         [0, 194.5,   296.1,   303.4,   262.7,   197.3,   92.5,    33.6, 0],
#         [0, 195.3,   299.0,   301.8,   267.6,   210.6,   110.4,   35.2, 0],
#         [0, 192.2,   300.5,   302.6,   270.9,   219.3,   126.8,   37.7, 0],
#         [0, 184.6,   296.6,   302.9,   270.6,   221.3,   134.5,   39.5, 0],
#         [0, 176.8,   292.8,   304.4,   273.3,   224.4,   137.4,   40.9, 0],
#         [0, 166.9,   289.2,   304.5,   278.8,   230.1,   138.6,   41.8, 0],
#         [0, 142.9,   263.8,   283.5,   264.0,   218.2,   129.4,   38.1, 0],
#         [0, 123.1,   241.6,   267.4,   250.9,   204.4,   119.3,   33.3, 0],
#         [0, 109.7,   222.7,   252.7,   236.9,   186.3,   104.5,   27.1, 0],
#         [0, 100.3,   207.8,   239.1,   222.8,   168.9,   89.4,    21.7, 0],
#         [0, 93.5,    202.7,   236.2,   218.9,   163.6,   84.6,    20.5, 0],
#         [0, 83.6,    195.2,   234.2,   214.1,   162.9,   87.4,    22.7, 0],
#         [0, 72.7,    180.2,   220.7,   200.0,   152.3,   82.1,    22.0, 0],
#         ]
#     return fertility


def default_age_fertility():
    ''' Less-developed countries, WPP2019_MORT_F15_3_LIFE_TABLE_SURVIVORS_FEMALE.xlsx, 1990-1995 '''
    f15 = 0.1 # Adjustment factor for women aged 15-20
    f20 = 0.5 # Adjustment factor for women aged 20-25
    fertility = {
            'bins': pl.array([ 0.,  5, 10,         15,         20,     25,     30,     35,      40,       45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]), 
            # 'f':    pl.array([ 0,  0,  0, f15*0.0706, f20*0.0196, 0.0180, 0.0115, 0.00659, 0.00304, 0.00091,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0])}
            'f':    pl.array([ 0.,  0,  0,   72.7,  180.2,  220.7,  200.0,   152.3,    82.1,    22.0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0])}
    fertility['f'] /= 1000 # Births per thousand to births per woman
    fertility['m'] = 0*fertility['f'] # Men don't have fertility -- probably could be handled differently!
    fertility['years'] = pl.array([1950., 1955, 1960, 1965, 1970, 1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025, 2030]) # Starting year bin
    fertility['trend'] = pl.array([194.3, 197.1, 202.9, 207.1, 207.1, 207.1, 207.1, 191.4, 177.1, 162.9, 150.0, 145.7, 142.9, 132.9, 125, 120, 115]) # Last 3 are projected!!
    fertility['trend'] /= fertility['trend'][-1]


    '''
    Change to fecundity rate from PRESTO study: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5712257/
    Fecundity rate in 15-20 age bin estimated at 0.329 of fecundity of 25-27 yr olds, based on fertility data above
    45-50 age bin also estimated at 0.10 of fecundity of 25-27 yr olds, based on fertility data
    '''
    f15 = 0.1  # Adjustment factor for women aged 15-20
    f20 = 0.5  # Adjustment factor for women aged 20-25
    fecundity = {
        'bins': pl.array([0., 5, 10, 12.5,  15,    20,     25,   28,  31,   34,   37,  40,   45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99]),
        'f': pl.array([0.,    0,  0, 70.8, 70.8, 70.8, 79.3,  77.9, 76.6, 74.8, 67.4, 55.5, 7.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])}
    fecundity['f'] /= 100  # Conceptions per hundred to conceptions per woman over 12 menstrual cycles of trying to conceive
    fecundity['m'] = 0 * fecundity['f']

    return fecundity


def default_maternal_mortality():
    '''
    Maternal mortality ratio (MMR) for Senegal.  Data from World Bank
    # of maternal deaths per 100,000 live births
    Maternal deaths: The annual number of female deaths from any cause related to or aggravated by pregnancy
    or its management (excluding accidental or incidental causes) during pregnancy and childbirth or within
    42 days of termination of pregnancy, irrespective of the duration and site of the pregnancy,
    expressed per 100,000 live births, for a specified time period.
    '''

    data = pl.array([
        [1990, 600],  # Estimated value from overall maternal mortality ratio from World Bank
        [2000, 553],
        [2001, 545],
        [2002, 537],
        [2003, 532],
        [2004, 526],
        [2005, 519],
        [2006, 514],
        [2007, 504],
        [2008, 492],
        [2009, 472],
        [2010, 447],
        [2011, 423],
        [2012, 400],
        [2013, 381],
        [2014, 364],
        [2015, 346],
        [2016, 330],
        [2017, 315],
    ])

    maternal_mortality = {}

    maternal_mortality['year'] = data[:,0]
    maternal_mortality['probs'] = data[:,1]/1e5

    return maternal_mortality


def default_infant_mortality():
    '''
    From World Bank indicators for infant morality (< 1 year) for Senegal, per 1000 live births
    From API_SP.DYN.IMRT.IN_DS2_en_excel_v2_1495452.numbers
    '''
    
    data = pl.array([
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
    
    methods['matrix'] = pl.array([
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
    
    methods['mcpr_years'] = pl.array([1950, 1980, 1986, 1992, 1997, 2005, 2010, 2012, 2014, 2015, 2016, 2017])
    
    mcpr_rates = pl.array([0.50, 1.0, 2.65, 4.53, 7.01, 7.62, 8.85, 11.3, 14.7, 15.3, 16.5, 18.8])
    # mcpr_rates /= 100
    # methods['mcpr_multipliers'] = (1-mcpr_rates)**2.0

    methods['trend'] = mcpr_rates[5]/mcpr_rates # normalize trend around 2005 so "no method to no method" matrix entry will increase or decrease based on mcpr that year
    
    methods['mcpr_multipliers'] = 10/mcpr_rates # No idea why it should be this...

    return methods
'''

def default_methods():
    methods = {}

    methods['map'] = {'None': 0,
                                 'Pill': 1,
                                 'IUDs': 2,
                                 'Injectables': 3,
                                 'Condoms': 4,
                                 'BTL': 5,
                                 'Rhythm': 6,
                                 'Withdrawal': 7,
                                 'Implants': 8,
                                 'Other': 9, }

    methods['names'] = list(methods['map'].keys())

    methods['probs_matrix'] = {
        '<18': pl.array([
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
        '18-20': pl.array([
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
        '21-25': pl.array([
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
        '>25': pl.array([
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
    methods['probs_matrix'] = pl.array([
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

    methods['mcpr_years'] = pl.array([1950, 1980, 1986, 1992, 1997, 2005, 2010, 2012, 2014, 2015, 2016, 2017])

    mcpr_rates = pl.array([0.50, 1.0, 2.65, 4.53, 7.01, 7.62, 8.85, 11.3, 14.7, 15.3, 16.5, 18.8])

    methods['trend'] = mcpr_rates[-1] / mcpr_rates  # normalize trend around 2017 so "no method to no method" matrix entry will increase or decrease based on mcpr that year, probs from 2018

    return methods

def default_methods_postpartum():
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

    methods_postpartum['probs_matrix_0-3'] = pl.array([0.84037327035, 0.02529892504, 0.00622658597,	0.06968071414, 0.00239363081, 0.00133645154, 0.00228238393,	0.00083542184, 0.04617844485, 0.00539417152])

    methods_postpartum['probs_matrix_4-6'] = pl.array([
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


    methods_postpartum['mcpr_years'] = pl.array([1950, 1980, 1986, 1992, 1997, 2005, 2010, 2012, 2014, 2015, 2016, 2017])

    mcpr_rates = pl.array([0.50, 1.0, 2.65, 4.53, 7.01, 7.62, 8.85, 11.3, 14.7, 15.3, 16.5, 18.8])

    methods_postpartum['trend'] = mcpr_rates[-1] / mcpr_rates  # normalize trend around 2005 so "no method to no method" matrix entry will increase or decrease based on mcpr that year

    return methods_postpartum
    
def default_efficacy():
    ''' From Guttmacher, fp/docs/gates_review/contraceptive-failure-rates-in-developing-world_1.pdf
    BTL failure rate from general published data'''

    method_efficacy = sc.odict({
            "None":        0.0,
            "Pill":        94.5,
            "IUDs":        98.6,
            "Injectable": 98.3,
            "Condoms":     94.6,
            "BTL":         99.5,
            "Rhythm":       86.1,
            "Withdrawal":   86.6,
            "Implants":     99.4,
            "Other":       94.5,
            })

    # method_efficacy[:] = 100 # To disable contraception
    
    method_efficacy = method_efficacy[:]/100
    
    # for key,value in method_efficacy.items():
    #     method_efficacy[key] = method_efficacy[key]/100
    
    # assert method_efficacy.keys() == default_methods() # Ensure ordering
    
    return method_efficacy


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
    percentage women who have had sex within the last year.
    From STAT Compiler DHS https://www.statcompiler.com/en/
    Using indicator "Timing of sexual intercourse"
    Includes women who have had sex "within the last four weeks" or "within the last year"
    For age 15 and 18, uses indicator "percent women who have had sex by exact age".
    Data taken from 2018 DHS, no trend over years for now
    '''


    sexually_active = pl.array([[0, 5, 10, 12.5, 15,  18,   20,   25,  30, 35, 40,    45, 50],
                                [0, 0,  0, 8, 10, 35.4, 50.2, 78.9, 80, 83, 88.1, 82.6, 82.6]])
    sexually_active[1] /= 100 # Convert from percent to rate per woman
    ages = pl.arange(resolution * max_age_preg + 1) / resolution
    activity_ages = sexually_active[0]
    activity_interp_model = si.interp1d(x=activity_ages, y=sexually_active[1])
    activity_interp = activity_interp_model(ages)  # Evaluate interpolation along resolution of ages

    return activity_interp

def default_sexual_activity_postpartum():
    '''
    Returns an array of monthly likelihood of having resumed sexual activity within 0-11 months postpatum
    From DHS Senegal 2018 calendar data
    '''

    postpartum_abstinent = pl.array([
        [0, 1.0],
        [1, 0.986062717770035],
        [2, 0.868512110726644],
        [3, 0.653136531365314],
        [4, 0.511784511784512],
        [5, 0.383512544802867],
        [6, 0.348534201954397],
        [7, 0.340557275541796],
        [8, 0.291095890410959],
        [9, 0.254285714285714],
        [10, 0.223529411764706],
        [11, 0.22258064516129],
    ])

    postpartum_activity = {}
    postpartum_activity['month'] = postpartum_abstinent[:, 0]
    postpartum_activity['percent_active'] = 1 - postpartum_abstinent[:, 1]

    return postpartum_activity


def default_lactational_amenorrhea():
    '''
    Returns an array of the percent of women by month postpartum 0-11 months who meet criteria for LAM:
    Exclusively breastfeeding, menses have not returned.  Extended out 5-11 months to better match data
    as those women continue to be postpartum insusceptible.
    From DHS Senegal calendar data
    '''

    data = pl.array([
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
    lactational_amenorrhea['percent'] = data[:, 1]

    return lactational_amenorrhea

def default_miscarriage_rates():
    '''
    Returns a linear interpolation of the likelihood of a miscarriage
    by age, taken from data from Magnus et al BMJ 2019: https://pubmed.ncbi.nlm.nih.gov/30894356/
    Data to be fed into likelihood of continuing a pregnancy once initialized in model
    '''

    miscarriage_rates = pl.array([[0, 5, 10, 12.5, 15,   20,   25,   30,   35,  40,   45,    50],
                                  [0, 0, 0,  16.7, 16.7, 11.2, 9.7, 10.8, 16.7, 33.2, 56.9, 56.9]])
    miscarriage_ages = miscarriage_rates[0]
    miscarriage_rates[1] /= 100
    ages = pl.arange(resolution * max_age_preg + 1) / resolution
    miscarriage_interp_model = si.interp1d(miscarriage_ages, miscarriage_rates[1])
    miscarriage_interp = miscarriage_interp_model(ages)
    miscarriage_interp = pl.minimum(1, pl.maximum(0, miscarriage_interp))

    return miscarriage_interp

def default_fecundity_ratio_nullip():
    '''
    Returns an array of fecundity ratios for a nulliparous woman vs a gravid woman
    Help correct for decreased likelihood of conceiving if never conceived
    from PRESTO study: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5712257/
    '''

    fecundity_ratio_nullip = pl.array([[0,    5,   10, 12.5, 15,   18,  20,  25,   30,   34,   37,  40, 45, 50],
                                       [1.0, 1.0, 1.0,  1.0, 1.0, 1.0, 1.0, 0.96, 0.95, 0.71, 0.73, 0.42, 0.42, 0.42]])

    return fecundity_ratio_nullip


def default_exposure_correction_age():
    '''
    Returns an array of experimental factors to be applied to account for
    residual exposure to either pregnancy or live birth by age.  Exposure to pregnancy will
    increase factor number and residual likelihood of avoiding live birth (mostly abortion,
    also miscarriage), will decrease factor number
    '''

    exposure_correction_age = pl.array([[0,        5,       10,      12.5,       15,          18,       20,          25,         30,        35,           40,       45,          50],
                                        [[1, 1], [1, 1], [1, 1], [1.0, 1.0], [1.4, 1.4], [1.5, 1.5], [1.6, 1.6], [1.8, 1.8], [1.6, 1.6], [1.4, 1.4], [1.2, 1.2], [0.8, 0.8], [0.1, 0.1]]])

    return exposure_correction_age

def default_exposure_correction_parity():
    '''
    Returns an array of experimental factors to be applied to account for residual exposure to either pregnancy
    or live birth by parity.
    '''

    exposure_correction_parity = pl.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                                           [0.5, 1.2, 1.0, 0.6, 0.6, 0.4, 0.4, 0.4, 0.2, 0.2, 0.1, 0.1, 0.1]])

    return exposure_correction_parity

def make_pars():
    pars = {}

    # User-tunable parameters
    pars['fertility_variation'] = [0.3, 1.3]  # Multiplicative range of fertility factors, from confidence intervals from PRESTO study, adjusted from 0.9-1.1 to account for calibration to data
    pars['method_age'] = 15  # When people start choosing a method
    pars['max_age'] = 99
    pars['preg_dur'] = [9, 9]  # Duration of a pregnancy, in months
    pars['switch_frequency'] = 12 # Number of months that pass before an agent can select a new method (selects on non-fractional years in timestep)
    pars['breastfeeding_dur'] = [1, 24]  # range in duration of breastfeeding per pregnancy, in months
    pars['age_limit_fecundity'] = 50
    pars['postpartum_length'] = 24  # Extended postpartum period, for tracking
    pars['end_first_tri'] = 3  # months at which first trimester ends, for miscarriage calculation
    pars['abortion_prob'] = 0.10
    pars['twins_prob'] = 0.018

    # Simulation parameters
    pars['name'] = 'Default' # Name of the simulation
    pars['n'] = 500*10 # Number of people in the simulation -- for comparing data from Impact 2
    pars['start_year'] = 1960
    pars['end_year'] = 2018
    pars['timestep'] = 1 # Timestep in months  DO NOT CHANGE
    pars['verbose'] = True
    pars['seed'] = 1 # Random seed, if None, don't reset
    
    # Complicated parameters
    pars['methods']            = default_methods()
    pars['methods_postpartum'] = default_methods_postpartum()
    pars['age_pyramid']        = default_age_pyramid()
    pars['age_mortality']      = default_age_mortality()
    pars['age_fertility']      = default_age_fertility()  # Changed to age_fecundity for now from age_fertility for use with LEMOD
    pars['method_efficacy']    = default_efficacy()
    pars['barriers']           = default_barriers()
    pars['maternal_mortality'] = default_maternal_mortality()
    pars['infant_mortality']    = default_infant_mortality()
    pars['sexual_activity']    = default_sexual_activity() # Returns linear interpolation of annual sexual activity based on age
    pars['sexual_activity_postpartum'] = default_sexual_activity_postpartum() # Returns array of likelihood of resuming sex per postpartum month
    pars['lactational_amenorrhea'] = default_lactational_amenorrhea()
    pars['miscarriage_rates']  = default_miscarriage_rates()
    pars['fecundity_ratio_nullip'] = default_fecundity_ratio_nullip()
    pars['exposure_correction_age']= default_exposure_correction_age()
    pars['exposure_correction_parity'] = default_exposure_correction_parity()

    # Population size array from Senegal data for plotting in sim
    pars['pop_years']          = pl.array([1982., 1983., 1984., 1985., 1986., 1987., 1988., 1989., 1990.,
       1991., 1992., 1993., 1994., 1995., 1996., 1997., 1998., 1999.,
       2000., 2001., 2002., 2003., 2004., 2005., 2006., 2007., 2008.,
       2009., 2010., 2011., 2012., 2013., 2014., 2015.])
    pars['pop_size']           = pl.array([18171.28189616, 18671.90398192, 19224.80410006, 19831.90843354,
       20430.1523254 , 21102.60419936, 21836.03759909, 22613.45931046,
       23427.16460161, 24256.32216052, 25118.86689077, 26016.96753169,
       26959.59866908, 27950.59840084, 28881.13025563, 29867.46434421,
       30902.09568706, 31969.90018092, 33062.81639339, 34143.09113842,
       35256.73878676, 36402.87472106, 37583.29671208, 38797.07733839,
       39985.26113334, 41205.93321424, 42460.86281582, 43752.50403785,
       45081.05663263, 46401.31950667, 47773.28969221, 49185.05339094,
       50627.82150134, 52100.32416946])

    return pars