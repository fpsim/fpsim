'''
Set the parameters for LEMOD-FP.
'''

import json
import os
import pylab as pl
import sciris as sc
import pandas as pd
from scipy import interpolate as si


DEFAULT_CONFIGURATION_FILE = os.path.join(os.path.dirname(__file__), 'config.json')
DEFAULTS_FILE = os.path.join(os.path.dirname(__file__), 'defaults.json')

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
    ''' From Impact 2 (?) <- Not sure what this refers to
    I believe this is a maternal mortality ratio (MMR)
    # of maternal deaths per 100,000 live births
    Matches this data: https://knoema.com/WBWDI2019Jan/world-development-indicators-wdi
    From WHO website:  https://www.who.int/data/gho/indicator-metadata-registry/imr-details/26
    Maternal deaths: The annual number of female deaths from any cause related to or aggravated by pregnancy
    or its management (excluding accidental or incidental causes) during pregnancy and childbirth or within
    42 days of termination of pregnancy, irrespective of the duration and site of the pregnancy,
    expressed per 100,000 live births, for a specified time period.
    '''
    maternal_mortality = {}
    maternal_mortality['years'] = pl.array([1985., 1990, 1995, 2000, 2005, 2010, 2015])
    maternal_mortality['probs'] = pl.array([711.,   540, 509,  488,  427,  375,  315])
    maternal_mortality['probs'] /= 1e5
    return maternal_mortality


def default_child_mortality():
    ''' From "When and Where Birth Spacing Matters for Child Survival: An International Comparison Using the DHS", Fig. 3 '''
    
    data = pl.array([
            [13.3032, 0.1502],
            [14.0506, 0.1456],
            [15.1708, 0.1384],
            [16.1011, 0.1311],
            [17.0384, 0.1265],
            [18.1615, 0.1203],
            [20.0332, 0.11],
            [22.0963, 0.1002],
            [24.1651, 0.0925],
            [24.1637, 0.092],
            [24.3522, 0.0915],
            [24.5408, 0.0909],
            [24.5394, 0.0904],
            [24.728, 0.0899],
            [24.7266, 0.0894],
            [24.7252, 0.0889],
            [24.9138, 0.0884],
            [28.1181, 0.0791],
            [32.0809, 0.0693],
            [38.1334, 0.0594],
            [44.0057, 0.0532],
            [50.075, 0.0496],
            [60.1336, 0.0459],
            [70.0023, 0.0422],
            [80.0567, 0.037],
        ])
    
    child_mortality = {}
    child_mortality['space'] = data[:,0]/12.0
    child_mortality['probs'] = data[:,1]
    
    return child_mortality

    
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
    
    methods['mcpr_multipliers'] = 10/mcpr_rates # No idea why it should be this...
    
    
    return methods


    
def default_efficacy():
    ''' From Guttmacher, fp/docs/gates_review/contraceptive-failure-rates-in-developing-world_1.pdf '''
    
    # Expressed as failure rates
    method_efficacy = sc.odict({
            "None":        0.0,
            "Lactation":   90.0,
            "Implants":    99.4,
            "Injectables": 98.3,
            "IUDs":        98.6,
            "Pill":        94.5,
            "Condoms":     94.6,
            "Other":       94.5,
            "Traditional": 86.6,
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

    exposure_correction_age =   pl.array([[0,       5,     10,      12.5,        15,          18,        20,        25,        30,        35,           40,        45,          50],
                                           [[1, 1], [1, 1], [1, 1], [1.9, 1.9], [2.8, 2.8], [2.8, 2.8], [3.1, 3.1], [3.6, 3.6], [3.1, 3.1], [2.7, 2.7], [2.4, 2.4], [1.6, 1.6], [0.1, 0.1]]])

    return exposure_correction_age


def default_exposure_correction_parity():
    '''
    Returns an array of experimental factors to be applied to account for residual exposure to either pregnancy
    or live birth by parity.
    '''

    exposure_correction_parity = pl.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                                       [0.25, 0.6, 0.5, 0.3, 0.3, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1]])

    return exposure_correction_parity


def load_configuration_file(configuration_file=None):
    if configuration_file is None:
        configuration_file = DEFAULT_CONFIGURATION_FILE
    try:
        with open(configuration_file, 'r') as f:
            parameters = json.load(f)
    except FileNotFoundError as e:
        e.args = (f'Required configuration file: {configuration_file} not found.',)
        raise e
    return parameters


def load_defaults_file(defaults_file=None):
    if defaults_file is None:
        defaults_file = DEFAULTS_FILE
    return load_configuration_file(configuration_file=defaults_file)


def set_fertility_variation(input_parameters, all_parameters, defaults):
    # Multiplicative range of fertility factors, from confidence intervals from PRESTO study, adjusted from 0.9-1.1 to account for calibration to data
    low = get_parameter(parameters=input_parameters, parameter='fertility_variation_low', defaults=defaults)
    high = get_parameter(parameters=input_parameters, parameter='fertility_variation_high', defaults=defaults)
    all_parameters['fertility_variation'] = [low, high]


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
#     'fertility_variation_low': 0.3,
#     'fertility_variation_high': 1.3,
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
    set_fertility_variation(input_parameters=input_parameters, all_parameters=pars, defaults=default_parameters)
    set_pregnancy_duration(input_parameters=input_parameters, all_parameters=pars, defaults=default_parameters)
    set_breastfeeding_duration(input_parameters=input_parameters, all_parameters=pars, defaults=default_parameters)

    ###
    # TODO: Finish porting these over to use input parameters
    ###

    # Complicated parameters
    pars['methods']            = default_methods()
    pars['age_pyramid']        = default_age_pyramid()
    pars['age_mortality']      = default_age_mortality()
    pars['age_fertility']      = default_age_fertility()  # Changed to age_fecundity for now from age_fertility for use with LEMOD
    pars['method_efficacy']    = default_efficacy()
    pars['barriers']           = default_barriers()
    pars['maternal_mortality'] = default_maternal_mortality()
    pars['child_mortality']    = default_child_mortality()
    pars['sexual_activity']    = default_sexual_activity() # Returns linear interpolation of sexual activity
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

    ###
    # END TODO
    ###

    # finish consuming all remaining input parameters
    for input_parameter in list(input_parameters.keys()):
        pars[input_parameter] = get_parameter(parameters=input_parameters, parameter=input_parameter, defaults=default_parameters)

    # now consume all remaining parameters NOT in the input parameters (use defaults, only)
    pars.update(default_parameters)

    return pars
