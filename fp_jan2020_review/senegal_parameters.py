import os
import pylab as pl
import pandas as pd
import sciris as sc

#%% Parameters for the calibration etc.

def abspath(path):
    cwd = os.path.abspath(os.path.dirname(__file__))
    output = os.path.join(cwd, path)
    return output

# pop_pyr_1982_fn = abspath('data/senegal-population-pyramid-1982.csv')
popsize_tfr_fn = abspath('data/senegal-popsize-tfr.csv')

# Load data
# pop_pyr_1982 = pd.read_csv(pop_pyr_1982_fn)
popsize_tfr  = pd.read_csv(popsize_tfr_fn, header=None)

# Handle population size
scale_factor = 1000
years = popsize_tfr.iloc[0,:].to_numpy()
popsize = popsize_tfr.iloc[1,:].to_numpy() / scale_factor




#%% Set parameters for the simulation

def default_age_pyramid():
    # ''' Starting age bin, male population, female population -- based on Senegal 1982 '''
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
                    ])
    
    return pyramid
    

def default_age_mortality():
    ''' Age-dependent mortality rates -- see age_dependent_mortality.py in the fp_analyses repository '''
    mortality = sc.odict([
            ('bins', pl.array([ 0,  5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95])), 
            ('m', pl.array([0.01365168, 0.00580404, 0.00180847, 0.0012517 , 0.00171919, 0.00226466, 0.00258822, 0.00304351, 0.00377434, 0.00496091, 0.00694581, 0.01035062, 0.01563918, 0.02397286, 0.03651509,0.05578357, 0.08468156, 0.12539009, 0.17939655, 0.24558742])), 
            ('f', pl.array([0.01213076, 0.00624896, 0.0017323 , 0.00114656, 0.00143726, 0.00175446, 0.00191577, 0.00214836, 0.00251644, 0.00315836, 0.00439748, 0.00638658, 0.00965512, 0.01506286, 0.02361487, 0.03781285, 0.06007898, 0.09345669, 0.14091699, 0.20357825])), ])
    return mortality


def default_age_fertility():
    ''' Less-developed countries, WPP2019_MORT_F15_3_LIFE_TABLE_SURVIVORS_FEMALE.xlsx, 1990-1995 '''
    f15 = 1#0.003 # Adjustment factor for women aged 15-20
    f20 = 1#0.25 # Adjustment factor for women aged 20-25
    fertility = sc.odict([
            ('bins', pl.array([ 0,  5, 10,      15,     20,     25,     30,     35,      40,       45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95])), 
            ('f',    pl.array([ 0,  0,  0, f15*0.0706, f20*0.0196, 0.0180, 0.0115, 0.00659, 0.00304, 0.00091,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]))])
    fertility['m'] = 0*fertility['f'] # Men don't have fertility -- probably could be handled differently!
    return fertility


def default_methods():
    return ['None', 'Lactation', 'Implants', 'Injectables', 'IUDs', 'Pill', 'Condoms', 'Other', 'Traditional']


    
def default_switching():
    
    matrix = pl.array([
       [8.81230657e-01, 0.00000000e+00, 9.56761433e-04, 1.86518124e-03,        1.44017774e-04, 8.45978530e-04, 1.80273996e-04, 1.61138768e-05,        1.46032008e-04],
       [2.21565806e-05, 1.52074712e-04, 2.01423460e-06, 3.02135189e-06,        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,        0.00000000e+00],
       [3.45441233e-04, 0.00000000e+00, 3.29206502e-02, 1.61138768e-05,        5.03558649e-06, 1.40996422e-05, 2.01423460e-06, 0.00000000e+00,        1.00711730e-06],
       [1.22767599e-03, 0.00000000e+00, 3.02135189e-05, 4.28810403e-02,        1.40996422e-05, 6.94910936e-05, 8.05693838e-06, 0.00000000e+00,        6.04270379e-06],
       [3.52491054e-05, 0.00000000e+00, 2.01423460e-06, 3.02135189e-06,        6.10715929e-03, 5.03558649e-06, 0.00000000e+00, 0.00000000e+00,        0.00000000e+00],
       [6.33476780e-04, 0.00000000e+00, 1.30925249e-05, 5.03558649e-05,        6.04270379e-06, 1.97092855e-02, 4.02846919e-06, 1.00711730e-06,        6.04270379e-06],
       [8.35907357e-05, 0.00000000e+00, 5.03558649e-06, 6.04270379e-06,        2.01423460e-06, 3.02135189e-06, 3.94689269e-03, 2.01423460e-06,        1.00711730e-06],
       [1.20854076e-05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,        0.00000000e+00, 0.00000000e+00, 3.02135189e-06, 1.74432716e-03,        1.00711730e-06],
       [4.93487476e-05, 0.00000000e+00, 2.01423460e-06, 4.02846919e-06,        0.00000000e+00, 2.01423460e-06, 0.00000000e+00, 0.00000000e+00,        4.45145846e-03]])
    
    matrix[0,0] *= 0.53
    
    switching = sc.odict()
    for i,method1 in enumerate(default_methods()):
        switching[method1] = matrix[i,:] / matrix[i,:].sum() # Normalize now
    
    return switching # initial
    

def default_efficacy():
    ''' From Guttmacher, fp/docs/gates_review/contraceptive-failure-rates-in-developing-world_1.pdf '''
    
    # Expressed as failure rates
    method_efficacy = sc.odict({
            'None':100.0, # WARNING, should this be 0.3, the pregnancy rate per act?
            'Lactation':10.0,
            'Implants':0.6,
            'Injectables':1.7,
            'IUDs':1.4,
            'Pill':5.5,
            'Condoms':5.4,
            'Other':5.5,
            'Traditional':13.4,
            })
    method_efficacy[:] = 100
    
    for key,value in method_efficacy.items():
        method_efficacy[key] = method_efficacy[key]/100
    
    assert method_efficacy.keys() == default_methods() # Ensure ordering
    
    return method_efficacy


def default_barriers():
    barriers = sc.odict({
        'No need':45.6,
        'Opposition':30.1,
        'Knowledge':10.0,
        'Access':2.7,
        'Health':11.6,
        })
    barriers[:] /= barriers[:].sum() # Ensure it adds to 1    
    return barriers

def make_pars():
    pars = sc.odict()

    # Simulation parameters
    pars['name'] = 'Default' # Name of the simulation
    pars['n'] = int(1274*0.49) # Number of people in the simulation -- from Impact 2 / 1000
    pars['start_year'] = 1962
    pars['end_year'] = 2015
    pars['timestep'] = 3 # Timestep in months
    pars['verbose'] = True
    pars['seed'] = 1 # Random seed, if None, don't reset
    
    pars['methods'] = default_methods()
    pars['age_pyramid'] = default_age_pyramid()
    pars['age_mortality'] = default_age_mortality()
    pars['age_fertility'] = default_age_fertility()
    pars['method_efficacy'] = default_efficacy()
    pars['barriers'] = default_barriers()
    pars['switching'] = default_switching() #pars['initial'], 
    pars['mortality_factor'] = 1#2.7
    pars['fertility_factor'] = 8#0.1*33 # No idea why this needs to be so high
    pars['fertility_variation'] = [1,1]#[0.3,2.0] # Multiplicative range of fertility factors
    pars['method_age'] = 15 # When people start choosing a method (sexual debut)
    pars['max_age'] = 99
    
    return pars