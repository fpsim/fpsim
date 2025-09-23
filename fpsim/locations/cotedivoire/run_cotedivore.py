"""
Script to create a calibrated model of Cote d'Ivore.
Run the model and generate plots showing the
discrepancies between the model and data.

Users may update values marked as USER-EDITABLE to match the context
they are modeling for a specific version of FPsim.
"""
# If running a local instance of fpsim, set path
# import sys
# sys.path.insert(0, '/Users/') # <<< USER-EDITABLE 
# sys.path.insert(0, '/Users/PATHTOFPSIM/') # <<< USER-EDITABLE 

import numpy as np
import fpsim as fp
import pandas as pd
import sciris as sc
import starsim as ss
import matplotlib.pyplot as plt
from fpsim import plotting as fpplt
import os

# Settings
country = 'cotedivoire'    # <<< USER-EDITABLE   
fpplt.Config.set_figs_directory('calib_results/figures/')
fpplt.Config.do_save = True
fpplt.Config.do_show = False
fpplt.Config.show_rmse = True

def make_pars(country):
    print(country)
    pars = fp.make_fp_pars()  # For default pars
    pars.update_location(country)

    # Modify individual fecundity and exposure parameters
    # These adjust each woman's probability of conception and exposure to pregnancy.
    pars['fecundity_var_low'] = 1.1 # <<< USER-EDITABLE: Adjust/override any parameters that are defined in fpsim/defaults.py 
    pars['fecundity_var_high'] = .7 # <<< USER-EDITABLE: Adjust/override any parameters that are defined in fpsim/defaults.py  
    pars['exposure_factor'] = 2 # <<< USER-EDITABLE: Adjust/override any parameters that are defined in fpsim/defaults.py  
    
    # Adjust contraceptive choice parameters
    # <<< USER-EDITABLE: Adjust/override any parameters that are defined in fpsim/defaults.py  
    cm_pars = dict(prob_use_year = 2020,  # Base year 
                   prob_use_trend_par = 0.01,  # Time trend in contraceptive use - adjust this to get steeper/slower trend
                   prob_use_intercept = 0.5,  # Intercept for the probability of using contraception - shifts the mCPR level
                   method_weights = np.array([8, 5, 5, 20, 2, 2, 0.015, 0.015, 1]))

    # Postpartum sexual activity correction or 'birth spacing preference'. Pulls values from {location}/data/birth_spacing_pref.csv by default
    # Set all to 1 to reset. Option to use 'optimize-space-prefs.py' script in this directory to determine values
    # 'months': array([ 0.,  3.,  6.,  9., 12., 15., 18., 21., 24., 27., 30., 33., 36., 39., 42., 45., 48., 51., 54.]),
    # The probability of sex --> very indirect, so need a larger term, 
    # when you are 2 years postpartum, dhs data sexual activity, probability of sex
    # <<< USER-EDITABLE: Adjust/override any parameters that are defined in fpsim/defaults.py  
    pars['spacing_pref']['preference'][:3] =  1  # Spacing of 0-6 months
    pars['spacing_pref']['preference'][3:6] = 0.5  # Spacing of 9-15 months
    pars['spacing_pref']['preference'][6:9] = 0.8  # Spacing of 18-24 months
    pars['spacing_pref']['preference'][9:] =  2  # Spacing of 27-36 months
 
    return pars, cm_pars


def make_sim(country, pars=None, stop=2021):
    if pars is None:
        pars, cm_pars = make_pars(country)
    pars.location = country

    method_choice = fp.SimpleChoice(pars=cm_pars, location=country)  

    sim = fp.Sim(pars=pars, contraception_module=method_choice)

    return sim



def plot_calib(sim, single_fig=False, fig_kwargs=None, legend_kwargs=None):
    """ Plots the commonly used plots for calibration
    Plotting class function which plots the primary calibration targets:
    method mix, method use, cpr, total fertility rate, birth spacing, 
    age at first birth, and age-specific fertility rate.
    
    """

    fpplt.plot_calib(sim)

    return


if __name__ == '__main__':
    do_run = True  # Whether to run the sim or load from file
    country = 'cotedivoire'

    if do_run:
        # Create simulation with parameters
        sim = make_sim(country)
        sim.run()

        # Create directory in current directory if it doesn't exist
        os.makedirs('calib_results', exist_ok=True)
        sc.saveobj(f'calib_results/{country}_calib.sim', sim)
    else:
        os.makedirs('calib_results', exist_ok=True)
        sim = sc.loadobj(f'calib_results/{country}_calib.sim')

    # Set options for plotting
    sc.options(fontsize=20)  # Set fontsize
    plot_calib(sim, single_fig=True)

