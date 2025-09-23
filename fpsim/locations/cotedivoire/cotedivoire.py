"""
This configuration file is for an FPsim model specific to Cote d'Ivore.

Users may update values marked as USER-EDITABLE to match the context
they are modeling for a specific version of FPsim.
"""
import numpy as np
from pathlib import Path
from fpsim import defaults as fpd
import fpsim.locations.data_utils as fpld
import fpsim as fp
import pandas as pd
import sciris as sc
import starsim as ss
import matplotlib.pyplot as plt
from fpsim import plotting as fpplt
import os

# %% Housekeeping

def scalar_pars():
    scalar_pars = {
        'postpartum_dur':       23,     # <<< USER-EDITABLE: Adjust/override any parameters that are defined in fpsim/defaults.py
    }
    return scalar_pars


def filenames():
    """ Data files for use with calibration, etc -- not needed for running a sim """
    base_dir = Path(__file__).resolve().parent / 'data'
    files = {}                              # <<< USER-EDITABLE: If setting a regional location and want to default
    # to using country data here where regional data may be unavailable, import the country module at the top of this
    # file and change this line to `files={country}.filenames()` to call the country filenames function before overwriting
    # with any regional files below. **If regional files for data below does not exist, remove that respective line (will then default to country data)
    files['base'] = base_dir
    files['basic_wb'] = base_dir / 'basic_wb.yaml' # From World Bank https://data.worldbank.org/indicator/SH.STA.MMRT
    files['popsize'] = base_dir / 'popsize.csv' # Downloaded from World Bank: https://data.worldbank.org/indicator/SP.POP.TOTL
    files['mcpr'] = base_dir / 'cpr.csv'  # From UN Population Division Data Portal, married women 1970-1986, all women 1990-2030
    files['tfr'] = base_dir / 'tfr.csv'   # From World Bank https://data.worldbank.org/indicator/SP.DYN.TFRT.IN
    files['asfr'] = base_dir / 'asfr.csv' # From UN World Population Prospects 2022: https://population.un.org/wpp/Download/Standard/Fertility/
    files['ageparity'] = base_dir / 'ageparity.csv' # Choose from either DHS 2014 or PMA 2022
    files['spacing'] = base_dir / 'birth_spacing_dhs.csv' # From DHS
    files['methods'] = base_dir / 'mix.csv' # From PMA
    files['afb'] = base_dir / 'afb.table.csv' # From DHS
    files['use'] = base_dir / 'use.csv' # From PMA
    files['education'] = base_dir / 'edu_initialization.csv' # From DHS
    return files


# %% Pregnancy exposure

def exposure_age():
    """
    Returns an array of experimental factors to be applied to account for
    residual exposure to either pregnancy or live birth by age.  Exposure to pregnancy will
    increase factor number and residual likelihood of avoiding live birth (mostly abortion,
    also miscarriage), will decrease factor number
    """
    exposure_correction_age = np.array([[0,     5,  10, 12.5, 15, 18, 20, 25, 30, 35,   40, 45, 50],
                                        [0.1, 0.1, 0.5,  0.1,0.1, 0.6,1.5,0.6,0.6,0.2,  0.3,0.5,0.2]])  # <<< USER-EDITABLE: Can be modified for calibration


    exposure_age_interp = fpld.data2interp(exposure_correction_age, fpd.spline_preg_ages)
    return exposure_age_interp


def exposure_parity():
    """
    Returns an array of experimental factors to be applied to account for residual exposure to either pregnancy
    or live birth by parity.
    """
    exposure_correction_parity = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                                           [1, 1, 1, 1, 1, 1, 1, 0.8, 0.5, 0.3, 0.15, 0.10, 0.05, 0.01]])  # <<< USER-EDITABLE: Can be modified for calibration
    exposure_parity_interp = fpld.data2interp(exposure_correction_parity, fpd.spline_parities)

    return exposure_parity_interp


# %% Make and validate parameters

def make_pars(location='cotedivoire', seed=None):  # <<< USER-EDITABLE: Change name of location; country name if country, region name if region
    """
    Take all parameters and construct into a dictionary
    """

    # Scalar parameters and filenames
    pars = scalar_pars()
    pars['abortion_prob'], pars['twins_prob'] = fpld.scalar_probs(location)     # <<< USER-EDITABLE: **If setting up regional location and want to use params from country data rather than regional data,
    pars.update(fpld.bf_stats(location))                                                # change 'location' argument being passed in any of these function calls to the country name (e.g. `fpld.scalar_probs('ethiopia')` )
    pars['filenames'] = filenames()

    # Demographics and pregnancy outcome
    pars['age_pyramid'] = fpld.age_pyramid(location)
    pars['age_mortality'] = fpld.age_mortality(location, data_year=2010)
    pars['urban_prop'] = fpld.urban_proportion(location)
    pars['maternal_mortality'] = fpld.maternal_mortality(location)
    pars['infant_mortality'] = fpld.infant_mortality(location)
    pars['miscarriage_rates'] = fpld.miscarriage()
    pars['stillbirth_rate'] = fpld.stillbirth(location)

    # Fecundity
    pars['age_fecundity'] = fpld.female_age_fecundity()
    pars['fecundity_ratio_nullip'] = fpld.fecundity_ratio_nullip()
    pars['lactational_amenorrhea'] = fpld.lactational_amenorrhea(location)

    # Pregnancy exposure
    pars['sexual_activity'] = fpld.sexual_activity(location)
    pars['sexual_activity_pp'] = fpld.sexual_activity_pp(location)
    pars['debut_age'] = fpld.debut_age(location)
    pars['exposure_age'] = exposure_age()
    pars['exposure_parity'] = exposure_parity()
    pars['spacing_pref'] = fpld.birth_spacing_pref(location)

    # Contraceptive methods
    pars['mcpr'] = fpld.mcpr(location)

    # Demographics: partnership and wealth status
    pars['age_partnership'] = fpld.age_partnership(location)
    pars['wealth_quintile'] = fpld.wealth(location)

    return pars

def make_pars_calib():
    pars = fp.make_fp_pars()  # For default pars
    pars.update_location(country)

    # Modify individual fecundity and exposure parameters
    # These adjust each woman's probability of conception and exposure to pregnancy.
    pars['fecundity_var_low'] = 0.5
    pars['fecundity_var_high'] = 1.5
    pars['exposure_factor'] = 0.5
    
    # Adjust contraceptive choice parameters
    cm_pars = dict(prob_use_year = 2020,  # Base year
                   prob_use_trend_par = 0.1,  # Time trend in contraceptive use - adjust this to get steeper/slower trend
                   prob_use_intercept = 0.6,  # Intercept for the probability of using contraception - shifts the mCPR level
                   method_weights = np.array([5, 10, 20, 1, 1, 1,0.01, 0.01, 0.1]))

    # Postpartum sexual activity correction or 'birth spacing preference'. Pulls values from {location}/data/birth_spacing_pref.csv by default
    # Set all to 1 to reset. Option to use 'optimize-space-prefs.py' script in this directory to determine values
    # 'months': array([ 0.,  3.,  6.,  9., 12., 15., 18., 21., 24., 27., 30., 33., 36., 39., 42., 45., 48., 51., 54.]),
    # The probability of sex --> very indirect, so need a larger term, 
    # when you are 2 years postpartum, dhs data sexual activity, probability of sex
    pars['spacing_pref']['preference'][:3] =  1  # Spacing of 0-6 months
    pars['spacing_pref']['preference'][3:6] = 0.5  # Spacing of 9-15 months
    pars['spacing_pref']['preference'][6:9] = 0.8  # Spacing of 18-24 months
    pars['spacing_pref']['preference'][9:] =  2  # Spacing of 27-36 months
 
    return pars, cm_pars


def make_sim(pars=None, stop=2021):
    if pars is None:
        pars, cm_pars = make_pars_calib()

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

    import sys
    sys.path.insert(0, '/Users/')
    sys.path.insert(0, '/Users/laurynbruce/')
    sys.path.insert(0, '/Users/laurynbruce/')
    sys.path.insert(0, '/Users/laurynbruce/Documents/')
    sys.path.insert(0, '/Users/laurynbruce/Documents/Lauryn_Bruce/UCSD/')
    sys.path.insert(0, '/Users/laurynbruce/Documents/Lauryn_Bruce/UCSD/research/')
    sys.path.insert(0, '/Users/laurynbruce/Documents/Lauryn_Bruce/UCSD/research/Gates/')
    sys.path.insert(0, '/Users/laurynbruce/Documents/Lauryn_Bruce/UCSD/research/Gates/project/')
    sys.path.insert(0, '/Users/laurynbruce/Documents/Lauryn_Bruce/UCSD/research/Gates/project/fpsim_3.3/')
    sys.path.insert(0, '/Users/laurynbruce/Documents/Lauryn_Bruce/UCSD/research/Gates/project/fpsim_3.3/fpsim/')
    sys.path.insert(0, '/Users/laurynbruce/Documents/Lauryn_Bruce/UCSD/research/Gates/project/fpsim_3.3/fpsim/fpsim/')

    if do_run:

        # Settings
        country = 'cotedivoire'
        fpplt.Config.set_figs_directory('calib_results/figures/')
        fpplt.Config.do_save = True
        fpplt.Config.do_show = False
        fpplt.Config.show_rmse = True

        # Create simulation with parameters
        sim = make_sim()
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

