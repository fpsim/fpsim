'''
A script for running plotting to compare the model to data.

PRIOR TO RUNNING:
1. Be sure to set the user global variables in the first section below (country, plotting options,
save option, and ageparity dataset name)

2. Ensure that fpsim/locations contains both a directory for the country
being calibrated as well as a corresponding location file (i.e. 'ethiopia.py')

3. In order to run this script, the country data must be stored in the country directory mentioned above and with the
following naming conventions:

ageparity.csv' # Age-parity distribution file
use.csv' # Dichotomous contraceptive method use
birth_spacing_dhs.csv'  # Birth-to-birth interval data
afb.table.csv'  # Ages at first birth in DHS for women age 25-50
cpr.csv'  # Contraceptive prevalence rate data; from UN Data Portal
asfr.csv'  # Age-specific data fertility rate data
mix.csv'  # Contraceptive method mix
tfr.csv'  # Total fertility rate data
popsize.csv'  # Population by year

4. Ensure that the data in the aforementioned files is formatted in the same manner as the kenya data files,
which were used as a standard in writing this script.

'''

import os
import pandas as pd
import sciris as sc
import fpsim as fp
from fpsim import plotting as plt

# Name of the country being calibrated. To note that this should match the name of the country data folder
country = 'ethiopia'

# Dataset contained in the ageparity csv file to which the model data will be compared (i.e. 'PMA 2022',
# 'DHS 2014', etc). If this is set to a dataset not included in the {country}_ageparity.csv file, you will receive
# an error when running the script.
ageparity_dataset = 'PMA 2019'

def run_sim():
        # Set up sim for country
        pars = fp.pars(location=country)
        pars['n_agents'] = 1_000  # Small population size
        pars['end_year'] = 2020  # 1961 - 2020 is the normal date range

        # Free parameters for calibration
        pars['fecundity_var_low'] = 0.95
        pars['fecundity_var_high'] = 1.05
        pars['exposure_factor'] = 1

        # Last free parameter, postpartum sexual activity correction or 'birth spacing preference'
        # Set all to 1 to reset
        spacing_pars = {'space0_6': 1, 'space18_24': 1, 'space27_36': 1,
                        'space9_15': 1}  # output from 'optimize-space-prefs-{country}.py'
        pars['spacing_pref']['preference'][:3] = spacing_pars['space0_6']
        pars['spacing_pref']['preference'][3:6] = spacing_pars['space9_15']
        pars['spacing_pref']['preference'][6:9] = spacing_pars['space18_24']
        # pars['spacing_pref']['preference'][9:] = spacing_pars['space27_36'] # Removing this bin for Kenya as it doesn't extend out

        # Only other free parameters are age-based exposure and parity-based exposure, can adjust manually in {country}.py

        # Print out free params being used
        print("FREE PARAMETERS BEING USED:")
        print(f"Fecundity range: {pars['fecundity_var_low']}-{pars['fecundity_var_high']}")
        print(f"Exposure factor: {pars['exposure_factor']}")
        print(f"Birth spacing preference: {spacing_pars}")
        print(f"Age-based exposure and parity-based exposure can be adjusted manually in {country}.py")

        # Run the sim
        sim = fp.Sim(pars=pars)
        sim.run()

        return sim


if __name__ == '__main__':
        # Run the simulation
        sim = run_sim()
        sim.plot()

        # Load validation data using the default mapping in the Config class
        val_data = plt.Config.load_validation_data(country)

        plt.Config.figs_directory = './figs_manual_calib'

        # Set options for plotting
        plt.plot_ageparity(sim, val_data['ageparity'], ageparity_dataset=ageparity_dataset)
        plt.plot_pop_growth(val_data['popsize'], sim.results, sim.pars)
        plt.plot_methods(val_data['methods'], val_data['use'], sim)
        plt.plot_cpr(val_data['mcpr'], sim.results, sim.pars)
        plt.plot_tfr(val_data['tfr'], sim.results)
        plt.plot_birth_space_afb(val_data['spacing'], val_data['afb'], sim.people)
        plt.plot_asfr(val_data['asfr'], sim.results, sim.pars)

