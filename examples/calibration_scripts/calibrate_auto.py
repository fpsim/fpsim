'''
A sample script for using the Calibration class to generate the optimal set of free parameters, run a sim with that set
of free params, and then generate plots showing the discrepancies between the model vs data.

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
country = 'kenya'

# Modify the figs directory and ensure it exists
plt.Config.figs_directory = './figs_auto_calib'


def run_sim():
        # Set up sim for country
        pars = fp.pars(location=country)
        pars['n_agents'] = 100 # Small population size
        pars['end_year'] = 2020 # 1961 - 2020 is the normal date range

        # Free parameters for calibration
        freepars = dict(
                fecundity_var_low = [0.95, 0.925, 0.975],
                fecundity_var_high = [1.05, 1.025, 1.075],
                exposure_factor = [1.0, 0.95, 1.0],
        )
        # Only other free parameters are age-based exposure and parity-based exposure, can adjust manually in {country}.py

        # Last free parameter, postpartum sexual activity correction or 'birth spacing preferece'
        # Set all to 1 to reset
        spacing_pars = {'space0_6': 1, 'space18_24': 1, 'space27_36': 1, 'space9_15': 1}  # output from 'optimize-space-prefs-{country}.py'
        pars['spacing_pref']['preference'][:3] = spacing_pars['space0_6']
        pars['spacing_pref']['preference'][3:6] = spacing_pars['space9_15']
        pars['spacing_pref']['preference'][6:9] = spacing_pars['space18_24']
        #pars['spacing_pref']['preference'][9:] = spacing_pars['space27_36'] # Removing this bin for Kenya as it doesn't extend out

        calibration = fp.Calibration(pars, calib_pars=freepars)
        calibration.calibrate()
        pars.update(calibration.best_pars)
        sim = fp.Sim(pars=pars)
        sim.run()

        return sim


if __name__ == '__main__':

        sim = run_sim()

        # Load in validation data
        val_data_list = ['ageparity', 'use', 'spacing', 'afb', 'mcpr', 'asfr', 'methods', 'tfr', 'popsize']
        val_data_sources = getattr(fp.locations, country).filenames()
        val_data = sc.objdict()
        for vd in val_data_list:
                val_data[vd] = pd.read_csv(f"{val_data_sources['base']}/{val_data_sources[vd]}")

        # Set options for plotting
        plt.plot_calib(sim, val_data)