"""
A script for running plotting to compare the model to data.

PRIOR TO RUNNING:
1. Be sure to set the country and parameter values correctly in run_sim below

2. Ensure that fpsim/locations contains both a directory for the country
being calibrated and a corresponding location file (i.e. 'ethiopia.py')

3. In order to run this script, the country data must be stored in the country directory mentioned above and with the
following naming conventions:

ageparity.csv # Age-parity distribution file
use.csv # Dichotomous contraceptive method use
birth_spacing_dhs.csv  # Birth-to-birth interval data
afb.table.csv  # Ages at first birth in DHS for women age 25-50
cpr.csv  # Contraceptive prevalence rate data; from UN Data Portal
asfr.csv  # Age-specific data fertility rate data
mix.csv  # Contraceptive method mix
tfr.csv  # Total fertility rate data
popsize.csv  # Population by year

4. Ensure that the data in the aforementioned files is formatted in the same manner as the kenya data files,
which were used as a standard in writing this script.
"""

import os
import numpy as np
import pandas as pd
import sciris as sc
import fpsim as fp
from fpsim import plotting as plt

country = 'kenya'

def run_sim():
        # Set up sim for country
        pars = fp.pars(location=country)
        pars['n_agents'] = 10_000  # Small population size
        pars['start_year'] = 2000
        pars['end_year'] = 2020  # 1961 - 2020 is the normal date range

        # Free parameters for calibration
        pars['fecundity_var_low'] = .8
        pars['fecundity_var_high'] = 3.25
        pars['exposure_factor'] = 3.5

        # Set option to save figures
        do_save = True

        # Dataset contained in the ageparity csv file to which the model data will be compared (i.e. 'PMA 2022',
        # 'DHS 2014', etc). If this is set to a dataset not included in the {country}_ageparity.csv file, you will receive
        # an error when running the script.
        ageparity_dataset = 'PMA 2022'

        sc.tic()

        cwd = os.path.dirname(os.path.abspath(__file__))
        country_dir = os.path.abspath(os.path.join(cwd, '../../fpsim/locations/', country))
        figs_dir    = os.path.join(country_dir, 'figs')
        if do_save and not os.path.exists(figs_dir):
            os.mkdir(figs_dir)


        # # Last free parameter, postpartum sexual activity correction or 'birth spacing preference'
        # # Set all to 1 to reset
        # # spacing_pars = {'space0_6': 1, 'space18_24': 1, 'space27_36': 1, 'space9_15': 1}  # output from 'optimize-space-prefs-{country}.py'
        # # pars['spacing_pref']['preference'][:3] = spacing_pars['space0_6']
        # # pars['spacing_pref']['preference'][3:6] = spacing_pars['space9_15']
        # # pars['spacing_pref']['preference'][6:9] = spacing_pars['space18_24']
        # #pars['spacing_pref']['preference'][9:] = spacing_pars['space27_36'] # Removing this bin for Kenya as it doesn't extend out
        #
        # # Only other free parameters are age-based exposure and parity-based exposure, can adjust manually in {country}.py

        # Adjust contraceptive choice parameters
        cm_pars = dict(
            prob_use_year=2020,
            prob_use_trend_par=0.03,
            force_choose=False,
            method_weights=np.array([0.34, .7, 0.6, 0.74, 0.76, 1, 1.63, 0.65, 9.5])
        )
        method_choice = fp.SimpleChoice(pars=cm_pars, location='kenya')
        sim = fp.Sim(pars=pars, contraception_module=method_choice, analyzers=[fp.cpr_by_age(), fp.method_mix_by_age()])
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

