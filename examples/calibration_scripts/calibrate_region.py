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
# Name of the region being calibrated
region = 'somali'
# Set option to save figures
do_save = 1

cwd = os.path.abspath(os.getcwd())
country_dir = f'../../fpsim/locations/{country}'
figs_dir = os.path.join(cwd, country_dir, 'regions/figs', region)

if do_save == 1 and os.path.exists(figs_dir) is False:
        os.makedirs(figs_dir, exist_ok=True)

def run_sim():
        # Set up sim for country
        pars = fp.pars(location=region)
        pars['n_agents'] = 10_000 # Small population size
        pars['end_year'] = 2016 # 1961 - 2020 is the normal date range

        # Free parameters for calibration
        #freepars = dict(
        #        fecundity_var_low = [0.95, 0.925, 0.975],
        #        fecundity_var_high = [1.05, 1.025, 1.075],
        #        exposure_factor = [1.0, 0.95, 1.0],
        #)
        # Other free parameters are age-based exposure and parity-based exposure, can adjust manually in {country}.py
        pars['fecundity_var_low'] = 0.95
        pars['fecundity_var_high'] = 1.0
        pars['exposure_factor'] = 1

        # Last free parameter, postpartum sexual activity correction or 'birth spacing preferece'
        # Set all to 1 to reset
        spacing_pars = {'space0_6': 1, 'space18_24': 1, 'space27_36': 1, 'space9_15': 1}  # output from 'optimize-space-prefs-{country}.py'
        pars['spacing_pref']['preference'][:3] = spacing_pars['space0_6']
        pars['spacing_pref']['preference'][3:6] = spacing_pars['space9_15']
        pars['spacing_pref']['preference'][6:9] = spacing_pars['space18_24']
        #pars['spacing_pref']['preference'][9:] = spacing_pars['space27_36'] # Removing this bin for Kenya as it doesn't extend out

        # Convert region name to the format used in the data
        if region == 'benishangul_gumuz':
                formatted_region = region.replace('_', '-').title()  # Replace underscore with dash and capitalize each word
        elif region == 'snnpr':
                formatted_region = 'SNNPR'
        else:
                formatted_region = region.replace('_', ' ').title()  # Replace underscore with space and capitalize each word

        # Import country data files to compare
        #data_asfr = pd.read_csv(f'{country_dir}/regions/data/asfr_region.csv').loc[lambda df: df['region'] == formatted_region]
        #data_methods = pd.read_csv(f'{country_dir}/regions/data/mix_region.csv').loc[lambda df: (df['region'] == formatted_region) & (df['year'] == pars['end_year'])]
        #data_tfr = pd.read_csv(f'{country_dir}/regions/data/tfr_region.csv').loc[lambda df: df['region'] == formatted_region]
        #data_use = pd.read_csv(f'{country_dir}/regions/data/use_region.csv').loc[lambda df: (df['region'] == formatted_region) & (df['year'] == pars['end_year'])]
        
        #calibration = fp.Calibration(pars, calib_pars=freepars)
        #calibration.calibrate()
        #pars.update(calibration.best_pars)
        sim = fp.Sim(pars=pars)
        sim.run()

        return sim, formatted_region

if __name__ == '__main__':

        sim, region = run_sim()
        sim.plot()

        # Load in validation data
        val_data_list = ['asfr_region', 'methods_region', 'tfr_region', 'use_region']       # Can reference regional files in {country}/regions/data
        val_data_sources = getattr(fp.locations, country).filenames()
        val_data = sc.objdict()
        for vd in val_data_list:
                val_data[vd] = pd.read_csv(f"{val_data_sources['base']}/{val_data_sources[vd]}").loc[lambda df: (df['region'] == region) & (df['year'] == sim.pars['end_year'])]

        # Set options for plotting
        plt.plot_asfr(val_data['asfr_region'], sim.results, sim.pars)
        plt.plot_methods(val_data['methods_region'], val_data['use_region'], sim)
        plt.plot_tfr(val_data['tfr_region'], sim.results)
