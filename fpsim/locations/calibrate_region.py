'''
A script for running plotting to compare the model to data.

PRIOR TO RUNNING:
1. Be sure to set the user global variables in the first section below (country, plotting options,
save option, and skyscrapers dataset name)

2. Ensure that fpsim/locations contains both a directory for the country
being calibrated as well as a corresponding location file (i.e. 'ethiopia.py')

3. In order to run this script, the country data must be stored in the country directory mentioned above and with the
following naming conventions:

skyscrapers.csv' # Age-parity distribution file
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
import numpy as np
import pandas as pd
import sciris as sc
import fpsim as fp
import pylab as pl
import seaborn as sns

####################################################
# GLOBAL VARIABLES: USER MUST SET

# Name of the country being calibrated. To note that this should match the name of the country data folder
country = 'ethiopia'
region = 'Amhara'

# Set options for plotting
do_plot_sim = True
do_plot_asfr = True
do_plot_methods = False
do_plot_skyscrapers = False
do_plot_cpr = False
do_plot_tfr = False
do_plot_pop_growth = False
do_plot_birth_space_afb = False

# Set option to save figures
do_save = 0

# Dataset contained in the skyscrapers csv file to which the model data will be compared (i.e. 'PMA 2022',
# 'DHS 2014', etc). If this is set to a dataset not included in the {country}_skyscrapers.csv file, you will receive
# an error when running the script.
#skyscrapers_dataset = 'PMA 2019'

####################################################

if do_save == 1 and os.path.exists(f'./{country}/subnational/figs') == False:
    os.mkdir(f'./{country}/subnational/figs')

# Import country data files to compare
data_asfr = pd.read_csv(f'./{country}/subnational/asfr_region.csv')
data_methods = pd.read_csv(f'./{country}/subnational/mix_region.csv')
data_tfr = pd.read_csv(f'./{country}/subnational/tfr_region.csv')
use = pd.read_csv(f'./{country}/subnational/use_region.csv') #Dichotomous contraceptive method use


# Set up global variables
age_bin_map = {
        '10-14': [10, 15],
        '15-19': [15, 20],
        '20-24': [20, 25],
        '25-29': [25, 30],
        '30-34': [30, 35],
        '35-39': [35, 40],
        '40-44': [40, 45],
        '45-49': [45, 50]
}

min_age = 15
max_age = 50
first_birth_age = 25  # age to start assessing first birth age in model
bin_size = 5
mpy = 12  # months per year

sc.tic()

# Set up sim for country
pars = fp.pars(location=country)
pars['n_agents'] = 1_000 # Small population size
pars['end_year'] = 2019 # 1961 - 2020 is the normal date range

# Free parameters for calibration
pars['fecundity_var_low'] = 0.7
pars['fecundity_var_high'] = 1.1
pars['exposure_factor'] = 1
pars['high_parity'] = 4
pars['high_parity_nonuse'] = 0.6

# Last free parameter, postpartum sexual activity correction or 'birth spacing preference'
# Set all to 1 to reset
spacing_pars = {'space0_6': 1, 'space18_24': 1, 'space27_36': 1, 'space9_15': 1}  # output from 'optimize-space-prefs-{country}.py'
pars['spacing_pref']['preference'][:3] = spacing_pars['space0_6']
pars['spacing_pref']['preference'][3:6] = spacing_pars['space9_15']
pars['spacing_pref']['preference'][6:9] = spacing_pars['space18_24']
#pars['spacing_pref']['preference'][9:] = spacing_pars['space27_36'] # Removing this bin for Kenya as it doesn't extend out

# Only other free parameters are age-based exposure and parity-based exposure, can adjust manually in {country}.py

# Print out free params being used
print("FREE PARAMETERS BEING USED:")
print(f"Fecundity range: {pars['fecundity_var_low']}-{pars['fecundity_var_high']}")
print(f"Exposure factor: {pars['exposure_factor']}")
print(f"High parity: {pars['high_parity']}")
print(f"High parity, nonuse: {pars['high_parity_nonuse']}")
print(f"Birth spacing preference: {spacing_pars}")
print(f"Age-based exposure and parity-based exposure can be adjusted manually in {country}.py")

# Run the sim
sim = fp.Sim(pars=pars, regional=True)
sim.run()

# Plot results from sim run
if do_plot_sim:
    sim.plot(do_save=True, filename=f'{country}/subnational/figs/fpsim.png')

# Save results
res = sim.results
df = sim.to_df()
df.to_csv(f'{country}/subnational/figs/ethiopia_region_results.csv', index=False)

# Save people from sim
ppl = sim.people

# Set up dictionaries to compare from model vs data
data_dict = {}
model_dict = {} # For comparison from model to data


# Start series of options for plotting data to model comaprisons
if do_plot_asfr:
        '''
        Plot age-specific fertility rate between model and data
        '''
        # Print ASFR form model in output
        for key in age_bin_map.keys():
            print(f'ASFR (annual) for age bin {key} in the last year of the sim: {res["asfr"][key][-1]}')

        x = [1, 2, 3, 4, 5, 6, 7, 8]

        # Load data
        year = data_asfr[data_asfr['year'] == pars['end_year']]
        region = year[year['region'] == region]
        asfr_data = region.drop(['year','region'], axis=1).values.tolist()[0]

        x_labels = []
        asfr_model = []

        # Extract from model
        for key in age_bin_map.keys():
                x_labels.append(key)
                asfr_model.append(res['asfr'][key][-1])

        # Plot
        fig, ax = pl.subplots()
        kw = dict(lw=3, alpha=0.7, markersize=10)
        ax.plot(x, asfr_data, marker='^', color='black', label="UN data", **kw)
        ax.plot(x, asfr_model, marker='*', color='cornflowerblue', label="FPsim", **kw)
        pl.xticks(x, x_labels)
        pl.ylim(bottom=-10)
        ax.set_title(f'{country.capitalize()}: Age specific fertility rate per 1000 woman years')
        ax.set_xlabel('Age')
        ax.set_ylabel('ASFR in 2019')
        ax.legend(frameon=False)
        sc.boxoff()

        if do_save:
            pl.savefig(f'{country}/figs/asfr.png')

        pl.show()

if do_plot_methods:
        '''
        Plots both dichotomous method use and non-use and contraceptive mix
        '''

        # Pull method definitions from parameters file
        # Method map; this remains constant across locations. True indicates modern method,
        # and False indicates traditional method
        methods_map_model = {  # Index, modern, efficacy
        'None': [0, False],
        'Withdrawal': [1, False],
        'Other traditional': [2, False],
        # 1/2 periodic abstinence, 1/2 other traditional approx.  Using rate from periodic abstinence
        'Condoms': [3, True],
        'Pills': [4, True],
        'Injectables': [5, True],
        'Implants': [6, True],
        'IUDs': [7, True],
        'Sterilization': [8, True],
        'Other modern': [9, True],
        }

        # Setup
        model_labels_all = list(methods_map_model.keys())
        model_labels_methods = sc.dcp(model_labels_all)
        model_labels_methods = model_labels_methods[1:]

        model_method_counts = sc.odict().make(keys=model_labels_all, vals=0.0)

        # Extract from model
        for i in range(len(ppl)):
                if ppl.alive[i] and not ppl.sex[i] and ppl.age[i] >= min_age and ppl.age[i] < max_age and ppl.region[i]==region:
                        model_method_counts[ppl.method[i]] += 1

        model_method_counts[:] /= model_method_counts[:].sum()

        data_methods = data_methods[data_methods['Year'] == pars['end_year'] and data_methods['Region'] == region]
        # Method mix from data - country PMA data (mix.csv)
        data_methods_mix = {
                'Withdrawal': data_methods.loc[data_methods['method'] == 'Withdrawal', 'Perc'].iloc[0],
                'Other traditional': data_methods.loc[data_methods['method'] == 'Other traditional', 'Perc'].iloc[0],
                'Condoms': data_methods.loc[data_methods['method'] == 'Condoms', 'Perc'].iloc[0],
                'Pills': data_methods.loc[data_methods['method'] == 'Pills', 'Perc'].iloc[0],
                'Injectables': data_methods.loc[data_methods['method'] == 'Injectables', 'Perc'].iloc[0],
                'Implants': data_methods.loc[data_methods['method'] == 'Implants', 'Perc'].iloc[0],
                'IUDs': data_methods.loc[data_methods['method'] == 'IUDs', 'Perc'].iloc[0],
                'Sterilization': data_methods.loc[data_methods['method'] == 'Sterilization', 'Perc'].iloc[0],
                'Other modern': data_methods.loc[data_methods['method'] == 'Other modern', 'Perc'].iloc[0]
        }

        # Method use from data - country PMA data (use.csv)
        no_use = use.loc[0, 'perc']
        any_method = use.loc[1, 'perc']
        data_methods_use = {
                'No use': no_use,
                'Any method': any_method
        }

        # Plot bar charts of method mix and use among users

        # Calculate users vs non-users in model
        model_methods_mix = sc.dcp(model_method_counts)
        model_use = [model_methods_mix['None'], model_methods_mix[1:].sum()]
        model_use_percent = [i * 100 for i in model_use]

        # Calculate mix within users in model
        model_methods_mix['None'] = 0.0
        model_users_sum = model_methods_mix[:].sum()
        model_methods_mix[:] /= model_users_sum
        mix_model = model_methods_mix.values()[1:]
        mix_percent_model = [i * 100 for i in mix_model]

        # Set method use and mix from data
        mix_percent_data = list(data_methods_mix.values())
        data_use_percent = list(data_methods_use.values())

        # Set up plotting
        use_labels = list(data_methods_use.keys())
        df_mix = pd.DataFrame({'PMA': mix_percent_data, 'FPsim': mix_percent_model}, index=model_labels_methods)
        df_use = pd.DataFrame({'PMA': data_use_percent, 'FPsim': model_use_percent}, index=use_labels)

        # Plot mix
        ax = df_mix.plot.barh(color={'PMA':'black', 'FPsim':'cornflowerblue'})
        ax.set_xlabel('Percent users')
        ax.set_title(f'{country.capitalize()}: Contraceptive Method Mix - Model vs Data')
        if do_save:
                pl.savefig(f"{country}/figs/method_mix.png", bbox_inches='tight', dpi=100)

        # Plot use
        ax = df_use.plot.barh(color={'PMA':'black', 'FPsim':'cornflowerblue'})
        ax.set_xlabel('Percent')
        ax.set_title(f'{country.capitalize()}: Contraceptive Method Use - Model vs Data')
        if do_save:
                pl.savefig(f"{country}/figs/method_use.png", bbox_inches='tight', dpi=100)


if do_plot_tfr:
        '''
        Plot total fertility rate for model vs data
        '''

        # Import data
        #data_tfr = pd.read_csv(f'tfr.csv')

        # Plot
        pl.plot(data_tfr['year'], data_tfr['tfr'], label='World Bank', color='black')
        pl.plot(res['tfr_years'], res['tfr_rates'], label='FPsim', color='cornflowerblue')
        pl.xlabel('Year')
        pl.ylabel('Rate')
        pl.title(f'{country.capitalize()}: Total Fertility Rate - Model vs Data')
        pl.legend()

        if do_save:
                pl.savefig(f'{country}/figs/tfr_over_sim.png')

        pl.show()

sc.toc()
print('Done.')
