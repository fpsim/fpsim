"""
This is a script that runs a comparison between a country's newest data and the calibrated model's predicted values
for that respective year. It can be used as a means of assessing a model's accuracy in its predictions.
"""
import os
import sciris as sc
import fpsim as fp
import numpy as np
import pandas as pd
import pylab as pl


# Global Variables
country = 'kenya'
latest_year = 2022
do_plot = True
do_save = True
min_age = 15
max_age = 50

# TODO: Import new data
new_mcpr = 45.9
new_asfr = [2.9, 62.6, 167.8, 171.6, 132.2, 77.0, 32.8, 11.7]
new_method_mix = {
    'Withdrawal': 1.04,
    'Other traditional': 4.46,
    'Condoms': 8.42,
    'Pill': 7.95,
    'Injectables': 34.07,
    'Implants': 33.96,
    'IUDs': 3.09,
    'BTL': 3.88,
    'Other modern': 3.13
}
# new_birth_spacing =

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


# Analyze difference between predicted and actual mcpr values. From UN Data Portal
def compare_mcpr():
    res_index = np.where(res['t'] == pars['end_year'])[0][0]
    predicted_mcpr = res['mcpr'][res_index] * 100
    print('#################### MCPR COMPARISON ####################')
    print(f'Predicted mcpr value: {predicted_mcpr}')
    print(f'Actual mcpr value: {new_mcpr}')

    # Set up plotting
    df_mcpr = pd.DataFrame({'UN Data': new_mcpr, 'FPsim': predicted_mcpr}, index=['MCPR'])

    # Plot use
    ax = df_mcpr.plot.barh(color={'UN Data': 'black', 'FPsim': 'cornflowerblue'})
    ax.set_xlabel('Percent')
    ax.set_title(f'{country.capitalize()}: {latest_year} MCPR - Model vs Data')
    if do_save:
        pl.savefig(f"../{country}/prediction_figs/mcpr.png", bbox_inches='tight', dpi=100)

    pl.show()

    return


# Analyze difference between predicted and actual asfr values. From UN Data Portal
def compare_asfr():
    x = [1, 2, 3, 4, 5, 6, 7, 8]

    x_labels = []
    asfr_model = []

    # Extract from model
    for key in age_bin_map.keys():
        x_labels.append(key)
        asfr_model.append(res['asfr'][key][-1])

    # Plot
    fig, ax = pl.subplots()
    kw = dict(lw=3, alpha=0.7, markersize=10)
    ax.plot(x, new_asfr, marker='^', color='black', label="UN Data", **kw)
    ax.plot(x, asfr_model, marker='*', color='cornflowerblue', label="FPsim", **kw)
    pl.xticks(x, x_labels)
    pl.ylim(bottom=-10)
    ax.set_title(f'{country.capitalize()}: {latest_year} Age-specific fertility rate per 1000 woman years')
    ax.set_xlabel('Age')
    ax.set_ylabel(f'ASFR')
    ax.legend(frameon=False)
    sc.boxoff()

    if do_save:
        pl.savefig(f'../{country}/prediction_figs/asfr.png')

    pl.show()

    print('#################### ASFR COMPARISON ####################')
    print(f'Predicted asfr value: {asfr_model}')
    print(f'Actual asfr value: {new_asfr}')

    return


# Analyze difference between predicted and actual mcpr values. Generated from method_mix_PMA.R
def compare_method_mix():
    # Pull method definitions from parameters file
    # Method map; this remains constant across locations. True indicates modern method,
    # and False indicates traditional method
    methods_map_model = {  # Index, modern, efficacy
        'None': [0, False],
        'Withdrawal': [1, False],
        'Other traditional': [2, False],
        # 1/2 periodic abstinence, 1/2 other traditional approx.  Using rate from periodic abstinence
        'Condoms': [3, True],
        'Pill': [4, True],
        'Injectables': [5, True],
        'Implants': [6, True],
        'IUDs': [7, True],
        'BTL': [8, True],
        'Other modern': [9, True],
    }

    # Setup
    model_labels_all = list(methods_map_model.keys())
    model_labels_methods = sc.dcp(model_labels_all)
    model_labels_methods = model_labels_methods[1:]

    model_method_counts = sc.odict().make(keys=model_labels_all, vals=0.0)

    # Extract from model
    for i in range(len(ppl)):
        if ppl.alive[i] and not ppl.sex[i] and min_age <= ppl.age[i] < max_age:
            model_method_counts[ppl.method[i]] += 1

    model_method_counts[:] /= model_method_counts[:].sum()
    model_methods_mix = sc.dcp(model_method_counts)

    # Calculate mix within users in model
    model_methods_mix['None'] = 0.0
    model_users_sum = model_methods_mix[:].sum()
    model_methods_mix[:] /= model_users_sum
    mix_model = model_methods_mix.values()[1:]
    mix_percent_model = [i * 100 for i in mix_model]

    # Set up plotting
    mix_percent_data = list(new_method_mix.values())
    df_mix = pd.DataFrame({'PMA': mix_percent_data, 'FPsim': mix_percent_model}, index=model_labels_methods)

    # Plot mix
    ax = df_mix.plot.barh(color={'PMA': 'black', 'FPsim': 'cornflowerblue'})
    ax.set_xlabel('Percent users')
    ax.set_title(f'{country.capitalize()}: {latest_year} Contraceptive Method Mix - Model vs Data')
    if do_save:
        pl.savefig(f'../{country}/prediction_figs/method_mix.png', bbox_inches='tight', dpi=100)

    pl.show()

    return


"""
# Analyze difference between predicted and actual birth spacing values. Generated from birth_spacing.R script
def compare_birth_spacing():
    return
"""


if __name__ == "__main__":
    # Set location and end_year sim parameters
    pars = fp.pars(location=country)
    pars['end_year'] = latest_year

    # Run simulation
    sc.tic()
    sim = fp.Sim(pars=pars)
    sim.run()

    # Save results and people to analyze
    res = sim.results
    ppl = sim.people

    # Generate and save plots, if configured
    if do_plot:
        sim.plot()
        if do_save == 1 and os.path.exists(f'../{country}/prediction_figs') is False:
            os.mkdir(f'../{country}/prediction_figs')

    # Run comparison between model predicted values and new actual data
    compare_mcpr()
    compare_asfr()
    compare_method_mix()
    #compare_birth_spacing()

    sc.toc()
    print('Done.')
