"""
This is a script that runs a comparison between a country's newest data and the calibrated model's predicted values
for that respective year. It can be used as a means of assessing a model's accuracy in its predictions.
"""

import sciris as sc
import fpsim as fp
import numpy as np

# Global Variables
country = 'kenya'
do_plot = True

# TODO: Import new data
new_mcpr = 45.9
# new_method_mix =
new_asfr = [2.9, 62.6, 167.8, 171.6, 132.2, 77.0, 32.8, 11.7]
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
    index = np.where(res['t'] == pars['end_year'])[0][0]
    predicted_mcpr = res['mcpr'][index] * 100
    print(f'Predicted mcpr value: {predicted_mcpr}')
    print(f'Actual mcpr value: {new_mcpr}')
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
    for i in range(len(sim.ppl)):
        if ppl.alive[i] and not ppl.sex[i] and ppl.age[i] >= sim.pars['min_age'] and ppl.age[i] < sim.pars['min_age']:
            model_method_counts[ppl.method[i]] += 1

    model_method_counts[:] /= model_method_counts[:].sum()
    model_methods_mix = sc.dcp(model_method_counts)

    # Calculate mix within users in model
    model_methods_mix['None'] = 0.0
    model_users_sum = model_methods_mix[:].sum()
    model_methods_mix[:] /= model_users_sum
    mix_model = model_methods_mix.values()[1:]
    mix_percent_model = [i * 100 for i in mix_model]

    return

# Analyze difference between predicted and actual asfr values. From UN Data Portal
def compare_asfr():
    # Print ASFR form model in output
    for key in age_bin_map.keys():
        print(f'ASFR (annual) for age bin {key} in the last year of the sim: {res["asfr"][key][-1]}')

    x = [1, 2, 3, 4, 5, 6, 7, 8]

    # Load data
    year = data_asfr[data_asfr['year'] == pars['end_year']]
    asfr_data = year.drop(['year'], axis=1).values.tolist()[0]

    x_labels = []
    asfr_model = []

    # Extract from model
    for key in age_bin_map.keys():
        x_labels.append(key)
        asfr_model.append(res['asfr'][key][-1])
    return

"""
# Analyze difference between predicted and actual birth spacing values. Generated from birth_spacing.R script
def compare_birth_spacing():
    return
"""

if __name__ == "__main__":

    pars = fp.pars(location=country)
    pars['end_year'] = 2022

    sc.tic()
    sim = fp.Sim(pars=pars)
    sim.run()

    ppl = sim.people

    if do_plot:
        sim.plot()

    # Save results
    res = sim.results

    compare_mcpr()
    compare_asfr()

    sc.toc()
    print('Done.')
