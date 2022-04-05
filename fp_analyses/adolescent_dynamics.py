import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sciris as sc
import seaborn as sns

import fpsim as fp
from fp_analyses import senegal_parameters as sp

# Housekeeping
sc.tic()
#plt.rcParams['font.size'] = 10

do_save = 1
do_print_stats = 1
do_plot_debut = 1
do_plot_age_first = 1
do_plot_nullip_ages = 1

# Set up sim

pars = sp.make_pars()
pars['n'] = 6000 # Adjust running parameters easily here
pars['start_year'] = 1960
sim = fp.Sim(pars)
sim.run()

# Ensure the figures folder exists
if do_save:
    if not os.path.exists(sp.abspath('figs')):
        print('No figures folder exists and do_save = True, creating...')
        os.makedirs(sp.abspath('figs'))

# Extract people and segment indices by live women in specific age strata
ppl = sim.people

inds_live_women_15_49 = sc.findinds(ppl.alive * (ppl.age >= 15) * (ppl.age < 50) * (ppl.sex == 0))
inds_live_women_10_49 = sc.findinds(ppl.alive * (ppl.age >= 10) * (ppl.age < 50) * (ppl.sex == 0))
inds_live_women_25_49 = sc.findinds(ppl.alive * (ppl.age >= 25) * (ppl.age < 50) * (ppl.sex == 0))
inds_live_women_10_15 = sc.findinds(ppl.alive * (ppl.age >= 10) * (ppl.age < 15) * (ppl.sex == 0))
inds_live_women_15_20 = sc.findinds(ppl.alive * (ppl.age >= 15) * (ppl.age < 20) * (ppl.sex == 0))
inds_live_women_20_25 = sc.findinds(ppl.alive * (ppl.age >= 20) * (ppl.age < 25) * (ppl.sex == 0))
inds_live_women_25_30 = sc.findinds(ppl.alive * (ppl.age >= 25) * (ppl.age < 30) * (ppl.sex == 0))
inds_live_women_30_35 = sc.findinds(ppl.alive * (ppl.age >= 30) * (ppl.age < 35) * (ppl.sex == 0))
inds_live_women_35_40 = sc.findinds(ppl.alive * (ppl.age >= 35) * (ppl.age < 40) * (ppl.sex == 0))
inds_live_women_40_50 = sc.findinds(ppl.alive * (ppl.age >= 40) * (ppl.age < 50) * (ppl.sex == 0))

# Calculate total live women in each age group
total_women_10_49 = len(ppl.uid[inds_live_women_10_49])
total_women_15_49 = len(ppl.uid[inds_live_women_15_49])
total_women_25_49 = len(ppl.uid[inds_live_women_25_49])

if do_print_stats:

    inds_live_women_debuted = sc.findinds((ppl.alive) * (ppl.age >= 25) * (ppl.age < 50) * (ppl.sex == 0) * (ppl.sexual_debut == 1))
    mean_debut_age = np.mean(ppl.sexual_debut_age[inds_live_women_debuted])
    error_debut_age = np.std(ppl.sexual_debut_age[inds_live_women_debuted])
    inds_at_least_one_birth = sc.findinds((ppl.alive) * (ppl.age < 50) * (ppl.sex == 0) * (ppl.parity > 0))
    mean_first_birth_age = np.mean(ppl.first_birth_age[inds_at_least_one_birth])
    error_first_birth_age = np.std(ppl.first_birth_age[inds_at_least_one_birth])

    debut_by_age = {}

    debut_age_mapping = {
        '12': [10, 12],
        '15': [12, 15],
        '18': [15, 18],
        '20': [18, 20],
        '22': [20, 22],
        '25': [22, 25],
    }

    for key,(age_low, age_high) in debut_age_mapping.items():
        inds_debuted = sc.findinds(ppl.alive * (ppl.age >= 25) * (ppl.age < 50) * (ppl.sex == 0) * (ppl.sexual_debut_age<= age_high))
        num_debuted = len(ppl.uid[inds_debuted])
        debut_by_age[key] = num_debuted / total_women_25_49

    inds_abstinent = sc.findinds(ppl.alive * (ppl.age >= 25) * (ppl.age < 50) * (ppl.sex == 0) * (ppl.sexual_debut == 0))
    abstinent = len(ppl.uid[inds_abstinent])

    print(f'Mean age of sexual debut for live women age 25-49 at end of sim: {mean_debut_age} +/- {error_debut_age}')
    print(f'Percent of live women age 25-49 who debuted by exact age given at end of sim: {debut_by_age}')
    print(f'Percent of live women age 25-49 who never have had sex: {abstinent/total_women_25_49}')
    print(f'Mean age of first (only live) birth for live women age age 25-49, including never given birth: {mean_first_birth_age} +/- {error_first_birth_age}')

if do_plot_debut:

    agents_25_49 = {}

    agents_25_49['uid'] = ppl.uid[inds_live_women_25_49]
    agents_25_49['debut_age'] = ppl.sexual_debut_age[inds_live_women_25_49]
    agents_25_49['debut_status'] = ppl.sexual_debut[inds_live_women_25_49]
    agents_25_49['age'] = ppl.age[inds_live_women_25_49]
    agents_25_49['parity'] = ppl.parity[inds_live_women_25_49]

    df_debut = pd.DataFrame(data=agents_25_49)

    df_debut = df_debut[df_debut.debut_age != 0]

    #df_debut.to_csv('/Users/Annie/model_postprocess_files/adolescents/sexual_debut'+'.csv')

    sns.histplot(data=df_debut, x="debut_age", binwidth=1)
    plt.show()

if do_plot_age_first:

    agents_10_49 = {}
    agents_10_49['uid'] = ppl.uid[inds_live_women_10_49]
    agents_10_49['age'] = ppl.age[inds_live_women_10_49]
    agents_10_49['parity'] = ppl.parity[inds_live_women_10_49]
    agents_10_49['age_first_birth'] = ppl.first_birth_age[inds_live_women_10_49]

    df_age_first = pd.DataFrame(data=agents_10_49)

    df_age_first.to_csv('/Users/Annie/model_postprocess_files/adolescents/age_first_birth' + '.csv')

    df_plot_age_first = df_age_first[df_age_first.parity > 0]

    sns.histplot(data=df_plot_age_first, x="age_first_birth", binwidth=1)
    plt.show()

if do_plot_nullip_ages:

    agents_15_49 = {}
    agents_15_49['uid'] = ppl.uid[inds_live_women_15_49]
    agents_15_49['age'] = ppl.age[inds_live_women_15_49]
    agents_15_49['parity'] = ppl.parity[inds_live_women_15_49]
    agents_15_49['age_first_birth'] = ppl.first_birth_age[inds_live_women_15_49]

    df_parity = pd.DataFrame(data=agents_15_49)
    df_parity.to_csv('/Users/Annie/model_postprocess_files/adolescents/agents_parity' + '.csv')

'''
if do_plot_method_use:

    agents_10_15 = {}
    agents_15_20 = {}
    agents_20_25 = {}
    agents_25_30 = {}
    agents_30_35 = {}
    agents_35_40 = {}
    agents_40_50 = {}
'''
sc.toc()





