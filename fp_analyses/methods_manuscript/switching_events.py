'''
A script to run a sim and calculate the total transitions between methods in a square matrix in the last 10 years of the sim
Separates both annual transitions and postpartum transitions
Can adjust how many timesteps to sum and output
'''

import pandas as pd
import fpsim as fp
from fpsim import defaults as fpd
from fp_analyses import senegal_parameters as sp
from fp_analyses.methods_manuscript import base
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

mpy = 12 # months per year to avoid magic numbers

y = 10 # number of years to go backwards from end of simulation to calculate transitions

# Run FPsim and extract results
pars = sp.make_pars()
pars['n'] = 2000 # Adjust running parameters easily here if needed
#pars['start_year'] = 1960
sim = fp.Sim(pars=pars)
sim.run()
results = sim.results

# Save important parameters for use in calculation
m = len(pars['methods']['map'])
timestep = pars['timestep']
npts = int(mpy * (pars['end_year'] - pars['start_year']) / timestep + 1)
names = pars['methods']['map'].keys()

# Initialize matrices of zeros to start adding transitions to
annual = np.zeros((m, m), dtype=int)
postpartum = np.zeros((m, m), dtype=int)
annual_ages = {}
postpartum_ages = {}

for key in fpd.method_age_mapping.keys():
    annual_ages[key] = np.zeros((m, m), dtype=int)
    postpartum_ages[key] = np.zeros((m, m), dtype=int)

a_steps = {}
p_steps = {}

# Range through timesteps and add transitions at each timestep
start_timestep = int(npts - ((y * (mpy/timestep)-1)))
stop_timestep = int(npts - 1)
print(f'Start timestep: {start_timestep}')
print(f'Stop timestep: {stop_timestep}')
step = 1
for i in range(start_timestep, stop_timestep):
    annual += results["switching_events_general"][i]
    postpartum += results["switching_events_postpartum"][i]
    for key in fpd.method_age_mapping.keys():
        annual_ages[key] += results["switching_events_" + key][i]
        print(f'i in range start to stop: {i}')
        postpartum_ages[key] += results["switching_events_pp_" + key][i]
    a_df_step = pd.DataFrame(results["switching_events_general"][i], columns=names, index=names) # only need next 5 lines for animations
    p_df_step = pd.DataFrame(results["switching_events_postpartum"][i], columns=names, index=names)
    a_steps[step] = a_df_step
    p_steps[step] = p_df_step
    step += 1

end_step = step

# Convert to dataframes to add method names and convert to log scale for better visualization
annual_df = pd.DataFrame(annual, columns=names, index=names)
postpartum_df = pd.DataFrame(postpartum, columns=names, index=names)
annual_ages_df = {}
postpartum_ages_df = {}
annual_ages_log = {}
postpartum_ages_log = {}

for key in fpd.method_age_mapping.keys():
    annual_ages_df[key] = pd.DataFrame(annual_ages[key], columns=names, index=names)
    annual_ages_log[key] = np.log10(annual_ages_df[key])
    postpartum_ages_df[key] = pd.DataFrame(postpartum_ages[key], columns=names, index=names)
    postpartum_ages_log[key] = np.log10(postpartum_ages_df[key])

fix, axs = plt.subplots(4,1)
for x in range(4):
    axs[x, 0].sns.heatmap(annual_ages_log[key], vmin=0, vmax=5.6, cmap = "YlGnBu", cbar_kws={'label': 'number of transitions (log 10)'})
    plt.title(f'Age {key}')

plt.show()





'''Uncomment blocks of code below if you'd like to print switching matrices out'''
# Print matrices
#new_line = '\n'
#print(f'Annual switching event matrix for the last {y} years of the simulation:{new_line}{annual_df}')
#print(f'Postpartum switching event matrix for the last {y} years of the simulation{new_line}{postpartum_df}')

#print(f'Annual switching event matrices for the last {y} years of the simulation by age group:{new_line}')
#for key in fpd.method_age_mapping.keys():
    #print(f'{key}: {annual_ages_df[key]}')

#print(f'Postpartum switching event matrices for the last {y} years of the simulation by age group:{new_line}')
#for key in fpd.method_age_mapping.keys():
    #print(f'{key}: {postpartum_ages_df[key]}')

'''Uncomment blocks of code below if you'd like to save switching events as csv'''
# Save matrices
#annual_df.to_csv('/model_postprocess_files/annual_switching.csv')  # Change filepath to yours
#postpartum_df.to_csv('/model_postprocess_files/postpartum_switching.csv') # Change filepath to yours

#for key in fpd.method_age_mapping.keys():
    #annual_ages_df[key].to_csv('/model_postprocess_files/annual_switching_age_'+key+'.csv')
    #postpartum_ages_df[key].to_csv('/model_postprocess_files/postpartum_switching_age_' + key + '.csv')

# Save each step matrix to be able to cycle through in animation
'''
Uncomment this code if you want to save all switching matrices at each time step
NOTE - this will save ~120 dataframes in separate csv files twice for annual and postpartum steps
'''
#for n in range(1, end_step):
    #a_steps[n].to_csv('annual_switching_step_'+str(n)+'.csv')
    #p_steps[n].to_csv('/postpartum_switching_step_'+str(n)+'.csv')

