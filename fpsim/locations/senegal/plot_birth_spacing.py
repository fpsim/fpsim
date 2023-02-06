'''
A script to specifically look at data comparison to birth spacing
Birth spacing data comes from senegal.py in the format .obj
No other calibrated location at this point has a birth spacing file
'''

import numpy as np
import pandas as pd
import sciris as sc
import fpsim as fp
import pylab as pl
import numpy as np

do_plot_sim = 1

spacing_file = 'BirthSpacing.obj'

min_age = 15
max_age = 50

# Load birth spacing data
data = sc.load(spacing_file)

# Set up sim for Senegal
pars = fp.pars(location='senegal')
pars['n_agents'] = 100_000 # Population size
pars['end_year'] = 2020 # 1961 - 2020 is the normal date range

# Free parameters for calibration
pars['fecundity_var_low'] = 0.7
pars['fecundity_var_high'] = 1.1

sim = fp.Sim(pars=pars)
sim.run()

if do_plot_sim:
    sim.plot()


spacing_bins = sc.odict({'0-12': 0, '12-24': 1, '24-48': 2, '>48': 4})  # Spacing bins in years
data_dict = {}
model_dict = {} # For comparison from model to data

# Extract birth spacing data from  data
spacing, first = data['spacing'], data['first'] # Extract from data
data_spacing_counts = sc.odict().make(keys=spacing_bins.keys(), vals=0.0)

# Spacing bins from data
spacing_bins_array = sc.cat(spacing_bins[:], np.inf)
for i in range(len(spacing_bins_array)-1):
    lower = spacing_bins_array[i]
    upper = spacing_bins_array[i+1]
    matches = np.intersect1d(sc.findinds(spacing >= lower), sc.findinds(spacing < upper))
    data_spacing_counts[i] += len(matches)

data_spacing_counts[:] /= data_spacing_counts[:].sum()
data_spacing_counts[:] *= 100
data_spacing_stats = np.array([pl.percentile(spacing, 25),
                                pl.percentile(spacing, 50),
                                pl.percentile(spacing, 75)])
data_age_first_stats = np.array([pl.percentile(first, 25),
                                    pl.percentile(first, 50),
                                    pl.percentile(first, 75)])

# Save to dictionary
data_dict['spacing_bins'] = np.array(data_spacing_counts.values())
data_dict['spacing_stats'] = data_spacing_stats
data_dict['age_first_stats'] = data_age_first_stats

# From model
model_age_first = []
model_spacing = []
model_spacing_counts = sc.odict().make(keys=spacing_bins.keys(), vals=0.0)
ppl = sim.people
for i in  range(len(ppl)):
    if ppl.alive[i] and not ppl.sex[i] and ppl.age[i] >= min_age and ppl.age[i] < max_age:
        if len(ppl.dobs[i]):
            model_age_first.append(ppl.dobs[i][0])
        if len(ppl.dobs[i]) > 1:
            for d in range(len(ppl.dobs[i]) - 1):
                space = ppl.dobs[i][d + 1] - ppl.dobs[i][d]
                ind = sc.findinds(space > spacing_bins[:])[-1]
                model_spacing_counts[ind] += 1
                model_spacing.append(space)

model_spacing_counts[:] /= model_spacing_counts[:].sum()
model_spacing_counts[:] *= 100
try:
    model_spacing_stats = np.array([np.percentile(model_spacing, 25),
                                    np.percentile(model_spacing, 50),
                                    np.percentile(model_spacing, 75)])
    model_age_first_stats = np.array([np.percentile(model_age_first, 25),
                                    np.percentile(model_age_first, 50),
                                    np.percentile(model_age_first, 75)])
except Exception as E: # pragma: nocover
        print(f'Could not calculate birth spacing, returning zeros: {E}')
        model_spacing_counts = {k:0 for k in spacing_bins.keys()}
        model_spacing_stats = np.zeros(data_spacing_stats.shape)
        model_age_first_stats = np.zeros(data_age_first_stats.shape)

# Save arrays to dictionary
model_dict['spacing_bins'] = np.array(model_spacing_counts.values())
model_dict['spacing_stats'] = model_spacing_stats
model_dict['age_first_stats'] = model_age_first_stats

#diff_dict = sc.odict().make(keys=model_dict.keys(), vals=(model_dict.values() - data_dict.values()))

diff = model_dict['spacing_bins'] - data_dict['spacing_bins']

res_bins = np.array([[model_dict['spacing_bins']], [data_dict['spacing_bins']], [diff]])

bins_frame = pd.DataFrame({'Model': model_dict['spacing_bins'], 'Data': data_dict['spacing_bins'], 'Diff': diff}, index=spacing_bins.keys())

print(bins_frame)


# Plot spacing bins
ax = bins_frame.plot.barh(color={'Data':'black', 'Model':'cornflowerblue', 'Diff':'red'})
ax.set_xlabel('Percent of live birth spaces')
ax.set_ylabel('Birth space in months')
ax.set_title('Birth space bins calibration - Senegal')

pl.savefig("birth_space_bins.png", bbox_inches='tight', dpi=100)