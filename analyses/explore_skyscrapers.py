'''
Explore alternatives to skyscrapers
'''

import numpy as np
import sciris as sc
import pylab as pl
import fpsim as fp

T = sc.timer()
do_save = True


#%% Load data

filename = sc.thispath(fp) / 'locations/ethiopia/ethiopia_skyscrapers.csv'
df = sc.dataframe.read_csv(filename) # Load data
dataset = ['DHS 2011', 'PMA 2019'][1] # Choose which dataset to use
df = df[df.dataset==dataset] # Filter data


#%% Wrangle data

# Map ages onto indices for plotting
agemap = {k:i for i,k in enumerate(df.age.unique())}
agemap_r = {i:k for k,i in agemap.items()}
age_inds = [agemap[k] for k in df.age.values]
df['age_ind'] = age_inds

# Construct 2D array
n_parity = df.parity.max() + 1
n_age = df.age_ind.max() + 1
ages = np.arange(n_age)
arr = np.zeros((n_parity, n_age))
for row in df.itertuples():
    arr[row.parity, row.age_ind] = row.percentage


norm_arr = arr.copy()
for a in ages:
    y = arr[:,a]
    norm_arr[:,a] = y*100/y.sum()


#%% Plotting

sc.options(dpi=150)
fig = pl.figure(figsize=(18,12))
tkw = dict(fontweight='bold')

for row,data in enumerate([arr, norm_arr]):
    
    suffix = ' (normalized)' if row else ''


    # Option 1: Skyscrapers
    ax1 = pl.subplot(2,3,1+3*row, projection='3d')
    sc.bar3d(data.T, ax=ax1, cmap='turbo', axkwargs=dict(azim=45))
    pl.xlabel('Age')
    pl.ylabel('Parity')
    pl.xticks(ages, agemap.keys())
    pl.title('Option 1: skyscrapers'+suffix, **tkw)
    
    # Option 2: Heat map
    ax2 = pl.subplot(2,3,2+3*row)
    pl.imshow(data, cmap='turbo', origin='lower')
    pl.xlabel('Age')
    pl.ylabel('Parity')
    pl.xticks(ages, agemap.keys())
    pl.title('Option 2: heatmap'+suffix, **tkw)
    pl.colorbar()
    
    # Option 3: Stacked lines
    ax3 = pl.subplot(2,3,3+3*row)
    colors = sc.vectocolor(ages, cmap='parula')
    for a in ages:
        y = data[:,a]
        pl.plot(y, 'o-', c=colors[a], label='Age '+agemap_r[a], lw=3, alpha=0.7)
    
    pl.legend(frameon=False)
    pl.xlabel('Parity')
    pl.ylabel('Percentage')
    pl.title('Option 3: Line plots'+suffix, **tkw)
    sc.boxoff(ax3)
    sc.setylim(ax=ax3)


#%% Tidy up

sc.figlayout(top=0.95, bottom=0.05, left=0.05, right=0.95)
pl.show()
if do_save:
    sc.savefig('skyscraper-options.png')

T.toc('Done')