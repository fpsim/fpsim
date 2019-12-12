# Simple example usage for LEMOD-FP

import pylab as pl
import pandas as pd
import sciris as sc
import lemod_fp as lfp

# Set parameters
do_run = True
do_plot = True
do_save = False
do_skyscrapers = True
pop_pyr_1982_fn = 'data/senegal-population-pyramid-1982.csv'
pop_pyr_2015_fn = 'data/senegal-population-pyramid-2015.csv'
popsize_tfr_fn = 'data/senegal-popsize-tfr.csv'

# Load data
pop_pyr_1982 = pd.read_csv(pop_pyr_1982_fn)
pop_pyr_2015 = pd.read_csv(pop_pyr_2015_fn)
popsize_tfr  = pd.read_csv(popsize_tfr_fn, header=None)

# Handle population size
scale_factor = 1000
years = popsize_tfr.iloc[0,:].to_numpy()
popsize = popsize_tfr.iloc[1,:].to_numpy() / scale_factor

if do_run:
    sim = lfp.Sim()
    sim.run()
    if do_plot:
        
        # Default plots
        fig = sim.plot(dosave=do_save)
        
        # Population size plot
        ax = fig.axes[-1] 
        ax.scatter(years, popsize, c='k', label='Data', zorder=1000)
        pl.legend()
    
        # Age pyrmaid
        fig2 = pl.figure(figsize=(20,14))
        
        M = pop_pyr_2015.loc[:,'M'].to_numpy()
        F = pop_pyr_2015.loc[:,'F'].to_numpy()
        M = M/M.sum()
        F = F/F.sum()
        bins = 5*pl.arange(len(M))
        
        plotstyle = {'marker':'o', 'lw':2}
        
        pl.plot(bins, M, c='b', label='Males', **plotstyle)
        pl.plot(bins, F, c='r', label='Females', **plotstyle)
        
        counts = pl.zeros(len(bins))
        people = list(sim.people.values())
        for person in people:
            if person.alive:
                binind = sc.findinds(bins<=person.age)[-1]
                counts[binind] += 1
        counts = counts/counts.sum()
        pl.plot(bins, counts, c='g', label='Model', **plotstyle)
        pl.legend()
        pl.xlabel('Age')
        pl.ylabel('Proportion')
       

if do_skyscrapers:
    def skyscraper(data, label=None, fig=None, nrows=None, ncols=None, idx=None, figkwargs=None, axkwargs=None):
        
        age_edges = list(range(15,55,5)) + [99]
        data['AgeBin'] = pd.cut(data['Age'], bins = age_edges, right=False)
        
        parity_edges = list(range(6+1)) + [99]
        data['ParityBin'] = pd.cut(data['Parity'], bins = parity_edges, right=False)

        data['AgeBinCode'] = data['AgeBin'].cat.codes
        data['ParityBinCode'] = data['ParityBin'].cat.codes
        age_bin_codes = pl.array(sorted(list(data['AgeBinCode'].unique())))
        parity_bin_codes = pl.array(sorted(list(data['ParityBinCode'].unique())))
        age_parity_data = pl.zeros((len(age_bin_codes), len(parity_bin_codes)))
        
        for i,row in data.iterrows():
            age_index = row['AgeBinCode']
            parity_index = row['ParityBinCode']
            weight = row['Weight']
            age_parity_data[age_index, parity_index] += weight
        
        axkwargs = dict(elev=37, azim=-31, nrows=nrows, ncols=ncols, index=idx)
        ax = sc.bar3d(fig=fig, data=age_parity_data, cmap='jet', axkwargs=axkwargs)
        age_bin_labels = list(data['AgeBin'].cat.categories)
        age_bin_labels[-1] = f'{age_edges[-2]}+'
        ax.set_xlabel('Age')
        ax.set_xticks(age_bin_codes+0.5) # To center the tick marks
        ax.set_xticklabels(age_bin_labels)
    
        parity_bin_labels = parity_edges[:-1]
        parity_bin_labels[-1] = f'{parity_edges[-2]}+'
        ax.set_ylabel('Parity')
        ax.set_yticks(parity_bin_codes+0.5)
        ax.set_yticklabels(parity_bin_labels)
    
        ax.set_zlabel('Women (weighted)')
        ax.set_title(label)
        return age_parity_data
    
    women = sc.loadobj('../fp_data/UHRI/senegal_parity_data.obj')
    parity_data = women.copy(deep=True)
    exclude_missing_parity = True
    if exclude_missing_parity:
        original_size = parity_data.shape[0]
        parity_data = parity_data.dropna(subset=['Parity'])
        new_size= parity_data.shape[0]
        print(f'Dropped {original_size-new_size} rows out of {original_size} due to missing parity data')
        print(parity_data['Parity'].unique())
    
    fig = pl.figure(figsize=(20,14))
    nrows = 1#2
    ncols = 1#(women['MethodClass'].nunique()+1) // nrows + 1
    idx = 0
#    for label, raw in parity_data.groupby('MethodClass'):
#        idx += 1
#        data = raw.copy(deep=True) # Just to be safe
#        skyscraper(data=data, label=label, fig=fig, nrows=nrows, ncols=ncols, idx=idx)
    
    all_women_age_parity = skyscraper(data=parity_data, label='All women', fig=fig, nrows=nrows, ncols=ncols, idx=idx+1)
    
    # Plot data
    age_bins = pl.arange(15,55,5)
    parity_bins = pl.arange(0,7)
    data = pl.zeros((len(age_bins), len(parity_bins)))
    for person in people:
        if not person.sex and person.age>=15 and person.age<50:
            age_bin = sc.findinds(age_bins<=person.age)[-1]
            parity_bin = sc.findinds(parity_bins<=person.parity)[-1]
            data[age_bin, parity_bin] += 1
    
    fig = pl.figure(figsize=(20,14))
    sc.bar3d(fig=fig, data=data, cmap='jet')
    pl.xlabel('Age')
    pl.ylabel('Parity')
    pl.gca().set_xticklabels(age_bins)
            
    
print('Done.')

