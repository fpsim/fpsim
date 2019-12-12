# Simple example usage for LEMOD-FP

import pylab as pl
import pandas as pd
import sciris as sc
import lemod_fp as lfp

# Set parameters
do_run = True
do_plot = True
do_save = False
pop_pyr_1982_fn = 'senegal-population-pyramid-1982.csv'
pop_pyr_2015_fn = 'senegal-population-pyramid-2015.csv'
popsize_tfr_fn = 'senegal-popsize-tfr.csv'

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
        ax.scatter(years, popsize, c='k', label='Data')
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
        people = sim.people.values()
        for person in people:
            binind = sc.findinds(bins<=person.age)[-1]
            counts[binind] += 1
        counts = counts/counts.sum()
        pl.plot(bins, counts, c='g', label='Model', **plotstyle)
        pl.legend()
        pl.xlabel('Age')
        pl.ylabel('Proportion')
            
    
print('Done.')

