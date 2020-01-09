# Run all analyses for Senegal

import pylab as pl
import pandas as pd
import sciris as sc
import lemod_fp as lfp
import senegal_parameters as sp

# Set parameters
do_run              = 1
do_plot_popsize     = 0
do_plot_pyramids    = 0
do_plot_skyscrapers = 1
do_save             = 0

pl.seed(1)

if do_run:
    pars = sp.make_pars()
    sim = lfp.Sim(pars=pars)
    
    def add_long_acting(sim):
        print('Added long-acting intervention')
        for person in sim.people.values():
            person.method = 'implant'
    
    def urhi(sim):
        print('Added URHI')
        switching = sim.pars['switching']
        print(switching)
        for i,method1 in enumerate(switching.keys()):
            switching[method1][0] *= 0.0
            switching[method1][:] = switching[method1][:]/switching[method1][:].sum()
        sim.pars['switching'] = switching
        print(switching)
        for person in sim.people.values():
            person.pars = sim.pars

    def serialize(sim):
        sc.saveobj(sp.abspath('serialized_pop.obj'), sim.people)

    def deserialize(sim):
        sim.people = sc.loadobj(sp.abspath('serialized_pop.obj'))

    
    # sim.add_intervention(intervention=add_long_acting, year=2000)
    # sim.add_intervention(intervention=urhi, year=2000)
    # sim.add_intervention(intervention=serialize, year=2000)
    # sim.add_intervention(intervention=deserialize, year=2010)
    sim.run()
    people = list(sim.people.values()) # Pull out people
    
    if do_plot_popsize:
        
        # Default plots
        fig = sim.plot()
        
        # Population size plot
        ax = fig.axes[-1] 
        ax.scatter(sp.years, sp.popsize, c='k', label='Data', zorder=1000)
        pl.legend()
        if do_save:
            pl.savefig(sp.abspath('figs/senegal_popsize.png'))
    
    if do_plot_pyramids:
        fig2 = pl.figure(figsize=(16,16))
        
        # Load 2015 population pyramid from DHS
        min_age = 15
        max_age = 50
        bin_size = 5
        pop_pyr_2015_fn = sp.abspath('dropbox/Population_Pyramid_-_All.csv')
        pop_pyr_2015  = pd.read_csv(pop_pyr_2015_fn, header=None)
        pop_pyr_2015 = pop_pyr_2015[pop_pyr_2015[0]=='2017']
        bins = pl.arange(min_age, max_age, bin_size)
        pop_props_2015 = pop_pyr_2015[2].to_numpy()
        
        plotstyle = {'marker':'o', 'lw':3}
        
        counts = pl.zeros(len(bins))
        for person in people:
            if person.alive:
                bininds = sc.findinds(bins<=person.age) # Could be refactored
                if len(bininds) and person.age < max_age:
                    counts[bininds[-1]] += 1
        counts = counts/counts.sum()
        
        x = pl.hstack([bins, bins[::-1]])
        PP = pl.hstack([pop_props_2015,-pop_props_2015[::-1]])
        CC = pl.hstack([counts,-counts[::-1]])
        
        pl.plot(PP, x, c='b', label='2015 data', **plotstyle)
        pl.plot(CC, x, c='g', label='Model', **plotstyle)
        
        pl.legend()
        pl.xlabel('Proportion')
        pl.ylabel('Age')
        sc.setylim()
        
        if do_save:
            pl.savefig(sp.abspath('figs/senegal_pyramids.png'))
            
       

    if do_plot_skyscrapers:
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
        
        women = sc.loadobj(sp.abspath('../fp_data/URHI/senegal_parity_data.obj'))
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
        n_age = len(age_bins)
        n_parity = len(parity_bins)
        x_age = pl.arange(n_age)
        x_parity = pl.arange(n_parity) # Should be the same
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
        pl.gca().set_xticks(pl.arange(n_age))
        pl.gca().set_yticks(pl.arange(n_parity))
        pl.gca().set_xticklabels(age_bins)
        pl.gca().set_yticklabels(parity_bins)
        # pl.gca().set_xlim([-1,7])
        
        # Age-parity
        fig = pl.figure(figsize=(20,14))
        pl.subplot(2,1,1)
        parity_data = data.sum(axis=0)
        parity_data = parity_data/parity_data.sum()
        pl.bar(x_parity, parity_data, width=0.4, label='Model')
        parity_urhi = all_women_age_parity.sum(axis=0)
        parity_urhi = parity_urhi/parity_urhi.sum()
        pl.bar(x_parity+0.4, parity_urhi, width=0.4, label='Data')
        pl.xlabel('Parity')
        pl.legend()
        
        pl.subplot(2,1,2)
        age_data = data.sum(axis=1)
        age_data = age_data/age_data.sum()
        pl.bar(x_age, age_data, width=0.4, label='Model')
        age_urhi = all_women_age_parity.sum(axis=1)
        age_urhi[-1] = 0 # For similarity with model, exlude >50
        age_urhi = age_urhi/age_urhi.sum()
        pl.bar(x_age+0.4, age_urhi, width=0.4, label='Data')
        pl.gca().set_xticks(x_age)
        pl.gca().set_xticklabels(age_bins)
        pl.legend()
        pl.xlabel('Age')
    
    
            
    
print('Done.')

