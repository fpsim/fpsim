'''
Run all analyses for Senegal.
'''

import os
import pylab as pl
import pandas as pd
import sciris as sc
import seaborn as sns
import fpsim as lfp
import senegal_parameters as sp

# Housekeeping
sc.tic()
pl.rcParams['font.size'] = 10

# Set parameters
do_run              = 1
do_store_postpartum = 1
do_plot_pregnancy_parity = 1
do_plot_popsize     = 1
do_plot_pyramids    = 1
do_plot_model_pyramid = 1
do_plot_skyscrapers = 1
do_plot_methods     = 1
do_plot_spacing     = 1
do_save             = 0

min_age = 15
max_age = 50
bin_size = 5
year_str = '2017'


# Files
def datapath(path):
    ''' Return the path of the parent folder -- TODO: remove duplication with calibration.py'''
    return sc.thisdir(__file__, os.pardir, 'dropbox', path)

pregnancy_parity_file = datapath('SNIR80FL.DTA')  # DHS Senegal 2018 file
pop_pyr_year_file = datapath('Population_Pyramid_-_All.csv')
skyscrapers_file = datapath('Skyscrapers-All-DHS.csv')
methods_file = datapath('Method_v312.csv')
spacing_file = datapath('BirthSpacing.csv')
popsize_file = datapath('senegal-popsize.csv')
barriers_file = datapath('DHSIndividualBarriers.csv')


if do_run:
    pars = sp.make_pars()
    sim = lfp.Sim(pars=pars)

    def add_long_acting(sim):
        print('Added long-acting intervention')
        for person in sim.people.values():
            person.method = 'Implant'

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

    # sim = lfp.multi_run(sim, n=1)
    sim.run()
    people = list(sim.people.values()) # Pull out people

    #if do_store_postpartum:

        #pp = sim.store_postpartum()
        #pp.to_csv(sp.abspath('data/postpartum_model2.csv'))

    if do_plot_pregnancy_parity:

        #Extract data on currently pregnant and parity at end of sim from model
        model = sim.store_postpartum()

        #Load Senegal DHS 2018 data
        dhs = pd.read_stata(pregnancy_parity_file, convert_categoricals=False)
        dhs = dhs[['v012', 'v213', 'v218']]
        dhs = dhs.rename(columns={'v012': 'Age', 'v213': 'Pregnant',
                                  'v218': 'Parity'})  # Parity means # of living children in DHS

        fig, axes = pl.subplots(3, 2, figsize = (18, 14))

        #fig.suptitle('FP Sim Model vs DHS data on age, pregnancy, and parity')

        sns.distplot(model['Age'], bins=37, ax = axes[0,0]).set_title('Age histogram in FP model')
        sns.distplot(dhs['Age'], bins=35, ax = axes[0,1]).set_title('Age histogram in Senegal 2018 DHS data')

        sns.violinplot(ax = axes[1,0], x='Pregnant', y='Age', data=model).set_title(
            'Age distribution of agents currently pregnant in FP model')
        sns.violinplot(ax = axes[1,1], x='Pregnant', y='Age', data=dhs).set_title(
            'Age distribution currently pregnant in 2018 DHS data')

        sns.boxplot(ax = axes[2,0], x='Parity', y='Age', data=model).set_title('Age-parity distributions FP model')
        sns.boxplot(ax = axes[2,1], x='Parity', y='Age', data=dhs).set_title('Age-parity distributions 2018 DHS data')

        pl.tight_layout()

        if do_save:
            pl.savefig(sp.abspath('figs/pregnancy_parity.png'))

    if do_plot_popsize:

        # Load data
        popsize_tfr  = pd.read_csv(popsize_file, header=None)

        # Handle population size
        pop_years = popsize_tfr.iloc[0,:].to_numpy()

        popsize = popsize_tfr.iloc[1,:].to_numpy() / sim.pars['n'] # Conversion factor from Senegal to 500 people, = 1 / 1000 * 1.4268 / 500

        # Default plots
        fig = sim.plot()

        sim.plot_postpartum()

        # Population size plot
        ax = fig.axes[0] # First axis on plot
        ax.scatter(pop_years, popsize, c='k', label='Data', zorder=1000)
        #pl.legend()
        if do_save:
            pl.savefig(sp.abspath('figs/senegal_popsize.png'))

        # MCPR -- TODO, copied from senegal_parameters.py
        mcpr_years = pl.array([1986, 1992, 1997, 2005, 2010, 2012, 2014, 2015, 2016, 2017])
        mcpr_rates = pl.array([2.65, 4.53, 7.01, 7.62, 8.85, 11.3, 14.7, 15.3, 16.5, 18.8])

        ax = fig.axes[1] # Second axis on plot
        ax.scatter(mcpr_years, mcpr_rates, c='k', label='Data', zorder=1000)
        #pl.legend()

        #350434.0 <--- Factor previously used to adjust population

    if do_plot_pyramids:
        fig2 = pl.figure(figsize=(16,16))

        # Load population pyramid from DHS
        pop_pyr_year  = pd.read_csv(pop_pyr_year_file, header=None)
        pop_pyr_year = pop_pyr_year[pop_pyr_year[0]==year_str]
        bins = pl.arange(min_age, max_age, bin_size)
        pop_props_year = pop_pyr_year[2].to_numpy()

        plotstyle = {'marker':'o', 'lw':3}

        counts = pl.zeros(len(bins))
        for person in people:
            if person.alive:
                bininds = sc.findinds(bins<=person.age) # Could be refactored
                if len(bininds) and person.age < max_age:
                    counts[bininds[-1]] += 1
        counts = counts/counts.sum()

        # x = pl.hstack([bins, bins[::-1]])
        # PP = pl.hstack([pop_props_year,-pop_props_year[::-1]])
        # CC = pl.hstack([counts,-counts[::-1]])

        # pl.plot(PP, x, c='b', label=f'{year_str} data', **plotstyle)
        # pl.plot(CC, x, c='g', label='Model', **plotstyle)

        pl.plot(pop_props_year, bins, c='b', label=f'{year_str} data', **plotstyle)
        pl.plot(counts, bins, c='g', label='Model', **plotstyle)

        pl.legend()
        pl.xlabel('Proportion')
        pl.ylabel('Age')
        pl.title('Age pyramid, 15-49, females only', fontweight='bold')
        sc.setylim()

        if do_save:
            pl.savefig(sp.abspath('figs/senegal_pyramids.png'))

    if do_plot_skyscrapers:

        # Set up
        min_age = 15
        max_age = 50
        bin_size = 5
        age_bins = pl.arange(min_age, max_age, bin_size)
        parity_bins = pl.arange(0,8)
        n_age = len(age_bins)
        n_parity = len(parity_bins)
        x_age = pl.arange(n_age)
        x_parity = pl.arange(n_parity) # Should be the same

        # Load data
        data_parity_bins = pl.arange(0,18)
        sky_raw_data  = pd.read_csv(skyscrapers_file, header=None)
        sky_raw_data = sky_raw_data[sky_raw_data[0]==year_str]
        sky_parity = sky_raw_data[2].to_numpy()
        sky_props = sky_raw_data[3].to_numpy()
        sky_arr = sc.odict()

        sky_arr['Data'] = pl.zeros((len(age_bins), len(parity_bins)))
        count = -1
        for age_bin in x_age:
            for dpb in data_parity_bins:
                count += 1
                parity_bin = min(n_parity-1, dpb)
                sky_arr['Data'][age_bin, parity_bin] += sky_props[count]
        assert count == len(sky_props)-1 # Ensure they're the right length


        # Extract from model
        sky_arr['Model'] = pl.zeros((len(age_bins), len(parity_bins)))
        for person in people:
            if person.alive and not person.sex and person.age>=min_age and person.age<max_age:
                age_bin = sc.findinds(age_bins<=person.age)[-1]
                parity_bin = sc.findinds(parity_bins<=person.parity)[-1]
                sky_arr['Model'][age_bin, parity_bin] += 1

        # Normalize
        for key in ['Data', 'Model']:
            sky_arr[key] /= sky_arr[key].sum() / 100

        # Plot skyscrapers
        for key in ['Data', 'Model']:
            fig = pl.figure(figsize=(20,14))

            sc.bar3d(fig=fig, data=sky_arr[key], cmap='jet')
            pl.xlabel('Age', fontweight='bold')
            pl.ylabel('Parity', fontweight='bold')
            pl.title(f'Age-parity plot for the {key.lower()}\n\n', fontweight='bold')
            pl.gca().set_xticks(pl.arange(n_age))
            pl.gca().set_yticks(pl.arange(n_parity))
            pl.gca().set_xticklabels(age_bins)
            pl.gca().set_yticklabels(parity_bins)
            pl.gca().view_init(30,45)
            pl.draw()
            if do_save:
                pl.savefig(sp.abspath(f'figs/senegal_skyscrapers_{key}.png'))


        # Plot sums
        fig = pl.figure(figsize=(20,14))
        labels = ['Parity', 'Age']
        x_axes = [x_parity, x_age]
        x_labels = [['0','1','2','3','4','5','6', '7+'],
                    ['15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49']]
        offsets = [0, 0.4]
        for i in range(2):
            pl.subplot(2,1,i+1)
            for k,key in enumerate(['Data', 'Model']):
                y_data = sky_arr[key].sum(axis=i)
                # y_data = y_data/y_data.sum()
                pl.bar(x_axes[i]+offsets[k], y_data, width=0.4, label=key)
                pl.gca().set_xticks(x_axes[i]+0.2)
                pl.gca().set_xticklabels(x_labels[i])
                pl.xlabel(labels[i])
                pl.ylabel('Percentage of population')
                pl.title(f'Population by: {labels[i]}', fontweight='bold')
                pl.legend()
                if do_save:
                    pl.savefig(sp.abspath(f'figs/senegal_age_parity_sums.png'))


    if do_plot_methods:
        data_method_counts = sc.odict().make(keys=sim.pars['methods']['names'], vals=0.0)
        model_method_counts = sc.dcp(data_method_counts)

        # Load data from DHS -- from dropbox/Method_v312.csv

        data = [
            ['Other', 'emergency contraception', 0.015216411570543636,2017.698615635373],
            ['Condoms', 'female condom', 0.005239036180154552,2017.698615635373],
            ['BTL', 'female sterilization', 0.24609377594176307,2017.698615635373],
            ['Implants', 'implants/norplant', 5.881839602070953,2017.698615635373],
            ['Injectables', 'injections', 7.101718239287355,2017.698615635373],
            ['IUDs', 'iud', 1.4865067612487317,2017.698615635373],
            ['Other', 'lactational amenorrhea (lam)', 0.04745447091361792,2017.698615635373],
            ['Condoms', 'male condom', 1.0697377418682412,2017.698615635373],
            ['None', 'not using', 80.10054235699272,2017.698615635373],
            ['Other', 'other modern method', 0.007832257135437748,2017.698615635373],
            ['Other', 'other traditional', 0.5127850142889963,2017.698615635373],
            ['Rhythm', 'periodic abstinence', 0.393946698444533,2017.698615635373],
            ['Pill', 'pill', 2.945874450486654,2017.698615635373],
            ['Rhythm', 'standard days method (sdm)', 0.06132534128612159,2017.698615635373],
            ['Withdrawal', 'withdrawal', 0.12388784228417069,2017.698615635373],
        ]


        '''
        Use to compare to old switching matrices
        data = [
            ['Other', 'emergency contraception', 0.015216411570543636,2017.698615635373],
            ['Condoms', 'female condom', 0.005239036180154552,2017.698615635373],
            ['Other', 'female sterilization', 0.24609377594176307,2017.698615635373],
            ['Implants', 'implants/norplant', 5.881839602070953,2017.698615635373],
            ['Injectables', 'injections', 7.101718239287355,2017.698615635373],
            ['IUDs', 'iud', 1.4865067612487317,2017.698615635373],
            ['Lactation', 'lactational amenorrhea (lam)', 0.04745447091361792,2017.698615635373],
            ['Condoms', 'male condom', 1.0697377418682412,2017.698615635373],
            ['None', 'not using', 80.10054235699272,2017.698615635373],
            ['Other', 'other modern method', 0.007832257135437748,2017.698615635373],
            ['Traditional', 'other traditional', 0.5127850142889963,2017.698615635373],
            ['Traditional', 'periodic abstinence', 0.393946698444533,2017.698615635373],
            ['Pill', 'pill', 2.945874450486654,2017.698615635373],
            ['Other', 'standard days method (sdm)', 0.06132534128612159,2017.698615635373],
            ['Traditional', 'withdrawal', 0.12388784228417069,2017.698615635373],
        ]
        '''
        for entry in data:
            data_method_counts[entry[0]] += entry[2]
        data_method_counts[:] /= data_method_counts[:].sum()

        # From model
        for person in people:
            if person.alive and not person.sex and person.age>=min_age and person.age<max_age:
                model_method_counts[person.method] += 1
        model_method_counts[:] /= model_method_counts[:].sum()

        # Make labels
        data_labels = data_method_counts.keys()
        for d in range(len(data_labels)):
            if data_method_counts[d]>0.01:
                data_labels[d] = f'{data_labels[d]}: {data_method_counts[d]*100:0.1f}%'
            else:
                data_labels[d] = ''
        model_labels = model_method_counts.keys()
        for d in range(len(model_labels)):
            if model_method_counts[d]>0.01:
                model_labels[d] = f'{model_labels[d]}: {model_method_counts[d]*100:0.1f}%'
            else:
                model_labels[d] = ''

        # Plot pies
        fig = pl.figure(figsize=(20,14))
        pl.subplot(2,2,1)
        pl.pie(data_method_counts[:], labels=data_labels)
        pl.title('Method use (DHS data)', fontweight='bold')

        pl.legend(data_method_counts.keys(), loc='upper right', bbox_to_anchor=(1.7,0.2))

        pl.subplot(2,2,2)
        pl.pie(model_method_counts[:], labels=model_labels)
        pl.title('Method use (model)', fontweight='bold')

        # Remake without None
        data_method_counts['None'] = 0.0
        model_method_counts['None'] = 0.0
        data_method_counts[:] /= data_method_counts[:].sum()
        model_method_counts[:] /= model_method_counts[:].sum()

        # Make labels
        data_labels = data_method_counts.keys()
        for d in range(len(data_labels)):
            if data_method_counts[d]>0.01:
                data_labels[d] = f'{data_labels[d]}: {data_method_counts[d]*100:0.1f}%'
            else:
                data_labels[d] = ''
        model_labels = model_method_counts.keys()
        for d in range(len(model_labels)):
            if model_method_counts[d]>0.01:
                model_labels[d] = f'{model_labels[d]}: {model_method_counts[d]*100:0.1f}%'
            else:
                model_labels[d] = ''

        pl.subplot(2,2,3)
        pl.pie(data_method_counts[:], labels=data_labels)
        pl.title('Method use among users (DHS data)', fontweight='bold')

        # pl.legend(data_method_counts.keys(), loc='upper right', bbox_to_anchor=(1,1.5))

        pl.subplot(2,2,4)
        pl.pie(model_method_counts[:], labels=model_labels)
        pl.title('Method use among users (model)', fontweight='bold')

        if do_save:
            pl.savefig(sp.abspath(f'figs/senegal_method_mix.png'))


    if do_plot_spacing:

        spacing_bins = sc.odict({'0-12':0,'12-24':1,'24-36':2,'>36':3}) # Spacing bins in years

        # From data
        data = pd.read_csv(spacing_file)

        right_year = data['SurveyYear']=='2010-11'
        not_first = data['Birth Order'] != 0
        is_first = data['Birth Order'] == 0
        filtered = data[(right_year) & (not_first)]
        spacing = filtered['Birth Spacing'].to_numpy()
        sorted_spacing = sorted(spacing)

        first_filtered = data[(right_year) & (is_first)]
        first = first_filtered['Birth Spacing'].to_numpy()
        sorted_first = sorted(first)

        x_ax = pl.linspace(0,100,len(sorted_spacing))
        x_ax_first = pl.linspace(0,100,len(sorted_first))

        # From model
        model_age_first = []
        model_spacing = []
        model_spacing_counts = sc.odict().make(keys=spacing_bins.keys(), vals=0.0)
        for person in people:
            if len(person.dobs):
                model_age_first.append(person.dobs[0])
            if len(person.dobs)>1:
                for d in range(len(person.dobs)-1):
                    space = person.dobs[d+1] - person.dobs[d]
                    ind = sc.findinds(space>spacing_bins[:])[-1]
                    model_spacing_counts[ind] += 1

                    model_spacing.append(space)

        model_ax = pl.linspace(0,100,len(model_spacing))

        model_age_ax = pl.linspace(0,100,len(model_age_first))

        # Plotting
        fig = pl.figure(figsize=(30,12))

        # tmpfig = pl.figure()
        # data_y, data_x, _ = pl.hist(spacing, bins=20)
        # model_y, model_x, _ = pl.hist(model_spacing, bins=20)
        # pl.close(tmpfig)
        # data_y /= data_y.sum()/100.0
        # model_y /= model_y.sum()/100.0
        # pl.plot(data_x[:-1], data_y, c='k', label='Data', lw=3)
        # pl.plot(model_x[:-1], model_y, c=(0.2,0.7,0.1), label='Model', lw=3)

        pl.subplot(1,2,1)
        pl.plot(x_ax, sorted(spacing), c='k', label='Data', lw=3)
        pl.plot(model_ax, sorted(model_spacing), c=(0.2,0.7,0.1), label='Model', lw=3)
        pl.xlim([0,100.1])
        sc.setylim()
        pl.xlabel('Probability (%)')
        pl.ylabel('Birth spacing (years)')
        pl.title(f'Birth spacing is: data={pl.mean(spacing):0.2f}±{pl.std(spacing):0.2f} years, model={pl.mean(model_spacing):0.2f}±{pl.std(model_spacing):0.2f} years')
        pl.legend()

        pl.subplot(1,2,2)
        pl.plot(x_ax_first, sorted(first), c='k', label='Data', lw=3)
        pl.plot(model_age_ax, sorted(model_age_first), c=(0.2,0.7,0.1), label='Model', lw=3)
        pl.xlim([0,100.1])
        sc.setylim()
        pl.xlabel('Probability (%)')
        pl.ylabel('Age at first birth (years)')
        pl.title(f'Age at first birth: data={pl.mean(first):0.2f}±{pl.std(first):0.2f} years, model={pl.mean(model_age_first):0.2f}±{pl.std(model_age_first):0.2f} years')
        pl.legend()

        if do_save:
            pl.savefig(sp.abspath(f'figs/senegal_birth_spacing.png'))





sc.toc()

print('Done.')

