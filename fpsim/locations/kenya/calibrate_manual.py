'''
The very start of a script for running plotting to compare the model to data
'''
import numpy as np
import pandas as pd
import sciris as sc
import fpsim as fp
import pylab as pl


sc.tic()

# Set options for plotting
do_plot_sim = True
do_plot_asfr = True
do_plot_methods = True
do_plot_skyscrapers = True


# Set up global variables
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

min_age = 15
max_age = 50
bin_size = 5

skyscrapers = pd.read_csv('kenya_skyscrapers.csv')
use = pd.read_csv('use_kenya.csv')

dataset = 'PMA 2022'  # Data to compare to for skyscrapers


# Set up sim for Kenya
pars = fp.pars(location='kenya')
pars['n_agents'] = 100_000 # Small population size
pars['end_year'] = 2020 # 1961 - 2020 is the normal date range

# Free parameters for calibration
pars['fecundity_var_low'] = 0.95
pars['fecundity_var_high'] = 1.05

sim = fp.Sim(pars=pars)
sim.run()

if do_plot_sim:
    sim.plot()

# Save results
res = sim.results


if do_plot_asfr:
        for key in age_bin_map.keys():
            print(f'ASFR (annual) for age bin {key} in the last year of the sim: {res["asfr"][key][-1]}')

        x = [1, 2, 3, 4, 5, 6, 7, 8]
        asfr_data = [3.03, 64.94, 172.12, 174.12, 136.10, 80.51, 34.88, 13.12]  # From UN Data Kenya 2020 (kenya_asfr.csv)
        x_labels = []
        asfr_model = []

        for key in age_bin_map.keys():
                x_labels.append(key)
                asfr_model.append(res['asfr'][key][-1])

        fig, ax = pl.subplots()
        kw = dict(lw=3, alpha=0.7, markersize=10)
        ax.plot(x, asfr_data, marker='^', color='black', label="UN data", **kw)
        ax.plot(x, asfr_model, marker='*', color='cornflowerblue', label="FPsim", **kw)
        pl.xticks(x, x_labels)
        pl.ylim(bottom=-10)
        ax.set_title('Age specific fertility rate per 1000 woman years')
        ax.set_xlabel('Age')
        ax.set_ylabel('ASFR in 2019')
        ax.legend(frameon=False)
        sc.boxoff()
        pl.savefig('figs/asfr.png')
        pl.show()

if do_plot_methods:

        methods_map_model = {  # Index, modern, efficacy
        'None': [0, False, 0.000],
        'Withdrawal': [1, False, 0.866],
        'Other traditional': [2, False, 0.861],
        # 1/2 periodic abstinence, 1/2 other traditional approx.  Using rate from periodic abstinence
        'Condoms': [3, True, 0.946],
        'Pill': [4, True, 0.945],
        'Injectables': [5, True, 0.983],
        'Implants': [6, True, 0.994],
        'IUDs': [7, True, 0.986],
        'BTL': [8, True, 0.995],
        'Other modern': [9, True, 0.880],
        }

        model_labels_all = list(methods_map_model.keys())
        model_labels_methods = sc.dcp(model_labels_all)
        model_labels_methods = model_labels_methods[1:]

        model_method_counts = sc.odict().make(keys=model_labels_all, vals=0.0)

        # From model
        ppl = sim.people
        for i in range(len(ppl)):
                if ppl.alive[i] and not ppl.sex[i] and ppl.age[i] >= min_age and ppl.age[i] < max_age:
                        model_method_counts[ppl.method[i]] += 1

        model_method_counts[:] /= model_method_counts[:].sum()

        # From data - Kenya PMA 2022 (mix_kenya.csv)
        data_methods_mix = {
                'Withdrawal': 1.03588605253422,
                'Other traditional': 4.45800961894192,
                'Condoms': 8.41657417684055,
                'Pill': 7.95412504624491,
                'Injectables': 34.0732519422863,
                'Implants': 33.9622641509434,
                'IUDs': 3.08916019237884,
                'BTL': 3.88457269700333,
                'Other modern': 3.12615612282649
        }

        # From data - Kenya PMA 2022 (use_kenya.csv)
        data_methods_use = {
                'No use': 51.1078954508456,
                'Any method': 48.8921045491544
        }

        # Plot bar charts of method mix and use among users

        # Calculate users vs non-users in model
        model_methods_mix = sc.dcp(model_method_counts)
        model_use = [model_methods_mix['None'], model_methods_mix[1:].sum()]
        model_use_percent = [i * 100 for i in model_use]

        # Calculate mix within users in model
        model_methods_mix['None'] = 0.0
        model_users_sum = model_methods_mix[:].sum()
        model_methods_mix[:] /= model_users_sum
        mix_model = model_methods_mix.values()[1:]
        mix_percent_model = [i * 100 for i in mix_model]

        # Set method use and mix from data
        mix_percent_data = list(data_methods_mix.values())
        data_use_percent = list(data_methods_use.values())

        # Set up plotting
        use_labels = list(data_methods_use.keys())
        df_mix = pd.DataFrame({'PMA': mix_percent_data, 'FPsim': mix_percent_model}, index=model_labels_methods)
        df_use = pd.DataFrame({'PMA': data_use_percent, 'FPsim': model_use_percent}, index=use_labels)

        ax = df_mix.plot.barh()
        ax.set_xlabel('Percent users')
        ax.set_title('Contraceptive method mix model vs data')

        pl.savefig("figs/method_mix.png", bbox_inches='tight', dpi=100)

        ax = df_use.plot.barh()
        ax.set_xlabel('Percent')
        ax.set_title('Contraceptive method use model vs data')

        pl.savefig("figs/method_use.png", bbox_inches='tight', dpi=100)

        #pl.show()


if do_plot_skyscrapers:

        age_keys = list(age_bin_map.keys())[1:]
        age_bins = pl.arange(min_age, max_age, bin_size)
        parity_bins = pl.arange(0, 7)
        n_age = len(age_bins)
        n_parity = len(parity_bins)
        x_age = pl.arange(n_age)
        x_parity = pl.arange(n_parity)  # Should be the same

        # Load data
        data_parity_bins = pl.arange(0,7)
        sky_raw_data = skyscrapers
        sky_raw_data = sky_raw_data[sky_raw_data['dataset'] == dataset]

        sky_parity = sky_raw_data['parity'].to_numpy()
        sky_props = sky_raw_data['percentage'].to_numpy()


        sky_arr = sc.odict()

        sky_arr['Data'] = pl.zeros((len(age_keys), len(parity_bins)))

        proportion = 0
        age_name = ''
        for age, row in sky_raw_data.iterrows():
                if row.age in age_keys:
                        age_ind = age_keys.index(row.age)
                        sky_arr['Data'][age_ind, row.parity] = row.percentage


        # Extract from model
        sky_arr['Model'] = pl.zeros((len(age_bins), len(parity_bins)))
        ppl = sim.people
        for i in range(len(ppl)):
                if ppl.alive[i] and not ppl.sex[i] and ppl.age[i] >= min_age and ppl.age[i] < max_age:
                        age_bin = sc.findinds(age_bins <= ppl.age[i])[-1]
                        parity_bin = sc.findinds(parity_bins <= ppl.parity[i])[-1]
                        sky_arr['Model'][age_bin, parity_bin] += 1


        # Normalize
        for key in ['Data', 'Model']:
                sky_arr[key] /= sky_arr[key].sum() / 100

        sky_arr['Diff: data - model'] = sky_arr['Data']-sky_arr['Model']

        # Plot skyscrapers
        for key in ['Data', 'Model', 'Diff: data - model']:
                fig = pl.figure(figsize=(20, 14))

                sc.bar3d(fig=fig, data=sky_arr[key], cmap='jet')
                pl.xlabel('Age', fontweight='bold')
                pl.ylabel('Parity', fontweight='bold')
                pl.title(f'Age-parity plot for the {key.lower()}\n\n', fontweight='bold')
                pl.gca().set_xticks(pl.arange(n_age))
                pl.gca().set_yticks(pl.arange(n_parity))
                pl.gca().set_xticklabels(age_bins)
                pl.gca().set_yticklabels(parity_bins)
                pl.gca().view_init(30, 45)
                pl.draw()

                pl.savefig('figs/skyscrapers_' + str(key.lower()) + '.png')

                pl.show()





sc.toc()
print('Done.')


'''
fig, (ax1, ax2) = pl.subplots(2)

        y_pos1 = pl.arange(len(model_labels_all))
        all_counts_model = model_method_counts.values()

        ax1.barh(y_pos1, all_counts_model, label = 'FPsim')
        ax1.set_yticks(y_pos1, labels=model_labels_all)
        ax1.invert_yaxis()  # labels read top-to-bottom
        ax1.set_xlabel('Percent method users')
        ax1.set_title('Method mix in FPsim Kenya including non-use')

        y_pos2 = pl.arange(len(model_labels_methods))
        model_methods_users = sc.dcp(model_method_counts)
        model_methods_users['None'] = 0.0
        model_methods_users[:] /= model_methods_users[:].sum()
        users_model = model_methods_users.values()[1:]
        users_percent_model = [i*100 for i in users_model]

        users_percent_data = data_methods_users.values()

        ax2.barh(y_pos2, users_percent_model, label = 'FPsim')
        ax2.barh(y_pos2, users_percent_data, label = 'PMA')
        ax2.set_yticks(y_pos2, labels=model_labels_methods)
        ax2.invert_yaxis()  # labels read top-to-bottom
        ax2.set_xlabel('Percent method users')
        ax2.set_title('Method mix in FPsim Kenya')

        pl.show()
        
        
        sky_arr['Data'] = pl.zeros((len(age_bins), len(parity_bins)))
        count = -1
        for age_bin in x_age:
                for pb in parity_bins:
                        count += 1
                        parity_bin = min(n_parity - 1, pb)
                        sky_arr['Data'][age_bin, parity_bin] += sky_props[count]
        assert count == len(sky_props) - 1  # Ensure they're the right length

        # Extract from model
        sky_arr['Model'] = pl.zeros((len(age_bins), len(parity_bins)))
        for i in range(len(ppl)):
                if ppl.alive[i] and not ppl.sex[i] and ppl.age[i] >= min_age and ppl.age[i] < max_age:
                        age_bin = sc.findinds(age_bins <= ppl.age[i])[-1]
                        parity_bin = sc.findinds(parity_bins <= ppl.parity[i])[-1]
                        sky_arr['Model'][age_bin, parity_bin] += 1

        # Normalize
        for key in ['Data', 'Model']:
                sky_arr[key] /= sky_arr[key].sum() / 100


        fig, axs = pl.subplots(3)

        fig.suptitle('Age Parity Distribution')

        axs[0].pcolormesh(age_bins, parity_bins, sky_arr.Data.transpose(), shading='nearest', cmap='turbo')
        axs[0].set_aspect(1. / ax.get_data_ratio())  # Make square
        axs[0].set_title('Age-parity plot: Kenya PMA 2022')
        axs[0].set_xlabel('Age')
        ax[0].set_ylabel('Parity')

        axs[1].pcolormesh(age_bins, parity_bins, sky_arr.Model.transpose(), shading='nearest', cmap='turbo')
        axs[1].set_aspect(1. / ax.get_data_ratio())  # Make square
        axs[1].set_title('Age-parity plot: Kenya PMA 2022')
        axs[1].set_xlabel('Age')
        axs[1].set_ylabel('Parity')

        pl.show()


'''