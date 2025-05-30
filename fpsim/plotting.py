"""
Standard plots used in calibrations
"""
import os
import fpsim as fp
import sciris as sc
import pylab as pl
import seaborn as sns
import pandas as pd
import numpy as np
from . import defaults as fpd

# Default Settings
min_age = 15
max_age = 50
bin_size = 5
age_bin_map = fpd.age_bin_map

class Config:
    """Configuration for plots"""
    do_save = True
    _figs_directory = 'figures'

    # Default validation data mapping (key: filename)
    default_val_data_mapping = {
        'ageparity': 'ageparity.csv',
        'use': 'use.csv',
        'spacing': 'birth_spacing_dhs.csv',
        'afb': 'afb.table.csv',
        'mcpr': 'cpr.csv',
        'asfr': 'asfr.csv',
        'methods': 'mix.csv',
        'tfr': 'tfr.csv',
        'popsize': 'popsize.csv'
    }

    @classmethod
    def set_figs_directory(cls, new_directory):
        """Set the figures directory and ensure it exists."""
        cls._figs_directory = new_directory
        sc.makefilepath(folder=cls._figs_directory, makedirs=True)

    @classmethod
    def get_figs_directory(cls):
        """Get the current figures directory."""
        return cls._figs_directory

    @classmethod
    def load_validation_data(cls, country, val_data_mapping=None):
        """
        Load validation data for the specified country.

        Args:
            country (str): The name of the country folder.
            val_data_mapping (dict, optional): Custom mapping of validation data keys to filenames.
                                               Defaults to `default_val_data_mapping`.

        Returns:
            sc.objdict: Loaded validation data as a dictionary-like object.
        """
        if val_data_mapping is None:
            val_data_mapping = cls.default_val_data_mapping

        base_path = getattr(fp.locations, country).filenames()['base']
        val_data = sc.objdict()

        for key, filename in val_data_mapping.items():
            filepath = os.path.join(base_path, filename)
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Validation data file not found: {filepath}")
            val_data[key] = pd.read_csv(filepath)

        return val_data

# Ensure the default directory exists
Config.set_figs_directory(Config._figs_directory)

def save_figure(filename):
    """Helper function to save a figure if saving is enabled."""
    if Config.do_save:
        sc.savefig(f"{Config.get_figs_directory()}/{filename}")


def pop_growth_rate(years, population):
    """Calculates growth rate as a time series to help compare model to data"""
    growth_rate = np.zeros(len(years) - 1)

    for i in range(len(years)):
        if population[i] == population[-1]:
            break
        growth_rate[i] = ((population[i + 1] - population[i]) / population[i]) * 100

    return growth_rate


def plot_by_age(sim):
    """Plot CPR by age and method mix by age."""
    # CPR by age
    fig, ax = pl.subplots()
    age_bins = [18, 20, 25, 35, 50]
    colors = sc.vectocolor(age_bins)
    for cind, (alabel, ares) in enumerate(sim.get_analyzer('cpr_by_age').results.items()):
        ax.plot(sim.results.t, ares, label=alabel, color=colors[cind])
    ax.legend(loc='best', frameon=False)
    ax.set_ylim([0, 1])
    ax.set_ylabel('CPR')
    ax.set_title('CPR')
    pl.show()
    save_figure('cpr_by_age.png')

    # Method mix by age
    fig, ax = pl.subplots()
    df = pd.DataFrame(sim.get_analyzer('method_mix_by_age').results)
    df['method'] = sim.contraception_module.methods.keys()
    df_plot = df.melt(id_vars='method')
    sns.barplot(x='method', y='value', hue='variable', data=df_plot, ax=ax, palette="viridis")
    ax.set_title('Method Mix by Age')
    pl.show()
    save_figure('method_mix_by_age.png')


def plot_asfr(sim, data):
    # Print ASFR form model in output
    for key in age_bin_map:
        print(f'ASFR (annual) for age bin {key} in the last year of the sim: {sim.results["asfr"][key][-1]}')

    x = [1, 2, 3, 4, 5, 6, 7, 8]

    # Load data
    year = data[data['year'] == sim.pars['end_year']]
    asfr_data = year.drop(['year'], axis=1).values.tolist()[0]

    x_labels = []
    asfr_model = []

    # Extract from model
    for key in age_bin_map:
        x_labels.append(key)
        asfr_model.append(sim.results['asfr'][key][-1])

    # Calculate RMSE
    rmse = np.sqrt(np.mean((np.array(asfr_data) - np.array(asfr_model)) ** 2))
    print(f'ASFR RMSE between model and data: {rmse}')

    # Plot
    fig, ax = pl.subplots()
    kw = dict(lw=3, alpha=0.7, markersize=10)
    ax.plot(x, asfr_data, marker='^', color='black', label="UN data", **kw)
    ax.plot(x, asfr_model, marker='*', color='cornflowerblue', label="FPsim", **kw)
    pl.xticks(x, x_labels)
    pl.ylim(bottom=-10)
    ax.set_title(f'Age specific fertility rate per 1000 woman years\n(RMSE: {rmse:.2f})')
    ax.set_xlabel('Age')
    ax.set_ylabel('ASFR')
    ax.legend(frameon=False)
    sc.boxoff()

    pl.show()
    save_figure('asfr.png')


def plot_methods(sim, data_methods, data_use):
    """
    Plots both dichotomous method data_use and non-data_use and contraceptive mix
    """
    ppl = sim.people

    # Setup
    model_labels_all = [m.label for m in sim.contraception_module.methods.values()]
    model_labels_methods = sc.dcp(model_labels_all)
    model_method_counts = sc.odict().make(keys=model_labels_all, vals=0.0)

    # Extract from model
    # TODO: refactor, this shouldn't need to loop over people, can just data_use a histogram
    for i in range(len(ppl)):
        if ppl.alive[i] and not ppl.sex[i] and ppl.age[i] >= min_age and ppl.age[i] < max_age:
            model_method_counts[ppl.method[i]] += 1

    model_method_counts[:] /= model_method_counts[:].sum()

    # Method mix from data - country PMA data (mix.csv)
    data_methods_mix = {
        'Pill': data_methods.loc[data_methods['method'] == 'Pill', 'perc'].iloc[0],
        'IUDs': data_methods.loc[data_methods['method'] == 'IUDs', 'perc'].iloc[0],
        'Injectables': data_methods.loc[data_methods['method'] == 'Injectables', 'perc'].iloc[0],
        'Condoms': data_methods.loc[data_methods['method'] == 'Condoms', 'perc'].iloc[0],
        'BTL': data_methods.loc[data_methods['method'] == 'BTL', 'perc'].iloc[0],
        'Withdrawal': data_methods.loc[data_methods['method'] == 'Withdrawal', 'perc'].iloc[0],
        'Implants': data_methods.loc[data_methods['method'] == 'Implants', 'perc'].iloc[0],
        'Other traditional': data_methods.loc[data_methods['method'] == 'Other traditional', 'perc'].iloc[0],
        'Other modern': data_methods.loc[data_methods['method'] == 'Other modern', 'perc'].iloc[0]
    }

    # Method data_use from data - country PMA data (data_use.csv)
    no_use = data_use.iloc[0]['perc']
    any_method = data_use.iloc[1]['perc']
    data_methods_use = {
        'No data_use': no_use,
        'Any method': any_method
    }

    # Plot bar charts of method mix and data_use among users

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

    # Set method data_use and mix from data
    mix_percent_data = list(data_methods_mix.values())
    data_use_percent = list(data_methods_use.values())

    # Calculate RMSE for method mix
    rmse_mix = np.sqrt(np.mean((np.array(mix_percent_data) - np.array(mix_percent_model)) ** 2))
    print(f'RMSE for method mix: {rmse_mix:.2f}')

    # Calculate RMSE for overall use
    rmse_use = np.sqrt(np.mean((np.array(data_use_percent) - np.array(model_use_percent)) ** 2))
    print(f'RMSE for overall use: {rmse_use:.2f}')

    # Set up plotting
    use_labels = list(data_methods_use.keys())
    df_mix = pd.DataFrame({'PMA': mix_percent_data, 'FPsim': mix_percent_model}, index=model_labels_methods[1:])
    df_use = pd.DataFrame({'PMA': data_use_percent, 'FPsim': model_use_percent}, index=use_labels)

    # Plot mix
    ax = df_mix.plot.barh(color={'PMA': 'black', 'FPsim': 'cornflowerblue'})
    ax.set_xlabel('Percent users')
    ax.set_title(f'Contraceptive Method Mix - Model vs Data\n(RMSE: {rmse_mix:.2f})')

    pl.tight_layout()

    pl.show()
    save_figure('method_mix.png')

    # Plot data_use
    ax = df_use.plot.barh(color={'PMA': 'black', 'FPsim': 'cornflowerblue'})
    ax.set_xlabel('Percent')
    ax.set_title(f'Contraceptive Method Use - Model vs Data\n(RMSE: {rmse_use:.2f})')

    pl.show()
    save_figure('method_use.png')


def plot_ageparity(sim, ageparity_data, ageparity_dataset='PMA 2022'):
    """
    Plot an age-parity distribution for model vs data
    """
    # Set up
    ppl = sim.people
    age_keys = list(age_bin_map.keys())
    age_bins = pl.arange(min_age, max_age, bin_size)
    parity_bins = pl.arange(0, 7)  # Plot up to parity 6
    n_age = len(age_bins)
    n_parity = len(parity_bins)

    # Load data
    ageparity_data = ageparity_data[ageparity_data['dataset'] == ageparity_dataset]
    sky_arr = sc.odict()

    sky_arr['Data'] = pl.zeros((len(age_keys), len(parity_bins)))

    for age, row in ageparity_data.iterrows():
        if row.age in age_keys and row.parity < 7:
            age_ind = age_keys.index(row.age)
            sky_arr['Data'][age_ind, row.parity] = row.percentage

    # Extract from model
    # TODO, refactor - can just use histogram instead of looping over agents
    sky_arr['Model'] = pl.zeros((len(age_keys), len(parity_bins)))
    for i in range(len(ppl)):
        if ppl.alive[i] and not ppl.sex[i] and ppl.age[i] >= min_age and ppl.age[i] < max_age:
            # Match age to age_keys
            for age_key, age_range in age_bin_map.items():
                if age_range[0] <= ppl.age[i] < age_range[1]:
                    age_bin = age_keys.index(age_key)
                    break
            else:
                continue  # Skip if no match is found

            # Find parity bin
            if ppl.parity[i] < len(parity_bins):
                parity_bin = int(ppl.parity[i])  # Ensure parity is an integer index
                sky_arr['Model'][age_bin, parity_bin] += 1

    # Normalize
    for key in ['Data', 'Model']:
        sky_arr[key] /= sky_arr[key].sum() / 100

    # Find diff to help visualize in plotting
    sky_arr['Diff_data-model'] = sky_arr['Data'] - sky_arr['Model']

    # Plot ageparity
    for key in ['Data', 'Model', 'Diff_data-model']:
        fig = pl.figure(figsize=(20, 14))

        pl.pcolormesh(sky_arr[key], cmap='parula')
        pl.xlabel('Age', fontweight='bold')
        pl.ylabel('Parity', fontweight='bold')
        pl.title(f'Age-parity plot for the {key.lower()}\n\n', fontweight='bold')
        pl.gca().set_xticks(pl.arange(n_age))
        pl.gca().set_yticks(pl.arange(n_parity))
        pl.gca().set_xticklabels(age_bins)
        pl.gca().set_yticklabels(parity_bins)
        # pl.gca().view_init(30, 45)
        pl.draw()

        save_figure(f'ageparity_{key.lower()}.png')
        pl.show()


def plot_cpr(sim, data_cpr):
    '''
    Plot contraceptive prevalence rate for model vs data
    '''
    # Import data
    data_cpr = data_cpr[data_cpr['year'] <= sim.pars['end_year']]  # Restrict years to plot
    res = sim.results

    # Align data for RMSE calculation
    years = data_cpr['year']
    data_values = data_cpr['cpr'].values
    model_values = np.interp(years, res['t'], res['cpr'] * 100)  # Interpolate model CPR to match data years

    # Calculate RMSE
    rmse = np.sqrt(np.mean((data_values - model_values) ** 2))
    print(f'RMSE for CPR: {rmse:.2f}')

    # Plot
    pl.plot(data_cpr['year'], data_cpr['cpr'], label='UN Data Portal', color='black')
    pl.plot(res['t'], res['cpr'] * 100, label='FPsim', color='cornflowerblue')
    pl.xlabel('Year')
    pl.ylabel('Percent')
    pl.title(f'Contraceptive Prevalence Rate - Model vs Data\n(RMSE: {rmse:.2f})')
    pl.legend()

    save_figure('cpr.png')
    pl.show()


def plot_tfr(sim, data_tfr):
    """
    Plot total fertility rate for model vs data
    """

    res = sim.results
    # Align model and data years for RMSE calculation
    data_years = data_tfr['year']
    data_tfr_values = data_tfr['tfr']
    model_tfr_values = np.interp(data_years, res['tfr_years'],
                                 res['tfr_rates'])  # Interpolate model to match data years

    # Calculate RMSE
    rmse = np.sqrt(np.mean((np.array(data_tfr_values) - np.array(model_tfr_values)) ** 2))
    print(f'RMSE for Total Fertility Rate: {rmse:.2f}')

    # Plot
    pl.plot(data_tfr['year'], data_tfr['tfr'], label='World Bank', color='black')
    pl.plot(res['tfr_years'], res['tfr_rates'], label='FPsim', color='cornflowerblue')
    pl.xlabel('Year')
    pl.ylabel('Rate')
    pl.title(f'Total Fertility Rate - Model vs Data\n(RMSE: {rmse:.2f})')
    pl.legend()

    save_figure('tfr.png')
    pl.show()


def plot_pop_growth(sim, data_popsize):
    """
    Plot annual population growth rate for model vs data
    """

    res = sim.results

    # Import data
    data_popsize = data_popsize[data_popsize['year'] <= sim.pars['end_year']]  # Restrict years to plot
    data_pop_years = data_popsize['year'].to_numpy()
    data_population = data_popsize['population'].to_numpy()

    # Extract from model
    model_growth_rate = pop_growth_rate(res['tfr_years'], res['pop_size'])

    data_growth_rate = pop_growth_rate(data_pop_years, data_population)

    # Plot
    pl.plot(data_pop_years[1:], data_growth_rate, label='World Bank', color='black')
    pl.plot(res['tfr_years'][1:], model_growth_rate, label='FPsim', color='cornflowerblue')
    pl.xlabel('Year')
    pl.ylabel('Rate')
    pl.title(f'Population Growth Rate - Model vs Data')
    pl.legend()

    save_figure('popgrowth.png')
    pl.show()


def plot_birth_space_afb(sim, data_spaces, data_afb):
    """
    Plot birth space and age at first birth for model vs data
    """
    # Set up
    ppl = sim.people
    spacing_bins = sc.odict({'0-12': 0, '12-24': 1, '24-48': 2, '>48': 4})  # Spacing bins in months
    model_age_first = []
    model_spacing = []
    model_spacing_counts = sc.odict().make(keys=spacing_bins.keys(), vals=0.0)
    data_spacing_counts = sc.odict().make(keys=spacing_bins.keys(), vals=0.0)

    # Extract age at first birth and birth spaces from model
    # TODO, refactor to avoid loops
    for i in range(len(ppl)):
        if ppl.alive[i] and not ppl.sex[i] and min_age <= ppl.age[i] < max_age:
            if ppl.first_birth_age[i] == -1:
                model_age_first.append(float('inf'))
            else:
                model_age_first.append(ppl.first_birth_age[i])
                if ppl.parity[i] > 1:
                    cleaned_birth_ages = ppl.birth_ages[i][~np.isnan(ppl.birth_ages[i])]
                    for d in range(len(cleaned_birth_ages) - 1):
                        space = cleaned_birth_ages[d + 1] - cleaned_birth_ages[d]
                        if space > 0:
                            ind = sc.findinds(space > spacing_bins[:])[-1]
                            model_spacing_counts[ind] += 1
                            model_spacing.append(space)

    # Normalize model birth space bin counts to percentages
    model_spacing_counts[:] /= model_spacing_counts[:].sum()
    model_spacing_counts[:] *= 100

    age_first_birth_model = pd.DataFrame(data=model_age_first)

    # Extract birth spaces and age at first birth from data
    for i, j in data_spaces.iterrows():
        space = j['space_mo'] / 12
        ind = sc.findinds(space > spacing_bins[:])[-1]
        data_spacing_counts[ind] += j['Freq']

    age_first_birth_data = pd.DataFrame(data=data_afb)

    # Normalize dat birth space bin counts to percentages
    data_spacing_counts[:] /= data_spacing_counts[:].sum()
    data_spacing_counts[:] *= 100

    # RMSE for Birth Spacing Bins
    data_spacing_bins = np.array(data_spacing_counts.values())
    model_spacing_bins = np.array(model_spacing_counts.values())
    rmse_spacing = np.sqrt(np.mean((data_spacing_bins - model_spacing_bins) ** 2))
    print(f'RMSE for Birth Spacing Bins: {rmse_spacing:.2f}')

    # RMSE for Age at First Birth
    model_afb_values = age_first_birth_model[age_first_birth_model[0] != float('inf')][0].values
    data_afb_values = age_first_birth_data['afb'].values

    # Compute mean for model and data to resolve shape mismatch
    model_afb_mean = np.mean(model_afb_values)
    data_afb_mean = np.mean(data_afb_values)

    # RMSE between aggregated means
    rmse_afb = np.sqrt((model_afb_mean - data_afb_mean) ** 2)
    print(f'Mean Age at First Birth - Model: {model_afb_mean:.2f}, Data: {data_afb_mean:.2f}')
    print(f'RMSE for Age at First Birth: {rmse_afb:.2f}')

    # Plot age at first birth (histogram with KDE)
    sns.histplot(data=age_first_birth_model, stat='proportion', kde=True, binwidth=1, color='cornflowerblue',
                 label='FPsim')
    sns.histplot(x=age_first_birth_data['afb'], stat='proportion', kde=True, weights=age_first_birth_data['wt'],
                 binwidth=1, color='dimgrey', label='DHS data')
    pl.xlabel('Age at first birth')
    pl.title(f'Age at First Birth - Model vs Data\n(RMSE: {rmse_afb:.2f})')
    pl.legend()

    save_figure('age_first_birth.png')
    pl.show()

    # Plot birth space bins with diff
    data_spacing_bins = np.array(data_spacing_counts.values())
    model_spacing_bins = np.array(model_spacing_counts.values())

    diff = model_spacing_bins - data_spacing_bins

    res_bins = np.array([[model_spacing_bins], [data_spacing_bins], [diff]])

    bins_frame = pd.DataFrame(
        {'Model': model_spacing_bins, 'Data': data_spacing_bins, 'Diff': diff},
        index=spacing_bins.keys())

    print(bins_frame)  # Print in output, remove if not needed

    ax = bins_frame.plot.barh(color={'Data': 'black', 'Model': 'cornflowerblue', 'Diff': 'red'})
    ax.set_xlabel('Percent of live birth spaces')
    ax.set_ylabel('Birth space in months')
    ax.set_title(f'Birth Space Bins - Model vs Data\n(RMSE: {rmse_spacing:.2f})')

    save_figure('birth_space_bins.png')
    pl.show()


def plot_paid_work(sim, data_employment):
    """
    Plot rates of paid employment between model and data
    """
    # Extract paid work from data
    data_empowerment = data_employment.iloc[1:-1]
    data_paid_work = data_empowerment[['age', 'paid_employment', 'paid_employment.se']].copy()
    age_bins = np.arange(min_age, max_age + 1, bin_size)
    data_paid_work['age_group'] = pd.cut(data_paid_work['age'], bins=age_bins, right=False)

    # Calculate mean and standard error for each age bin
    employment_data_grouped = data_paid_work.groupby('age_group', observed=False)['paid_employment']
    employment_data_mean = employment_data_grouped.mean().tolist()
    employment_data_se = data_paid_work.groupby('age_group', observed=False)['paid_employment.se'].apply(
        lambda x: np.sqrt(np.sum(x ** 2)) / len(x)).tolist()

    # Extract paid work from model
    employed_counts = {age_bin: 0 for age_bin in age_bins}
    total_counts = {age_bin: 0 for age_bin in age_bins}

    # Count the number of employed and total people in each age bin
    ppl = sim.people
    for i in range(len(ppl)):
        if ppl.alive[i] and not ppl.sex[i] and min_age <= ppl.age[i] < max_age:
            age_bin = age_bins[sc.findinds(age_bins <= ppl.age[i])[-1]]
            total_counts[age_bin] += 1
            if ppl.paid_employment[i]:
                employed_counts[age_bin] += 1

    # Calculate the percentage of employed people in each age bin and their standard errors
    percentage_employed = {}
    percentage_employed_se = {}
    age_bins = np.arange(min_age, max_age, bin_size)
    for age_bin in age_bins:
        total_ppl = total_counts[age_bin]
        if total_ppl != 0:
            employed_ratio = employed_counts[age_bin] / total_ppl
            percentage_employed[age_bin] = employed_ratio
            percentage_employed_se[age_bin] = (employed_ratio * (
                    1 - employed_ratio) / total_ppl) ** 0.5
        else:
            percentage_employed[age_bin] = 0
            percentage_employed_se[age_bin] = 0

    employment_model = list(percentage_employed.values())
    employment_model_se = list(percentage_employed_se.values())

    # Calculate RMSE
    rmse = np.sqrt(np.mean((np.array(employment_data_mean) - np.array(employment_model)) ** 2))
    print(f'RMSE for Paid Employment: {rmse:.2f}')

    # Set up plotting
    labels = age_bin_map
    x_pos = np.arange(len(labels))
    fig, ax = pl.subplots()
    width = 0.35

    # Plot Data
    ax.barh(x_pos - width / 2, employment_data_mean, width, label='DHS', color='black')
    ax.errorbar(employment_data_mean, x_pos - width / 2, xerr=employment_data_se, fmt='none', ecolor='gray',
                capsize=5)

    # Plot Model
    ax.barh(x_pos + width / 2, employment_model, width, label='FPsim', color='cornflowerblue')
    ax.errorbar(employment_model, x_pos + width / 2, xerr=employment_model_se, fmt='none', ecolor='gray',
                capsize=5)

    # Set labels and title
    ax.set_xlabel('Percent Women with Paid Work')
    ax.set_ylabel('Age Bin')
    ax.set_title(f'Kenya: Paid Employment - Model vs Data\n(RMSE: {rmse:.2f})')
    ax.set_yticks(x_pos)
    ax.set_yticklabels(labels)
    ax.legend()

    save_figure('paid_employment.png')
    pl.show()


def plot_education(sim, data_education):
    """
    Plot years of educational attainment between model and data
    """
    pl.clf()

    # Extract education from data
    data_edu = data_education[['age', 'edu', 'se']].sort_values(by='age')
    data_edu = data_edu.query(f"{min_age} <= age < {max_age}").copy()
    age_bins = np.arange(min_age, max_age + 1, bin_size)
    data_edu['age_group'] = pd.cut(data_edu['age'], bins=age_bins, right=False)

    # Calculate mean and standard error for each age bin
    education_data_grouped = data_edu.groupby('age_group', observed=False)['edu']
    education_data_mean = education_data_grouped.mean().tolist()
    education_data_se = data_edu.groupby('age_group', observed=False)['se'].apply(
        lambda x: np.sqrt(np.sum(x ** 2)) / len(x)).tolist()

    # Extract education from model
    model_edu_years = {age_bin: [] for age_bin in np.arange(min_age, max_age, bin_size)}
    ppl = sim.people
    for i in range(len(ppl)):
        if ppl.alive[i] and not ppl.sex[i] and min_age <= ppl.age[i] < max_age:
            age_bin = age_bins[sc.findinds(age_bins <= ppl.age[i])[-1]]
            model_edu_years[age_bin].append(ppl.edu_attainment[i])

    # Calculate average # of years of educational attainment for each age
    model_edu_mean = []
    model_edu_se = []
    for age_group in model_edu_years:
        if len(model_edu_years[age_group]) != 0:
            avg_edu = sum(model_edu_years[age_group]) / len(model_edu_years[age_group])
            se_edu = np.std(model_edu_years[age_group], ddof=1) / np.sqrt(len(model_edu_years[age_group]))
            model_edu_mean.append(avg_edu)
            model_edu_se.append(se_edu)
        else:
            model_edu_years[age_group] = 0
            model_edu_se.append(0)

    # Calculate RMSE
    rmse = np.sqrt(np.mean((np.array(education_data_mean) - np.array(model_edu_mean)) ** 2))
    print(f'RMSE for Education Attainment: {rmse:.2f}')

    # Set up plotting
    labels = age_bin_map
    x_pos = np.arange(len(labels))
    fig, ax = pl.subplots()
    width = 0.35

    # Plot DHS data
    ax.barh(x_pos - width / 2, education_data_mean, width, label='DHS', color='black')
    ax.errorbar(education_data_mean, x_pos - width / 2, xerr=education_data_se, fmt='none', ecolor='gray',
                capsize=5)

    # Plot FPsim data
    ax.barh(x_pos + width / 2, model_edu_mean, width, label='FPsim', color='cornflowerblue')
    ax.errorbar(model_edu_mean, x_pos + width / 2, xerr=model_edu_se, fmt='none', ecolor='gray', capsize=5)

    # Set labels and title
    ax.set_xlabel('Avg Years of Education Attainment')
    ax.set_ylabel('Age Bin')
    ax.set_title(f'Kenya: Years of Education - Model vs Data\n(RMSE: {rmse:.2f})')
    ax.set_yticks(x_pos)
    ax.set_yticklabels(labels)
    ax.legend()

    save_figure('education.png')
    pl.show()


def plot_all(sim, val_data):
    """Plots all the figures above besides empowerment plots"""
    plot_by_age(sim)
    plot_asfr(sim, val_data['asfr'])
    plot_methods(sim, val_data['methods'], val_data['use'])
    plot_ageparity(sim, val_data['ageparity'])
    plot_cpr(sim, val_data['mcpr'])
    plot_tfr(sim, val_data['tfr'])
    plot_pop_growth(sim, val_data['popsize'])
    plot_birth_space_afb(sim, val_data['spacing'], val_data['afb'])
    return

def plot_calib(sim, val_data):
    """Plots all the commonly used plots for calibration"""
    plot_methods(sim, val_data['methods'], val_data['use'])
    plot_cpr(sim, val_data['mcpr'])
    plot_tfr(sim, val_data['tfr'])
    plot_birth_space_afb(val_data['spacing'], val_data['afb'])
    plot_asfr(sim, val_data['asfr'])
    return

