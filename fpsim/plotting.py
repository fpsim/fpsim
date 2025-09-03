"""
Standard plots used in calibrations and analyses
"""
import os
import fpsim as fp
import sciris as sc
import pylab as pl
import seaborn as sns
import pandas as pd
import numpy as np
from . import defaults as fpd
from pathlib import Path

# Default Settings
min_age = 15
max_age = 50
bin_size = 5

age_bin_map = fpd.age_bin_map
rmse_scores = {}


class Config:
    """Configuration for plots"""
    do_save = True
    show_rmse = True
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
        'popsize': 'popsize.csv',
        'education': 'education.csv'
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
    def load_validation_data(cls, location, val_data_mapping=None, keys=None):
        """
        Load validation data for the specified country or region.
        Falls back to country-level data if region-specific file is not found.

        Args:
            location (str): The name of the location folder (region or country).
            val_data_mapping (dict, optional): Custom mapping of validation data keys to filenames.
                                               Defaults to `default_val_data_mapping`.
            keys (str or list of str, optional): Specific metric(s) to load (e.g., 'asfr' or ['asfr', 'tfr']).

        Returns:
            sc.objdict: Loaded validation data (either full or partial dict with requested keys).
        """
        if val_data_mapping is None:
            val_data_mapping = cls.default_val_data_mapping

        # Handle single string input for keys
        if isinstance(keys, str):
            keys = [keys]

        # Filter mapping if specific keys requested
        if keys is not None:
            val_data_mapping = {k: v for k, v in val_data_mapping.items() if k in keys}

        loc_mod = getattr(fp.locations, location)
        file_paths = loc_mod.filenames()
        val_data = sc.objdict()

        for key, filename in val_data_mapping.items():
            file_path = file_paths.get(key, None)
            if file_path is None:
                raise ValueError(f"No path defined for key '{key}' in filenames() for location '{location}'.")

            if not Path(file_path).exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            val_data[key] = pd.read_csv(file_path)

            # Filter to region if 'region' column exists
            if 'region' in val_data[key].columns:
                val_data[key] = val_data[key][val_data[key]['region'] == location].copy()
                val_data[key].drop(columns=['region'], inplace=True)

        return val_data

def save_figure(filename):
    """Helper function to save a figure if saving is enabled."""
    if Config.do_save:
        sc.savefig(f"{Config.get_figs_directory()}/{filename}")

def compute_rmse(model_vals, data_vals):
    """
    Compute mean-normalized Root Mean Squared Error (RMSE) between model output and real data.

    Normalization is done by dividing RMSE by the mean of the data values.

    :param model_vals: List or numpy array of model-generated values
    :param data_vals: List or numpy array of real-world data values
    :return: Normalized RMSE (unitless)
    """
    model_vals = np.array(model_vals)
    data_vals = np.array(data_vals)

    rmse = np.sqrt(np.mean((model_vals - data_vals) ** 2))
    mean_val = np.mean(data_vals)

    if mean_val != 0:
        rmse /= mean_val  # Mean normalization

    return rmse

def pop_growth_rate(years, population):
    """Calculates growth rate as a time series to help compare model to data"""
    growth_rate = np.zeros(len(years) - 1)

    for i in range(len(years)):
        if population[i] == population[-1]:
            break
        growth_rate[i] = ((population[i + 1] - population[i]) / population[i]) * 100

    return growth_rate


def plot_cpr_by_age(sim):
    """
    Plot CPR by age.
    Note: This plot can only be run if the sim used the analyzer 'cpr_by_age'
    """
    # CPR by age
    fig, ax = pl.subplots()
    age_bins = ['<18', '18-20', '20-25', '25-35', '>35']
    for age_key in age_bins:
        ares = sim.analyzers['cpr_by_age'].results[age_key]
        ax.plot(sim.results['timevec'], ares, label=age_key)
    ax.legend(loc='best', frameon=False)
    ax.set_ylim([0, 1])
    ax.set_ylabel('CPR')
    ax.set_title('CPR')
    save_figure('cpr_by_age.png')
    pl.show()


def plot_asfr(sim):
    """Plots age-specific fertility rate"""

    data = Config.load_validation_data(sim.pars['location'], keys='asfr')['asfr']

    x = [1, 2, 3, 4, 5, 6, 7, 8]

    # Extract ASFR from simulation results
    year = data[data['year'] == sim.pars['stop']]   # Compare model to data at the end year of the sim
    asfr_data = year.drop(['year'], axis=1).values.tolist()[0]

    # Extract ASFR from simulation results
    x_labels = []
    asfr_model = sim.connectors.fp.asfr[2:-1, -1]
    # Compute mean-normalized RMSE
    rmse_scores['asfr'] = compute_rmse(asfr_model, asfr_data)

    # Plot
    fig, ax = pl.subplots()
    kw = dict(lw=3, alpha=0.7, markersize=10)
    ax.plot(x, asfr_data, marker='^', color='black', label="UN data", **kw)
    ax.plot(x, asfr_model, marker='*', color='cornflowerblue', label="FPsim", **kw)
    pl.xticks(x, x_labels)
    pl.ylim(bottom=-10)
    if Config.show_rmse is True:
        ax.set_title(f"Age specific fertility rate per 1000 woman years\n(RMSE: {rmse_scores['asfr']:.2f})")
    else:
        ax.set_title(f'Age specific fertility rate per 1000 woman years')
    ax.set_xlabel('Age')
    ax.set_ylabel('ASFR')
    ax.legend(frameon=False)
    sc.boxoff()

    save_figure('asfr.png')
    pl.show()


def plot_methods(sim):
    """
    Plots both dichotomous method data_use and non-data_use and contraceptive mix
    """
    # Load data
    data = Config.load_validation_data(sim.pars['location'], keys=['methods', 'use'])
    data_methods = data['methods']
    data_use = data['use']

    # Setup
    ppl = sim.people
    cm = sim.connectors.contraception
    model_labels_all = [m.label for m in cm.methods.values()]
    model_labels_methods = sc.dcp(model_labels_all)
    mm = ppl.fp.method[ppl.female & (ppl.age >= min_age) & (ppl.age < max_age)]
    mm_counts, _ = np.histogram(mm, bins=len(model_labels_all))
    mm_counts = mm_counts/mm_counts.sum()
    model_method_counts = sc.odict(zip(model_labels_all, mm_counts))

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
    no_use = data_use.loc[data_use['use'] == 0, 'perc'].values[0]
    any_method = data_use.loc[data_use['use'] == 1, 'perc'].values[0]
    data_methods_use = {
        'No method use': no_use,
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

    # Compute mean-normalized RMSE
    rmse_scores['method_mix'] = compute_rmse(mix_percent_model, mix_percent_data)
    rmse_scores['use'] = compute_rmse(model_use_percent, data_use_percent)

    # Set up plotting
    use_labels = list(data_methods_use.keys())
    df_mix = pd.DataFrame({'PMA': mix_percent_data, 'FPsim': mix_percent_model}, index=model_labels_methods[1:])
    df_mix = df_mix.iloc[::-1]
    df_use = pd.DataFrame({'PMA': data_use_percent, 'FPsim': model_use_percent}, index=use_labels)

    # Plot mix
    ax = df_mix.plot.barh(color={'PMA': 'black', 'FPsim': 'cornflowerblue'})
    ax.set_xlabel('Percent users')
    if Config.show_rmse is True:
        ax.set_title(f"Contraceptive Method Mix - Model vs Data\n(RMSE: {rmse_scores['method_mix']:.2f})")
    else:
        ax.set_title(f'Contraceptive Method Mix - Model vs Data')

    pl.tight_layout()
    save_figure('method_mix.png')
    pl.show()

    # Plot data_use
    ax = df_use.plot.barh(color={'PMA': 'black', 'FPsim': 'cornflowerblue'})
    ax.set_xlabel('Percent')
    if Config.show_rmse is True:
        ax.set_title(f"Contraceptive Method Use - Model vs Data\n(RMSE: {rmse_scores['use']:.2f})")
    else:
        ax.set_title(f'Contraceptive Method Use - Model vs Data')

    pl.tight_layout()
    save_figure('method_use.png')
    pl.show()


def plot_ageparity(sim):
    """
    Plot an age-parity distribution for model vs data
    """
    # Load data
    ageparity_data = Config.load_validation_data(sim.pars['location'], keys=['ageparity'])['ageparity']

    # Set up
    ppl = sim.people
    age_keys = list(age_bin_map.keys())
    age_bins = pl.arange(min_age, max_age, bin_size)
    parity_bins = pl.arange(0, 7)  # Plot up to parity 6
    n_age = len(age_bins)
    n_parity = len(parity_bins)

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
        if ppl.alive.values[i] and ppl.female.values[i] and ppl.age.values[i] >= min_age and ppl.age.values[i] < max_age:
            # Match age to age_keys
            for age_key, age_range in age_bin_map.items():
                if age_range[0] <= ppl.age.values[i] < age_range[1]:
                    age_bin = age_keys.index(age_key)
                    break
            else:
                continue  # Skip if no match is found

            # Find parity bin
            if ppl.parity.values[i] < len(parity_bins):
                parity_bin = int(ppl.parity.values[i])  # Ensure parity is an integer index
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


def plot_cpr(sim):
    '''
    Plot contraceptive prevalence rate for model vs data
    '''
    # Import data
    data_cpr = Config.load_validation_data(sim.pars['location'], keys=['mcpr'])['mcpr']
    data_cpr = data_cpr[data_cpr['year'] <= sim.pars['stop']]  # Restrict years to plot
    res = sim.results

    # Align data for RMSE calculation
    years = data_cpr['year']
    data_values = data_cpr['cpr'].values
    model_values = np.interp(years, res['timevec'], res.contraception.cpr * 100)  # Interpolate model CPR to match data years

    # Compute mean-normalized RMSE
    rmse_scores['cpr'] = compute_rmse(model_values, data_values)

    # Plot
    pl.plot(data_cpr['year'], data_cpr['cpr'], label='UN Data Portal', color='black')
    pl.plot(res['timevec'], res.contraception.cpr * 100, label='FPsim', color='cornflowerblue')
    pl.xlabel('Year')
    pl.ylabel('Percent')
    if Config.show_rmse is True:
        pl.title(f"Contraceptive Prevalence Rate - Model vs Data\n(RMSE: {rmse_scores['cpr']:.2f})")
    else:
        pl.title(f'Contraceptive Prevalence Rate - Model vs Data')
    pl.legend()

    save_figure('cpr.png')
    pl.show()


def plot_tfr(sim):
    """
    Plot total fertility rate for model vs data
    """
    # Load data
    res = sim.results
    df = res.fp.to_df(resample='year', use_years=True)
    data_tfr = Config.load_validation_data(sim.pars['location'], keys=['tfr'])['tfr']

    # Align model and data years for RMSE calculation
    data_years = data_tfr['year']
    data_tfr_values = data_tfr['tfr']
    model_tfr_values = np.interp(data_years, df.index, df.tfr)  # Interpolate to match data years (??)

    # Compute mean-normalized RMSE
    rmse_scores['tfr'] = compute_rmse(model_tfr_values, data_tfr_values)

    # Plot
    pl.plot(data_tfr['year'], data_tfr['tfr'], label='World Bank', color='black')
    pl.plot(df.index, df.tfr, label='FPsim', color='cornflowerblue')
    pl.xlabel('Year')
    pl.ylabel('Rate')
    if Config.show_rmse is True:
        pl.title(f"Total Fertility Rate - Model vs Data\n(RMSE: {rmse_scores['tfr']:.2f})")
    else:
        pl.title(f'Total Fertility Rate - Model vs Data')
    pl.legend()

    save_figure('tfr.png')
    pl.show()


def plot_pop_growth(sim):
    """
    Plot annual population growth rate for model vs data
    """
    data_popsize = Config.load_validation_data(sim.pars['location'], keys=['popsize'])['popsize']
    res = sim.results

    # Import data
    data_popsize = data_popsize[data_popsize['year'] <= sim.pars['stop']]  # Restrict years to plot
    data_pop_years = data_popsize['year'].to_numpy()
    data_population = data_popsize['population'].to_numpy()

    # Extract from model
    model_growth_rate = pop_growth_rate(res.timevec, res.n_alive)
    data_growth_rate = pop_growth_rate(data_pop_years, data_population)

    # Plot
    pl.plot(data_pop_years[1:], data_growth_rate, label='World Bank', color='black')
    pl.plot(res.fp.timevec[1:], model_growth_rate, label='FPsim', color='cornflowerblue')
    pl.xlabel('Year')
    pl.ylabel('Rate')
    pl.title(f'Population Growth Rate - Model vs Data')
    pl.legend()

    save_figure('popgrowth.png')
    pl.show()


def plot_afb(sim):
    """Plot age at first birth: model vs survey data"""
    data_afb = Config.load_validation_data(sim.pars['location'], keys='afb')['afb']

    # Extract model AFB values
    model_afb = [age for age in sim.people.fp.first_birth_age if age != -1]
    model_afb = np.array(model_afb)

    # Clean and bin data AFB values
    data_afb_clean = data_afb[data_afb['afb'].apply(np.isfinite)]
    data_afb_vals = data_afb_clean['afb'].to_numpy()
    data_afb_weights = data_afb_clean['wt'].to_numpy()

    # Histogram bins
    bins = np.arange(10, 50, 1)  # From age 10 to 45 in 1-year bins

    # Bin both model and data into normalized histograms
    model_hist, _ = np.histogram(model_afb, bins=bins, density=True)
    data_hist, _ = np.histogram(data_afb_vals, bins=bins, density=True)

    # Ensure the two distributions have the same shape
    if len(model_hist) != len(data_hist):
        raise ValueError(f"Histogram size mismatch: Model = {len(model_hist)}, Data = {len(data_hist)}")

    # Compute RMSE
    rmse_scores['afb'] = compute_rmse(model_hist, data_hist)

    # Plot
    sns.histplot(model_afb, stat='proportion', kde=True, binwidth=1, color='cornflowerblue', label='FPsim')
    sns.histplot(x=data_afb_vals, stat='proportion', kde=True, weights=data_afb_weights,
                 binwidth=1, color='dimgrey', label='DHS data')
    pl.xlabel('Age at first birth')
    if Config.show_rmse:
        pl.title(f"Age at First Birth - Model vs Data\n(RMSE: {rmse_scores['afb']:.2f})")
    else:
        pl.title('Age at First Birth - Model vs Data')
    pl.legend()

    save_figure('age_first_birth.png')
    pl.show()


def plot_birth_spacing(sim):
    """
    Plot birth space and age at first birth for model vs data
    """
    # Load data
    data_spacing = Config.load_validation_data(sim.pars['location'], keys=['spacing'])['spacing']

    # Set up
    ppl = sim.people
    spacing_bins = sc.odict({'0-12': 0, '12-24': 1, '24-48': 2, '>48': 4})  # Spacing bins in months

    # Count model birth spacing
    model_spacing_counts = sc.odict().make(keys=spacing_bins.keys(), vals=0.0)
    for i in range(len(ppl)):
        if ppl.alive.values[i] and ppl.female.values[i] and ppl.age.values[i] >= min_age and ppl.age.values[i] < max_age:
            if ppl.parity.values[i] > 1:
                clean_ages = ppl.fp.birth_ages.values[i][~np.isnan(ppl.fp.birth_ages.values[i])]
                for d in range(len(clean_ages) - 1):
                    space = clean_ages[d + 1] - clean_ages[d]
                    if space > 0:
                        ind = sc.findinds(space > spacing_bins[:])[-1]
                        model_spacing_counts[ind] += 1

    # Normalize to percent
    model_spacing_counts[:] /= model_spacing_counts[:].sum()
    model_spacing_counts[:] *= 100

    # Count data birth spacing
    data_spacing_counts = sc.odict().make(keys=spacing_bins.keys(), vals=0.0)
    for _, row in data_spacing.iterrows():
        space = row['space_mo'] / 12
        ind = sc.findinds(space > spacing_bins[:])[-1]
        data_spacing_counts[ind] += row['Freq']

    # Normalize to percent
    data_spacing_counts[:] /= data_spacing_counts[:].sum()
    data_spacing_counts[:] *= 100

    # Convert to arrays for RMSE
    model_bins = np.array(model_spacing_counts.values())
    data_bins = np.array(data_spacing_counts.values())

    rmse_scores['birth_spacing'] = compute_rmse(model_bins, data_bins)

    # Diff for visualization
    diff = model_bins - data_bins
    bins_frame = pd.DataFrame({
        'Model': model_bins,
        'Data': data_bins,
        'Diff': diff
    }, index=spacing_bins.keys())

    ax = bins_frame.plot.barh(color={'Data': 'black', 'Model': 'cornflowerblue', 'Diff': 'red'})
    ax.set_xlabel('Percent of live birth spaces')
    ax.set_ylabel('Birth spacing (months)')
    if Config.show_rmse:
        ax.set_title(f"Birth Spacing - Model vs Data\n(RMSE: {rmse_scores['birth_spacing']:.2f})")
    else:
        ax.set_title('Birth Spacing - Model vs Data')

    save_figure('birth_spacing_bins.png')
    pl.show()


def plot_paid_work(sim, data_employment):
    """
    Plot rates of paid employment between model and data; only used for empowerment analyses
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
        if ppl.alive.values[i] and ppl.female.values[i] and min_age <= ppl.age.values[i] < max_age:
            age_bin = age_bins[sc.findinds(age_bins <= ppl.age.values[i])[-1]]
            total_counts[age_bin] += 1
            if ppl.paid_employment.values[i]:
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
    rmse_scores['paid_work'] = np.sqrt(np.mean((np.array(employment_data_mean) - np.array(employment_model)) ** 2))

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
    if Config.show_rmse is True:
        ax.set_title(f"Kenya: Paid Employment - Model vs Data\n(RMSE: {rmse_scores['paid_work']:.2f})")
    else:
        ax.set_title(f'Kenya: Paid Employment - Model vs Data')
    ax.set_yticks(x_pos)
    ax.set_yticklabels(labels)
    ax.legend()

    save_figure('paid_employment.png')
    pl.show()


def plot_education(sim):
    """
    Plot years of educational attainment between model and data
    """
    pl.clf()

    # Extract education from data
    data_education = Config.load_validation_data(sim.pars['location'], keys=['education'])['education']

    data_edu = data_education[['age', 'edu']].sort_values(by='age')
    data_edu = data_edu.query(f"{min_age} <= age < {max_age}").copy()
    age_bins = np.arange(min_age, max_age + 1, bin_size)
    data_edu['age_group'] = pd.cut(data_edu['age'], bins=age_bins, right=False)

    # Calculate mean and standard error for each age bin
    education_data_grouped = data_edu.groupby('age_group', observed=False)['edu']
    education_data_mean = education_data_grouped.mean().tolist()

    # Extract education from model
    model_edu_years = {age_bin: [] for age_bin in np.arange(min_age, max_age, bin_size)}
    ppl = sim.people
    for i in range(len(ppl)):
        if ppl.alive.values[i] and ppl.female.values[i] and min_age <= ppl.age.values[i] < max_age:
            age_bin = age_bins[sc.findinds(age_bins <= ppl.age.values[i])[-1]]
            model_edu_years[age_bin].append(ppl.edu.attainment.values[i])

    # Calculate average # of years of educational attainment for each age
    model_edu_mean = []
    for age_group in model_edu_years:
        if len(model_edu_years[age_group]) != 0:
            avg_edu = sum(model_edu_years[age_group]) / len(model_edu_years[age_group])
            model_edu_mean.append(avg_edu)
        else:
            model_edu_years[age_group] = 0

    # Calculate RMSE
    rmse_scores['education'] = compute_rmse(model_edu_mean, education_data_mean)

    # Set up plotting
    labels = age_bin_map
    filtered_labels = {
        k: v for k, v in labels.items()
        if int(k.split('-')[0]) >= min_age and int(k.split('-')[1]) < max_age
    }
    x_pos = np.arange(len(filtered_labels))
    fig, ax = pl.subplots()
    width = 0.35

    # Plot DHS data
    ax.barh(x_pos - width / 2, education_data_mean, width, label='DHS', color='black')

    # Plot FPsim data
    ax.barh(x_pos + width / 2, model_edu_mean, width, label='FPsim', color='cornflowerblue')

    # Set labels and title
    ax.set_xlabel('Avg Years of Education Attainment')
    ax.set_ylabel('Age Bin')
    if Config.show_rmse is True:
        ax.set_title(f"Kenya: Years of Education - Model vs Data\n(RMSE: {rmse_scores['education']:.2f})")
    else:
        ax.set_title(f'Kenya: Years of Education - Model vs Data')
    ax.set_yticks(x_pos)
    ax.set_yticklabels(filtered_labels)
    ax.legend()

    save_figure('education.png')
    pl.show()


def plot_all(sim):
    """Plots all the figures above besides empowerment plots"""

    plot_methods(sim)
    plot_cpr(sim)
    plot_tfr(sim)
    plot_afb(sim)
    plot_birth_spacing(sim)
    plot_asfr(sim)
    plot_ageparity(sim)
    plot_pop_growth(sim)
    plot_education(sim)
    return

def plot_calib(sim):
    """Plots the commonly used plots for calibration"""

    plot_methods(sim)
    plot_cpr(sim)
    plot_tfr(sim)
    plot_afb(sim)
    plot_birth_spacing(sim)
    plot_asfr(sim)
    return

