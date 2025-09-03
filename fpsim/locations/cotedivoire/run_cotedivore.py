"""
Script to create a calibrated model of Niger.
Run the model and generate plots showing the
discrepancies between the model and data.
"""
import numpy as np
import fpsim as fp
import pandas as pd
import sciris as sc
import starsim as ss
import matplotlib.pyplot as pl
from fpsim import plotting as fpplt

import sys
sys.path.insert(0, '/Users/')
sys.path.insert(0, '/Users/laurynbruce/')
sys.path.insert(0, '/Users/laurynbruce/')
sys.path.insert(0, '/Users/laurynbruce/Documents/')
sys.path.insert(0, '/Users/laurynbruce/Documents/Lauryn_Bruce/UCSD/')
sys.path.insert(0, '/Users/laurynbruce/Documents/Lauryn_Bruce/UCSD/research/')
sys.path.insert(0, '/Users/laurynbruce/Documents/Lauryn_Bruce/UCSD/research/Gates/')
sys.path.insert(0, '/Users/laurynbruce/Documents/Lauryn_Bruce/UCSD/research/Gates/project/')
sys.path.insert(0, '/Users/laurynbruce/Documents/Lauryn_Bruce/UCSD/research/Gates/project/fpsim/')
sys.path.insert(0, '/Users/laurynbruce/Documents/Lauryn_Bruce/UCSD/research/Gates/project/fpsim/fpsim/')

# Settings
country = 'cotedivoire'
fpplt.Config.set_figs_directory('figures/')
fpplt.Config.do_save = True
fpplt.Config.do_show = False
fpplt.Config.show_rmse = False


def make_pars():
    pars = fp.make_fp_pars()  # For default pars
    pars.update_location(country)

    # Modify individual fecundity and exposure parameters
    # These adjust each woman's probability of conception and exposure to pregnancy.
    pars['fecundity_var_low'] = 1
    pars['fecundity_var_high'] = 1
    pars['exposure_factor'] = 1
    
    # Adjust contraceptive choice parameters
    pars['prob_use_year'] = 2020  # Base year
    pars['prob_use_trend_par'] = 0.06  # Time trend in contraceptive use - adjust this to get steeper/slower trend
    pars['prob_use_intercept'] = -0.7  # Intercept for the probability of using contraception - shifts the mCPR level
    pars['method_weights'] = np.array([1, 15, 4, 0.1, 1, 1, 2, 0.2, 1])  # Weights for the methods in method_list in methods.py (excluding 'none', so starting with 'pill' and ending in 'othmod').

    # Postpartum sexual activity correction or 'birth spacing preference'. Pulls values from {location}/data/birth_spacing_pref.csv by default
    # Set all to 1 to reset. Option to use 'optimize-space-prefs.py' script in this directory to determine values
    # 'months': array([ 0.,  3.,  6.,  9., 12., 15., 18., 21., 24., 27., 30., 33., 36., 39., 42., 45., 48., 51., 54.]),
    # The probability of sex --> very indirect, so need a larger term, when you are 2 years postpartum, dhs data sexual activity, probability of sex
    pars['spacing_pref']['preference'][:3] =  1  # Spacing of 0-6 months
    pars['spacing_pref']['preference'][3:6] = 1  # Spacing of 9-15 months
    pars['spacing_pref']['preference'][6:9] = 2  # Spacing of 18-24 months
    pars['spacing_pref']['preference'][9:] =  0.7  # Spacing of 27-54 months 

    return pars


def make_sim(pars=None, stop=2021):
    if pars is None:
        pars = make_pars()

    # Run the sim
    sim = fp.Sim(
        start=2000,
        stop=stop,
        n_agents=1000,
        location=country,
        pars=pars,
        analyzers=[fp.cpr_by_age(), fp.method_mix_by_age()],
    )

    return sim


def plot_cpr(sim, start_year=2005, end_year=None, ax=None, legend_kwargs={}):
    '''
    Plot contraceptive prevalence rate for model vs data
    '''

    # Import data
    fpet = pd.read_csv('mcpr.csv')
    res = sim.results

    # Data to plot
    plot_data = fpet.loc[fpet.year >= start_year]
    si = sc.findfirst(res['timevec'] >= start_year)

    # Plot
    ax.plot(plot_data['year'], plot_data['50%']*100, label='FPET', color='black')
    ax.fill_between(plot_data['year'], plot_data['2.5%']*100, plot_data['97.5%']*100, color='lightgray', label='FPET 95% CI')
    ax.plot(res['timevec'].years[si:], res.contraception.mcpr[si:] * 100, label='FPsim', color='cornflowerblue')
    ax.set_xlabel('Year')
    ax.set_ylabel('Percent')
    ax.set_title(f'mCPR')
    ax.legend(**legend_kwargs)

    return ax


def plot_calib(sim, single_fig=False, fig_kwargs=None, legend_kwargs=None):
    """ Plots the commonly used plots for calibration """

    if legend_kwargs is None:
        legend_kwargs = {'frameon': False, 'loc': 'best', 'fontsize': 15}

    if single_fig:
        if fig_kwargs is None:
            fig_kwargs = {'figsize': (15, 9)}
        fig, axes = pl.subplots(2, 3, **fig_kwargs)
        axes = axes.flatten()

    def ax_arg(i):
        """Returns the appropriate axis for plotting"""
        return axes[i] if single_fig else None

    plot_cpr(sim, ax=ax_arg(0), legend_kwargs=legend_kwargs)
    fpplt.plot_tfr(sim, ax=ax_arg(1), legend_kwargs=legend_kwargs)
    fpplt.plot_method_use(sim, ax=ax_arg(2), legend_kwargs=legend_kwargs)
    fpplt.plot_method_mix(sim, ax=ax_arg(3), legend_kwargs=legend_kwargs)
    fpplt.plot_afb(sim, ax=ax_arg(4), legend_kwargs=legend_kwargs)
    fpplt.plot_birth_spacing(sim, ax=ax_arg(5), legend_kwargs=legend_kwargs)

    if single_fig:
        fig.tight_layout()
        fig_name = 'figures/calibration_plots.png'
        sc.savefig(fig_name, dpi=100)

    return


if __name__ == '__main__':
    do_run = True  # Whether to run the sim or load from file
    if do_run:
        # Create simulation with parameters
        sim = make_sim()
        sim.run()
        sc.saveobj(f'results/{country}_calib.sim', sim)
    else:
        sim = sc.loadobj(f'results/{country}_calib.sim')

    # Set options for plotting
    # sc.options(fontsize=20)  # Set fontsize
    plot_calib(sim, single_fig=True)

