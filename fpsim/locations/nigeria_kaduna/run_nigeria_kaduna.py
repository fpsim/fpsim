"""
Script to create a calibrated model of Nigeria Kaduna.
Run the model and generate plots showing the
discrepancies between the model and data.

Users may update values marked as USER-EDITABLE to match the context
they are modeling for a specific version of FPsim.
"""
# If running a local instance of fpsim, set path
# import sys

import numpy as np
import fpsim as fp
import pandas as pd
import sciris as sc
import starsim as ss
import matplotlib.pyplot as plt
from fpsim import plotting as fpplt
import os

# Settings
fpplt.Config.set_figs_directory('calib_results/figures/')
fpplt.Config.do_save = True
fpplt.Config.do_show = False
fpplt.Config.show_rmse = True

def plot_calib(sim, single_fig=False, fig_kwargs=None, legend_kwargs=None):
    """ Plots the commonly used plots for calibration
    Plotting class function which plots the primary calibration targets:
    method mix, method use, cpr, total fertility rate, birth spacing, 
    age at first birth, and age-specific fertility rate.
    
    """

    fpplt.plot_calib(sim)

    return


if __name__ == '__main__':
    do_run = True  # Whether to run the sim or load from file
    country = 'nigeria_kaduna'

    if do_run:
        # Create simulation with parameters
        pars=dict(
            location=country,
            n_agents=5000,
            end_year=2020,
        )
        sim = fp.Sim(pars=pars)
        sim.init()
        sim.run()

        # Create directory in current directory if it doesn't exist
        os.makedirs('calib_results', exist_ok=True)
        sc.saveobj(f'calib_results/{country}_calib.sim', sim)
    else:
        os.makedirs('calib_results', exist_ok=True)
        sim = sc.loadobj(f'calib_results/{country}_calib.sim')

    # Set options for plotting
    sc.options(fontsize=20)  # Set fontsize
    plot_calib(sim, single_fig=True)