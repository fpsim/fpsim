'''
A sample script for using the Calibration class to generate the optimal set of free parameters, run a sim with that set
of free params, and then generate plots showing the discrepancies between the model vs data.

PRIOR TO RUNNING:
1. Be sure to set the user global variables in the first section below (country, plotting options,
save option, and ageparity dataset name)

2. Ensure that fpsim/locations contains both a directory for the country
being calibrated as well as a corresponding location file (i.e. 'ethiopia.py')

3. In order to run this script, the country data must be stored in the country directory mentioned above and with the
following naming conventions:

ageparity.csv' # Age-parity distribution file
use.csv' # Dichotomous contraceptive method use
birth_spacing_dhs.csv'  # Birth-to-birth interval data
afb.table.csv'  # Ages at first birth in DHS for women age 25-50
cpr.csv'  # Contraceptive prevalence rate data; from UN Data Portal
asfr.csv'  # Age-specific data fertility rate data
mix.csv'  # Contraceptive method mix
tfr.csv'  # Total fertility rate data
popsize.csv'  # Population by year

4. Ensure that the data in the aforementioned files is formatted in the same manner as the kenya data files,
which were used as a standard in writing this script.

'''
import os
import numpy as np
import pandas as pd
import sciris as sc
import fpsim as fp
from fpsim import plotting as plt

"""
Calibrate the full-empowered complexity model
"""
from fpsim.methods import EmpoweredChoice
from fpsim import plotting as plt

location = 'kenya'
do_save = 1

def run_sim():
    pars = fp.pars(location=location)

    # Settings
    pars['n_agents'] = 10_000
    pars['start_year'] = 2000
    pars['end_year'] = 2020

    # Free parameters for calibration

    # Free parameters for calibration
    freepars = dict(
            fecundity_var_low=[1, 0.925, 0.975],
            fecundity_var_high=[1.8, 1.5, 1.85],
            exposure_factor=[1.8, 0.95, 1.9],
    )

    # Adjust contraceptive choice parameters
    cm_pars = dict(
        prob_use_intercept=1.8,
        prob_use_year=1970,
        prob_use_trend_par=.023,
        method_weights=np.array([.009, 1.4, 8.8, .06, .12, .08, 14, 2, .12]),
    )

    empow_pars = dict(
        age_weights=np.array([-.2, .9, 1, -.1, -.5, -.3, .9])
    )

    method_choice = EmpoweredChoice(pars=cm_pars, location=location)
    empwr = fp.Empowerment(pars=empow_pars)
    sim = fp.Sim(
        pars=pars,
        contraception_module=method_choice,
        analyzers=[fp.cpr_by_age(), fp.method_mix_by_age(), fp.education_recorder()],
        education_module=fp.Education(location=location),
        empowerment_module=empwr
    )

    calibration = fp.Calibration(pars, calib_pars=freepars)
    calibration.calibrate()
    pars.update(calibration.best_pars)
    sim.run()

    return sim


if __name__ == '__main__':

    cwd = os.path.abspath(os.getcwd())
    country_data_dir = f'../../fpsim/locations/{location}/data'
    figs_dir = os.path.join(country_data_dir, 'figs')
    if do_save and not os.path.exists(figs_dir):
            os.makedirs(figs_dir, exist_ok=True)

    sim = run_sim()

    # Load in validation data
    val_data_list = ['ageparity', 'use', 'spacing', 'afb', 'mcpr', 'asfr', 'methods', 'tfr', 'popsize', 'empowerment', 'education']
    val_data_sources = fp.locations.kenya.filenames()
    val_data = sc.objdict()
    for vd in val_data_list:
        val_data[vd] = pd.read_csv(f"{val_data_sources['base']}/{val_data_sources[vd]}")

    # Plot calibration figures
    plt.plot_emp_calib(sim, val_data)

