'''
A script to test plotting function for FPsim
Goal is to use plot functions to compare model vs data for specific outputs:
    - CPR
    - Maternal mortality ratio
    - Infant mortality rate
    - Population growth?
    - TFR?
'''


import numpy as np
import pandas as pd
import sciris as sc
import fpsim as fp
import pylab as pl

sc.tic()

# Set up sim for Kenya
pars = fp.pars(location='kenya')
pars['n_agents'] = 100_000 # Small population size
pars['end_year'] = 2020 # 1961 - 2020 is the normal date range

# Free parameters for calibration
pars['fecundity_var_low'] = 0.9
pars['fecundity_var_high'] = 1.1

sim = fp.Sim(pars=pars)
sim.run()

# sim.plot(to_plot = {'cpr':  'CPR (contraceptive prevalence rate)'}, do_save=True, filename='sim_cpr.png', new_fig=True)

# Save results
res = sim.results

# Import data
data_cpr = pd.read_csv('kenya_cpr.csv') # From UN Data Portal
data_cpr = data_cpr[data_cpr['year'] <= 2020] # Restrict years to plot

pl.plot(data_cpr['year'], data_cpr['cpr'], label='UN Data Portal Kenya')
pl.plot(res['t'], res['cpr']*100, label='FPsim')
pl.xlabel('Year')
pl.ylabel('Percent')
pl.title('Contraceptive Prevalence Rate in Data vs Model')
pl.legend()
pl.show()

# Plot birth spacing



# Maternal mortality ratio end of model

maternal_deaths = np.sum(self.model_results['maternal_deaths'][-mpy * 3:])
        births_last_3_years = np.sum(self.model_results['births'][-mpy * 3:])
        self.model['maternal_mortality_ratio'] = (maternal_deaths / births_last_3_years) * 100000


def model_infant_mortality_rate(self):
        infant_deaths = np.sum(self.model_results['infant_deaths'][-mpy:])
        births_last_year = np.sum(self.model_results['births'][-mpy:])
        self.model['infant_mortality_rate'] = (infant_deaths / births_last_year) * 1000

        return


def model_crude_death_rate(self):
        total_deaths = np.sum(self.model_results['deaths'][-mpy:]) + \
                       np.sum(self.model_results['infant_deaths'][-mpy:]) + \
                       np.sum(self.model_results['maternal_deaths'][-mpy:])
        self.model['crude_death_rate'] = (total_deaths / self.model_results['pop_size'][-1]) * 1000
        return


def model_crude_birth_rate(self):
        births_last_year = np.sum(self.model_results['births'][-mpy:])
        self.model['crude_birth_rate'] = (births_last_year / self.model_results['pop_size'][-1]) * 1000
        return

