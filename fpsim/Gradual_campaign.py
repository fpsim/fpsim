#----------------------------------------------------------The baseline fitting-------------------------------------------------------------------------------------------------------------------------#
import sciris as sc
import fpsim as fp
import matplotlib.pyplot as plt
import numpy as np

def run_baseline():
    pars = dict(
        n_agents   = 46750,
        location   = 'nuhdss',
        start_year = 2012,
        end_year   = 2040,
    )

    sim = fp.Sim(pars)
    sim.run()
    sim.plot()

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#  
#------------------------------------------------------------------------Gradual roll-out of campaigns-------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

#---------------------------------Running campaigns for each individual methods of family planning e.g IUD at a single coverage------------------------------------------------------------------------#

def run_impl_campaign():
    n_agents = 10_000
    start_year = 2024
    end_year = 2040
  # Define logistic growth parameters
    C_max = 0.9                                                                             # Maximum contraceptive uptake (90%)
    C_0 = 0.02                                                                              # Initial contraceptive uptake (5%)
    t0 = 2030                                                                               # Year when 50% of max coverage is reached
    r = 0.2                                                                                 # Growth rate (adjustable)
    a = (C_max / C_0) - 1                                                                   # Derived from initial coverage
    # Logistic growth function for campaign roll-out
    def logistic_campaign(year):
        return C_max / (1 + a * np.exp(-r * (year - t0)))
    # Generate a range of years and compute coverage

    gradual_scen = None
    for year in range(start_year, end_year + 1):
        coverage_effect = logistic_campaign(year)

        # Define dynamic scenario with logistic campaign roll-out
        gradual_scen += fp.make_scen(method='Implants', init_factor=coverage_effect, year=year)

    # Set up simulation parameters
    pars = fp.pars(location='nuhdss', n_agents=n_agents, start_year=start_year, end_year=end_year)
    # Define scenarios
    scens = fp.Scenarios(pars=pars, repeats=3)
    scens.add_scen(label='Baseline')
    scens.add_scen(gradual_scen, label='Gradual Campaign Roll-out')
    scens.run()
    scens.plot()


##########################################################################################################################################


    


#Resolving this

import sciris as sc
import fpsim as fp
import numpy as np

def run_impl_campaign1():
    n_agents = 10_000
    start_year = 2024
    end_year = 2040

    # Define logistic growth parameters
    C_max = 0.9  # Maximum contraceptive uptake (90%)
    C_0 = 0.02   # Initial contraceptive uptake (2%)
    t0 = 2030    # Year when 50% of max coverage is reached
    r = 0.2      # Growth rate

    a = (C_max / C_0) - 1  # Derived from initial coverage

    # Logistic growth function for campaign roll-out
    def logistic_campaign(year):
        return C_max / (1 + a * np.exp(-r * (year - t0)))

    # Define dynamic scenario changes
    changes = []
    for year in range(start_year, end_year + 1):
        coverage_effect = logistic_campaign(year)
        changes.append({'method': 'Implants', 'init_factor': coverage_effect, 'year': year})

    # Create scenario
    gradual_scen = fp.make_scen(changes)

    # Set up simulation parameters
    pars = fp.pars(location='nuhdss', n_agents=n_agents, start_year=start_year, end_year=end_year)

    # Define scenarios
    scens = fp.Scenarios(pars=pars, repeats=3)
    scens.add_scen(label='Baseline')
    scens.add_scen(gradual_scen, label='Gradual Campaign Roll-out')

    # Run and plot scenarios
    scens.run()
    scens.plot()



import sciris as sc
import fpsim as fp
import numpy as np
import matplotlib.pyplot as plt

def logistic_growth(years, C_max, C_0, t0, r):
    """Compute logistic growth values for MCPR."""
    a = (C_max / C_0) - 1
    return C_max / (1 + a * np.exp(-r * (years - t0)))

def run_scenarios():
    n_agents = 10000
    start_year = 2012  # Updated to start from 2012
    campaign_start = 2024  # Campaign starts in 2024
    end_year = 2040
    years = np.arange(start_year, end_year + 1)

    # Define logistic growth parameters
    C_max = 0.9  # Maximum contraceptive uptake (90%)
    C_0 = 0.02   # Initial contraceptive uptake (2%)
    t0 = 2030    # Year when 50% of max coverage is reached
    r = 0.2      # Growth rate

    # Define gradual scenario with logistic growth only from 2024
    gradual_mcpr = np.full_like(years, C_0, dtype=float)  # Set initial values to C_0
    gradual_mcpr[years >= campaign_start] = logistic_growth(years[years >= campaign_start], C_max, C_0, t0, r)
    
    baseline_mcpr = logistic_growth(years, C_max, C_0, t0 + 5, r * 0.8)  # Slower growth for baseline

    # Create scenario modifications
    changes_gradual = [{'method': 'Implants', 'init_factor': gradual_mcpr[i], 'year': years[i]} for i in range(len(years)) if years[i] >= campaign_start]

    # Define simulation parameters
    pars = fp.pars(location='nuhdss', n_agents=n_agents, start_year=start_year, end_year=end_year)
    scens = fp.Scenarios(pars=pars, repeats=3)
    scens.add_scen(fp.make_scen([]), label='Baseline')  # Unchanged baseline
    scens.add_scen(fp.make_scen(changes_gradual), label='Gradual Campaign')

    # Run and plot
    scens.run()
    scens.plot()




    #   Hard coded values

import sciris as sc
import fpsim as fp
import numpy as np
import matplotlib.pyplot as plt

def logistic_growth(years, C_max, C_0, t0, r):
    """Compute logistic growth values for MCPR."""
    a = (C_max / C_0) - 1
    return C_max / (1 + a * np.exp(-r * (years - t0)))

def run_scenarios1():
    n_agents = 10000
    start_year = 2012  # Baseline starts from 2012
    campaign_start = 2024  # Campaign starts in 2024
    end_year = 2040
    years = np.arange(start_year, end_year + 1)

    # Define initial values for key indicators (2012-2015)
    initial_values = {
        'mcp': {2012: 15.0, 2013: 18.0, 2014: 20.0, 2015: 23.0},  # Modern contraceptive prevalence (%)
        'births': {2012: 1200, 2013: 1300, 2014: 1350, 2015: 1400},  # Live births
        'stillbirths': {2012: 30, 2013: 32, 2014: 35, 2015: 38},  # Stillbirths
        'maternal_deaths': {2012: 10, 2013: 12, 2014: 14, 2015: 16},  # Maternal deaths
        'infant_deaths': {2012: 80, 2013: 85, 2014: 90, 2015: 95},  # Infant deaths
        'imr': {2012: 45.0, 2013: 42.0, 2014: 40.0, 2015: 38.0}  # Infant mortality rate (per 1,000 live births)
    }

    # Define logistic growth parameters
    C_max = 0.9  # Maximum contraceptive uptake (90%)
    C_0 = 0.02   # Initial contraceptive uptake (2%)
    t0 = 2030    # Year when 50% of max coverage is reached
    r = 0.2      # Growth rate

    # Define MCPR scenarios
    gradual_mcpr = np.array([initial_values['mcp'].get(year, C_0) for year in years], dtype=float)
    gradual_mcpr[years >= campaign_start] = logistic_growth(years[years >= campaign_start], C_max, C_0, t0, r)
    
    baseline_mcpr = np.array([initial_values['mcp'].get(year, logistic_growth(year, C_max, C_0, t0 + 5, r * 0.8)) for year in years])

    # Create scenario modifications
    changes_gradual = [{'method': 'Implants', 'init_factor': gradual_mcpr[i], 'year': years[i]} for i in range(len(years)) if years[i] >= campaign_start]

    # Define simulation parameters
    pars = fp.pars(location='nuhdss', n_agents=n_agents, start_year=start_year, end_year=end_year)
    scens = fp.Scenarios(pars=pars, repeats=3)
    scens.add_scen(fp.make_scen([]), label='Baseline')  # Unchanged baseline
    scens.add_scen(fp.make_scen(changes_gradual), label='Gradual Campaign')

    # Run and plot
    scens.run()
    scens.plot()

    



if __name__ == '__main__':
   
    #run_scenarios()
    run_scenarios1()