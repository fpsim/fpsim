import fpsim as fp
import numpy as np
#--------------------------------------------------------Static and continous campaign implementation--------------------------------------------------------------------------------------------------#"
n_agents = 10_000
start_year = 2000
end_year = 2040

effect_size = 0.6
coverage = 0.75
init_factor = 1.0 + effect_size * coverage

scen = fp.make_scen(method='Implants', init_factor=init_factor, year=2025)

pars = fp.pars(location='nuhdss', n_agents=n_agents, start_year=start_year, end_year=end_year)

scens = fp.Scenarios(pars=pars, repeats=3)
scens.add_scen(label='Baseline')
scens.add_scen(scen, label='Campaign')
scens.run()
scens.plot()

#"-------------------------------------------------implementation of the logistic growth model for the campaigns---------------------------------------------------------------------------------------"#

n_agents = 10_000
start_year = 2000
end_year = 2040

# Logistic growth parameters
C_max = 0.9                                                                             # Maximum contraceptive uptake (90%)
C_0 = 0.05                                                                              # Initial contraceptive uptake (5%)
t0 = 2030                                                                               # Year when 50% of max coverage is reached
r = 0.3                                                                                 # Growth rate (adjustable)
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
scens.add_scen(scen, label='Gradual Campaign Roll-out')
scens.run()
scens.plot()


#"-------------------------------------------------------------implementation of the periodic campaigns----------------------------------------------------------------------------------------------"#

n_agents = 10_000
start_year = 2000
end_year = 2040

# Define periodic campaign parameters
C_max = 0.6                                                                                               # Maximum campaign effect
T = 1                                                                                                     # Campaign repeats every 1 year (annually)
t0 = 2025                                                                                                 # Campaign starts in 2025

# Function to compute time-dependent coverage factor
def periodic_campaign(year):
    return 1.0 + C_max * np.sin(np.pi * (year - t0) / T) ** 2                                             # Implementing given equation

# Define dynamic scenario where campaign effect changes over time
scen = fp.make_scen(method='Implants', init_factor=periodic_campaign, year=np.arange(start_year, end_year + 1))

# Set up the simulation parameters
pars = fp.pars(location='nuhdss', n_agents=n_agents, start_year=start_year, end_year=end_year)

# Define scenarios
scens = fp.Scenarios(pars=pars, repeats=3)
scens.add_scen(label='Baseline')
scens.add_scen(scen, label='Periodic Campaign')

# Run the scenarios and plot 
scens.run()
scens.plot()


#"----------------------------------------------------------------implementation of the pulse campaigns-----------------------------------------------------------------------------------------------"#

# Define simulation parameters
n_agents = 10_000
start_year = 2000
end_year = 2040

# Define pulse campaign parameters
C_max = 0.8                                                                                                      # Maximum coverage 
t_campaign = 2025                                                                                                # Year when the pulse campaign happens

# Pulse campaign function (approximating Dirac delta function)
def pulse_campaign(year):
    return C_max if year == t_campaign else 1.0                                                                  # Apply effect only at t_campaign

# Generate a range of years for visualization
years = np.arange(start_year, end_year + 1)
coverage_effect = np.array([pulse_campaign(y) for y in years])

# Define scenario with pulse campaign
scen = fp.make_scen(method='Implants', init_factor=pulse_campaign, year=years)

# Set up simulation parameters
pars = fp.pars(location='nuhdss', n_agents=n_agents, start_year=start_year, end_year=end_year)

# Define scenarios
scens = fp.Scenarios(pars=pars, repeats=3)
scens.add_scen(label='Baseline')
scens.add_scen(scen, label='Pulse Campaign')
# Run the simulation
scens.run()
scens.plot()


#"----------------------------------------------------------------implementation of the exponential campaigns-----------------------------------------------------------------------------------------"#

# Define simulation parameters
n_agents = 10_000
start_year = 2000
end_year = 2040

years = np.arange(start_year, end_year + 1)  # Simulation time range

# Exponential Roll-out Parameters 
C_max_exp = 0.85                                                                                                  # Maximum coverage (saturation level)
r_exp = 0.1                                                                                                       # Growth rate

# Exponential campaign function
def exponential_campaign(year):
    return C_max_exp * (1 - np.exp(-r_exp * (year - start_year))) if year >= start_year else 1.0

# Generate campaign effect over time for visualization
coverage_exponential = np.array([exponential_campaign(y) for y in years])

scen_exponential = fp.make_scen(method='Implants', init_factor=exponential_campaign, year=years)

# Set up FPSim parameters
pars = fp.pars(location='nuhdss', n_agents=n_agents, start_year=start_year, end_year=end_year)

# Initialize and run simulation
scens = fp.Scenarios(pars=pars, repeats=3)
scens.add_scen(label='Baseline')
scens.add_scen(scen_exponential, label='Exponential Roll-out')
scens.run()
scens.plot()

#-------------------------------------------------------------implementation of all the campaigns at once----------------------------------------------------------------------------------------------#