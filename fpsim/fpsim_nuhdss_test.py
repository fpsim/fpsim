import fpsim as fp

pars = dict(
    n_agents   = 756,
    location   = 'nuhdss',
    start_year = 2012,
    end_year   = 2035,
)

sim = fp.Sim(pars)
sim.run()
fig = sim.plot()



#sim.to_df()
#sim.df.to_csv(r'results.csv')
#print("Done.")


import fpsim as fp

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


import fpsim as fp

def run_simulation():
    n_agents = 50000
    start_year = 2012
    end_year = 2030

    effect_size = 0.6
    coverage = 0.60
    init_factor = 1.0 + effect_size * coverage
    scen = fp.make_scen(method='Injectables', init_factor=init_factor, year=2025)

    pars = fp.pars(location='nuhdss', n_agents=n_agents, start_year=start_year, end_year=end_year)

    scens = fp.Scenarios(pars=pars, repeats=3)
    scens.add_scen(label='Baseline')
    scens.add_scen(scen, label='Campaign')
    scens.run()
    scens.plot()

if __name__ == '__main__':
    run_simulation()


import fpsim as fp

# Parameters
n_agents = 756
start_year = 2012
end_year = 2035

effect_size = 0.52

# Initialize factors for male involvement interventions
init_factor_info_sessions = 1.0 + effect_size  # Effect of informational sessions
approval_change_factor = 1.0 + effect_size * 0.5  # Effect on approval change

# Create scenarios for male involvement interventions
scen_info_sessions = fp.make_scen(
    method='Informational_sessions',
    init_factor=init_factor_info_sessions,
    year=2025
)

# Set up the parameters for the simulation
pars = fp.pars(
    location='nuhdss',
    n_agents=n_agents,
    start_year=start_year,
    end_year=end_year
)

# Create the Scenarios object and add scenarios
scens = fp.Scenarios(pars=pars, repeats=3)
scens.add_scen(label='Baseline')
scens.add_scen(scen_info_sessions, label='Informational Sessions')

# Run and plot the scenarios
scens.run()
scens.plot()

if __name__ == '__main__':
    run_simulation()

import fpsim as fp

def run_simulation():
    # Number of agents to simulate
    n_agents = 50000
    
    # Define the start and end year for the simulation
    start_year = 2012
    end_year = 2030

    # Define the effect size
    effect_size = 0.6
    
    # Coverage levels to simulate
    coverages = [0.45, 0.6, 0.8]
    
    # Create a list to store scenarios
    scenarios = []

    # Create scenarios for each coverage level
    for coverage in coverages:
        init_factor = 1.0 + effect_size * coverage
        scen = fp.make_scen(method='Injectables', init_factor=init_factor, year=2025)
        scenarios.append((scen, f'Campaign_{coverage}'))
    
    # Set the parameters for the simulation including location, number of agents, and time frame
    pars = fp.pars(location='nuhdss', n_agents=n_agents, start_year=start_year, end_year=end_year)

    # Create a Scenarios object with the defined parameters and set the number of repeats
    scens = fp.Scenarios(pars=pars, repeats=3)
    
    # Add the baseline scenario
    scens.add_scen(label='Baseline')
    
    # Add the campaign scenarios
    for scen, label in scenarios:
        scens.add_scen(scen, label=label)
    
    # Run the simulation for all scenarios
    scens.run()
    
    # Plot the results of the simulation
    scens.plot()

# Run the simulation when the script is executed
if __name__ == '__main__':
    run_simulation()
