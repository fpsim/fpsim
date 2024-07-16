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
