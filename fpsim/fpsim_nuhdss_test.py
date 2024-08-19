#----------------------------------------------------The baseline fitting--------------------------------------------------------------------------------------------------------------------#
import fpsim as fp


def run_baseline():
    pars = dict(
        n_agents   = 756,
        location   = 'nuhdss',
        start_year = 2012,
        end_year   = 2035,
    )

    sim = fp.Sim(pars)
    sim.run()
    sim.plot()

#---------------------------------Running campaigns for each individual methods of family planning e.g IUD at a single coverage--------------------------------------------------------------#

def run_impl_campaign():
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

#----------------------------------------------------Running campaigns for Injectables at a single coverage---------------------------------------------------------------------------------#

def run_inj_campaign():
    n_agents = 10000
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

#---------------------------------implementation of the male involvement-------------------------------------------------------------------------------------------------------------------#

def run_male_inv():
    # Parameters
    n_agents = 10000
    start_year = 2012
    end_year = 2030

    effect_size = 0.60

    # Initialize factors for male involvement interventions
    approval_change_factor = 1.0 + effect_size * 0.5  # Effect on approval change

    ### HERE WE NEED TO DEFINE AN INTERVENTION
    # Define intervention params
    # NEED TO DEFINE INTERVENTION PARAMS HERE; BELOW ARE EXAMPLES ONLY
    year = 2015
    val = .5
    par = 'exposure_factor'
    info_intervention = fp.change_par(par=par, years=year, vals=val, verbose=True)

    # Create scenarios for male involvement interventions
    scen_info_sessions = fp.make_scen(interventions=info_intervention)

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
    scens.add_scen(scen_info_sessions, label='Male Involvement')

    # Run and plot the scenarios
    scens.run()
    scens.plot()

#----------------------------------------------implementaion of campaigns at different rates on the MCPR and other rates---------------------------------------------------------------------#

def run_campaign_coverage():
    # Number of agents to simulate
    n_agents = 10000

    # Define the start and end year for the simulation
    start_year = 2012
    end_year = 2030

    # Define the effect size
    effect_size = 0.6

    # Coverage levels to simulate
    coverages = [0.10, 0.30, 0.60]

    # Create a list to store scenarios
    scenarios = []

    # Create scenarios for each coverage level
    for coverage in coverages:
        init_factor = 1.0 + effect_size * coverage
        scen = fp.make_scen(method='Injectables', init_factor=init_factor, year=2025)
        scenarios.append((scen, f'Campaign - { coverage}'))

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


if __name__ == '__main__':
    #run_baseline()
    #run_impl_campaign()
    #run_inj_campaign()
    run_male_inv()
    #run_campaign_coverage()


#----------------------------------------------implementation of male involvement on the MCPR and other rates--------------------------------------------------------------------------------#
