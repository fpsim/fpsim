'''
Run efficacy scenarios for the GR
'''

import sciris as sc
import fpsim as fp

# Define basic things here
method_names = fp.pars()['methods']['names']


if __name__ == '__main__':

    T = sc.timer()

    debug   = 0 # Set population size and duration
    one_sim = 0 # Just run one sim

    #%% Define sim parameters
    scen_year = 2020 # Year to start the different scenarios
    if not debug:
        pars = dict(
            n          = 10_000,
            start_year = 1980,
            end_year   = 2030,
        )
        repeats   = 10 # How many duplicates of each sim to run
    else:
        pars = dict(
            n          = 1_000,
            start_year = 2000,
            end_year   = 2010,
        )
        repeats = 3

    #%% Define scenarios

    # Increased efficacy
    eff_scen = sc.objdict(
        eff={method:0.994 for method in method_names if method != 'None'} # Set all efficacies to 1.0 except for None
    )

    # Increased uptake high efficacy
    uptake_scen = sc.objdict(
        label = 'Increased efficacy, high eff',
        year = scen_year,
        eff = {'Other modern':0.994}, # Co-opt an unused method and simulate a medium-efficacy method
        probs = [
            dict(
                source = 'None', # Source method, 'all' for all methods
                dest   = 'Other modern', # Destination
                factor = None, # Factor by which to multiply existing probability
                value  = 0.2, # Alternatively, specify the absolute probability of switching to this method
                ages   = ['>25'], # Which age keys to modify -- if not specified, all
            ),
        ]
    )

    # Increased uptake moderate efficacy
    uptake_scen_mod = sc.objdict(
        label='Increased uptake, mod eff',
        year = scen_year,
        eff = {'Other modern':0.93}, # Co-opt an unused method and simulate a medium-efficacy method
        probs = [
            dict(
                source = 'None', # Source method, 'all' for all methods
                dest   = 'Other modern', # Destination
                factor = None, # Factor by which to multiply existing probability
                value  = 0.2, # Alternatively, specify the absolute probability of switching to this method
                ages   = ['>25'], # Which age keys to modify -- if not specified, all
            ),
        ]
    )

    # Define distribution for low efficacy
    low_eff = dict(dist='uniform', par1=0.80, par2=0.90)

    # Increased uptake low efficacy
    uptake_scen_low = sc.objdict(
        label='Increased uptake, low eff',
        year = scen_year,
        eff = {'Other modern':low_eff}, # Co-opt an unused method and simulate a medium-efficacy method
        probs = [
            dict(
                source = 'None', # Source method, 'all' for all methods
                dest   = 'Other modern', # Destination
                factor = None, # Factor by which to multiply existing probability
                value  = 0.2, # Alternatively, specify the absolute probability of switching to this method
                ages   = ['>25'], # Which age keys to modify -- if not specified, all
            ),
        ]
    )

    # Increased uptake low efficacy
    uptake_2x_25 = sc.objdict(
        label='Inj 2x uptake >25 annually',
        year = scen_year,
        eff = {'Injectables': 0.983}, # Co-opt an unused method and simulate a medium-efficacy method
        probs = [
            dict(
                source = 'None', # Source method, 'all' for all methods
                dest   = 'Injectables', # Destination
                factor = 2, # Factor by which to multiply existing probability
                value  = None, # Alternatively, specify the absolute probability of switching to this method
                ages   = ['>25'], # Which age keys to modify -- if not specified, all
            ),
        ]
    )

    # Increased uptake low efficacy
    uptake_pp_20 = sc.objdict(
        label='Inj 75% prob uptake pp < 21',
        year = scen_year,
        eff = {'Injectables': 0.983}, # Co-opt an unused method and simulate a medium-efficacy method
        probs = [
            dict(
                source = 'None', # Source method, 'all' for all methods
                dest   = 'Injectables', # Destination
                factor = None, # Factor by which to multiply existing probability
                value  = 0.75, # Alternatively, specify the absolute probability of switching to this method
                ages   = ['<18', '18-20'], # Which age keys to modify -- if not specified, all
            ),
        ]
    )

    uptake_scen_20 = sc.objdict(
        label='Half disc prob inj < 21',
        year = scen_year,
        eff={'Injectables': 0.983},  # Co-opt an unused method and simulate a medium-efficacy method
        probs=[
            dict(
                source = 'Injectables',  # Source method, 'all' for all methods
                dest   = 'None',  # Destination
                factor = 0.5,  # Factor by which to multiply existing probability
                value  = None,  # Alternatively, specify the absolute probability of switching to this method
                ages   = ['<18', '18-20'],  # Which age keys to modify -- if not specified, all
            ),
        ]
    )


    #%% Create sims
    scens = fp.Scenarios(pars=pars, repeats=repeats)
    scens.add_scen(label='Baseline')
    scens.add_scen(uptake_2x_25)
    scens.add_scen(uptake_pp_20)
    scens.add_scen(uptake_scen_20)

    # Run scenarios
    scens.run()

    # Plot and print results
    scens.plot_sims()
    scens.plot_scens()
    scens.plot_cpr()

    print(scens.results.df)
    print('Done.')
    T.toc()

