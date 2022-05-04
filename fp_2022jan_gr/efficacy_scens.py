'''
Run efficacy scenarios for the GR
'''

import sciris as sc
import fpsim as fp
import fp_analyses as fa

# Define basic things here
default_pars = fa.senegal_parameters.make_pars()
method_names = default_pars['methods']['names']


if __name__ == '__main__':

    debug   = False # Set population size and duration
    one_sim = False # Just run one sim

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
    # eff = fp.update_methods(scen_year, eff_scen) # Create intervention

    # Increased uptake high efficacy
    uptake_scen = sc.objdict(
        label='Increased efficacy, high eff',
        eff = {'Other modern':0.994}, # Co-opt an unused method and simulate a medium-efficacy method
        probs = [
            dict(
                source = 'None', # Source method, 'all' for all methods
                dest   = 'Other modern', # Destination
                factor = None, # Factor by which to multiply existing probability
                value  = 0.2, # Alternatively, specify the absolute probability of switching to this method
                keys   = ['>25'], # Which age keys to modify -- if not specified, all
            ),
        ]
    )
    # uptake = fp.update_methods(scen_year, uptake_scen) # Create intervention


    # Increased uptake moderate efficacy
    uptake_scen_mod = sc.objdict(
        label='Increased uptake, mod eff',
        eff = {'Other modern':0.93}, # Co-opt an unused method and simulate a medium-efficacy method
        probs = [
            dict(
                source = 'None', # Source method, 'all' for all methods
                dest   = 'Other modern', # Destination
                factor = None, # Factor by which to multiply existing probability
                value  = 0.2, # Alternatively, specify the absolute probability of switching to this method
                keys   = ['>25'], # Which age keys to modify -- if not specified, all
            ),
        ]
    )
    # uptake_mod = fp.update_methods(scen_year, uptake_scen_mod) # Create intervention

    # Define distribution for low efficacy
    low_eff = dict(dist='uniform', par1=0.80, par2=0.90)

    # Increased uptake low efficacy
    uptake_scen_low = sc.objdict(
        label='Increased uptake, low eff',
        eff = {'Other modern':low_eff}, # Co-opt an unused method and simulate a medium-efficacy method
        probs = [
            dict(
                source = 'None', # Source method, 'all' for all methods
                dest   = 'Other modern', # Destination
                factor = None, # Factor by which to multiply existing probability
                value  = 0.2, # Alternatively, specify the absolute probability of switching to this method
                keys   = ['>25'], # Which age keys to modify -- if not specified, all
            ),
        ]
    )
    # uptake_low = fp.update_methods(scen_year, uptake_scen_low) # Create intervention


    # Increased uptake low efficacy
    uptake_scen_25 = sc.objdict(
        label='Inj 2x uptake >25 annually',
        eff = {'Injectables': 0.983}, # Co-opt an unused method and simulate a medium-efficacy method
        probs = [
            dict(
                source = 'None', # Source method, 'all' for all methods
                dest   = 'Injectables', # Destination
                factor = 2, # Factor by which to multiply existing probability
                value  = None, # Alternatively, specify the absolute probability of switching to this method
                keys   = ['>25'], # Which age keys to modify -- if not specified, all
            ),
        ]
    )
    # uptake_2x_25 = fp.update_methods(scen_year, uptake_scen_25) # Create intervention

    # Increased uptake low efficacy
    uptake_scen_20 = sc.objdict(
        label='Inj 75% prob uptake pp < 21',
        eff = {'Injectables': 0.983}, # Co-opt an unused method and simulate a medium-efficacy method
        probs = [
            dict(
                source = 'None', # Source method, 'all' for all methods
                dest   = 'Injectables', # Destination
                factor = None, # Factor by which to multiply existing probability
                value  = 0.75, # Alternatively, specify the absolute probability of switching to this method
                keys   = ['<18', '18-20'], # Which age keys to modify -- if not specified, all
            ),
        ]
    )
    # uptake_pp_20 = fp.update_methods(scen_year, uptake_scen_20, matrix='probs_matrix_1-6') # Create intervention

    uptake_scen_20 = sc.objdict(
        label='Half disc prob inj < 21',
        eff={'Injectables': 0.983},  # Co-opt an unused method and simulate a medium-efficacy method
        probs=[
            dict(
                source='Injectables',  # Source method, 'all' for all methods
                dest='None',  # Destination
                factor=0.5,  # Factor by which to multiply existing probability
                value=None,  # Alternatively, specify the absolute probability of switching to this method
                keys=['<18', '18-20'],  # Which age keys to modify -- if not specified, all
            ),
        ]
    )
    # disc = fp.update_methods(scen_year, uptake_scen_20)  # Create intervention


    #%% Create sims
    scens = fp.Scenarios(pars=pars, repeats=repeats)
    scens.add_scen(label='Baseline')
    scens.add_scen(uptake_2x_25)
    scens.add_scen(uptake_pp_20)
    scens.add_scen(uptake_scen_20)

    # Run scenarios
    scens.run()

    # Plot and print results
    scens.plot(plot_sims=False)
    scens.plot(plot_sims=True)

    print(scens.results.df)


    # sims1 = make_sims(repeats=repeats, label='Baseline', **pars)
    # # sims2 = make_sims(repeats=repeats, interventions=eff, , **pars)
    # # sims3 = make_sims(repeats=repeats, interventions=uptake, label='Increased uptake, high eff', **pars)
    # # sims4 = make_sims(repeats=repeats, interventions=uptake_mod, , **pars)
    # # sims5 = make_sims(repeats=repeats, interventions=uptake_low, label=, **pars)
    # sims6 = make_sims(repeats=repeats, interventions=uptake_2x_25, , **pars)
    # sims7 = make_sims(repeats=repeats, interventions=uptake_pp_20, , **pars)
    # sims8 = make_sims(repeats=repeats, interventions=disc,
    #                   , **pars)


    # #%% Run
    # if one_sim:
    #     sim = sims1[4]
    #     sim.run()
    #     sim.plot()

    # else:
    #     msim = run_sims(sims1,
    #                     # sims2,
    #                     # sims3,
    #                     # sims4,
    #                     # sims5,
    #                     sims6,
    #                     sims7,
    #                     sims8)


    #%% Plotting
    # msim2 = msim.remerge()
    # msim.plot(plot_sims=False) # This plots all 15 individual sims as lines of 3 colors
    # msim2.plot(plot_sims=True) # This plots 3 multisims with uncertainty bands in the same 3 colors


    #     # Analyze
    # results = analyze_sims(msim)
    # print(results.df)
    # print(results.stats)
    print('Done.')

