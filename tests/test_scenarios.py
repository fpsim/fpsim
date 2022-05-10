"""
Run tests on the Scenarios class.
"""

import sciris as sc
import fpsim as fp

# Global settings
p = sc.objdict() # Custom parameters
p.n          = 500 # Population size
p.start_year = 2000 # Start year of sims
p.end_year   = 2010 # End year of sims
p.verbose    = 0 # Verbosity to use
int_year = 2002 # Year to start the interventions
serial   = False # Whether to run in serial (for debugging)
do_plot  = True # Whether to do plotting in interactive mode


def make_sims(interventions):
    ''' Make simulations with paticular interventions '''
    simlist = sc.autolist()
    for intv in interventions:
        pars = fp.pars(interventions=intv, **p)
        simlist += fp.Sim(pars=pars)
    return simlist


def test_update_methods_eff():
    """
    Checks that fp.update_methods() properly updates sim.pars efficacies
    """
    low_eff = dict(dist='uniform', par1=0.80, par2=0.90)
    high_eff = dict(dist='uniform', par1=0.91, par2=0.95)

    scen_low_eff = dict(eff={'Other modern':low_eff})
    scen_high_eff = dict(eff={'Other modern':high_eff})

    low_eff = fp.update_methods(int_year, scen_low_eff)
    high_eff = fp.update_methods(int_year, scen_high_eff)

    simlist = make_sims([low_eff, high_eff])
    msim = fp.MultiSim(sims=simlist)
    msim.run(serial=serial)

    low_eff_post_sim = msim.sims[0].pars['method_efficacy'][9]
    high_eff_post_sim = msim.sims[1].pars['method_efficacy'][9]

    msg = f"Method efficacy after updating to about .93 is {high_eff_post_sim} and after updating to about 0.85 is actually {low_eff_post_sim}"
    assert high_eff_post_sim > low_eff_post_sim, msg
    return msim


def test_update_methods_probs():
    """
    Checks that fp.update_methods() function properly updates sim.pars for
    both the selected age keys, and the type (methods or postpartum_methods) of
    transition matrix
    """

    target_prob1 = 0.2 # Specify the target contraceptive probability
    target_prob2 = 0.8

    scen_no_keys = sc.objdict(
        probs = [
            dict(
                source = 'None', # Source method, 'all' for all methods
                dest   = 'Other modern', # Destination
                factor = None, # Factor by which to multiply existing probability
                value  = 0.2 # Alternatively, specify the absolute probability of switching to this method
            ),
        ]
    )

    scen_keys = sc.objdict(
        probs = [
            dict(
                source = 'None', # Source method, 'all' for all methods
                dest   = 'Other modern', # Destination
                value  = target_prob1, # Alternatively, specify the absolute probability of switching to this method
                keys   = ['<18','18-20'], # Which age keys to modify -- if not specified, all
            ),
            dict(
                source = 'Other modern', # Source method, 'all' for all methods
                dest   = 'None', # Destination
                value  = target_prob2, # Alternatively, specify the absolute probability of switching to this method
                keys   = ['<18','18-20'], # Which age keys to modify -- if not specified, all
            )
        ]
    )

    # Make interventions
    uptake_no_keys_methods = fp.update_methods(int_year, scen_no_keys, matrix='probs_matrix') # Create intervention
    uptake_keys_methods    = fp.update_methods(int_year, scen_keys,    matrix='probs_matrix') # Create intervention
    uptake_no_keys_pp      = fp.update_methods(int_year, scen_no_keys, matrix='probs_matrix_1-6') # Create intervention
    uptake_keys_pp         = fp.update_methods(int_year, scen_keys,    matrix='probs_matrix_1-6') # Create intervention

    # Make and runs ims
    simlist = make_sims([uptake_no_keys_methods, uptake_keys_methods, uptake_no_keys_pp, uptake_keys_pp])
    msim = fp.MultiSim(sims=simlist)
    msim.run(serial=serial)

    # Tests
    m0 = msim.sims[0].pars['methods']
    m1 = msim.sims[1].pars['methods']
    m2 = msim.sims[2].pars['methods_postpartum']
    m3 = msim.sims[3].pars['methods_postpartum']
    none  = m0['map']['None']
    other = m0['map']['Other modern']

    assert m0['probs_matrix']['21-25'][none][other] != m1['probs_matrix']['21-25'][none][other], "update_methods did not change contraceptive matrix for key 21-25"
    assert m0['probs_matrix']['21-25'][none][other] == target_prob1, f"update_methods did not change contraceptive matrix 21-25 to spcified {target_prob1}"
    assert m1['probs_matrix']['<18'][none][other]   == target_prob1, f"update_methods did not change contraceptive matrix <25 to spcified {target_prob1}"

    assert m2['probs_matrix_1-6']['21-25'][none][other] != m3['probs_matrix_1-6']['21-25'][none][other], "update_methods did not change postpartum contraceptive matrix for key 21-25"
    assert m2['probs_matrix_1-6']['21-25'][none][other] == target_prob1, f"update_methods did not change postpartum contraceptive matrix for 21-25 to specified {target_prob1}"
    assert m3['probs_matrix_1-6']['<18'][none][other]   == target_prob1, f"update_methods did not change postpartum contraceptive matrix for <18 to specified {target_prob1}"

    assert m3['probs_matrix_1-6']['<18'][other][none]   == target_prob2, "After updating method switching postpartum for <18 for None to {target_prob2}, value didn't change"
    assert m3['probs_matrix_1-6']['21-25'][other][none] != target_prob2, "After updating method postpartum for 21-25, value is still {target_prob2}"

    return msim


def test_scenarios(do_plot=do_plot):
    ''' Test the actual Scenarios object '''

    # Increased uptake high efficacy
    uptake_scen1 = sc.objdict(
        label='Increased modern',
        eff = {'Other modern':0.994}, # Co-opt an unused method and simulate a medium-efficacy method
        probs = [ # Specify by value
            dict(
                source = 'None', # Source method, 'all' for all methods
                dest   = 'Other modern', # Destination
                value  = 0.2, # Alternatively, specify the absolute probability of switching to this method
                keys   = ['>25'], # Which age keys to modify -- if not specified, all
            ),
        ]
    )

    uptake_scen2 = sc.objdict(
        label = 'Increased injectables',
        eff = {'Injectables': 0.95},
        probs = [
            # Reduce switching from injectables
            dict( # Specify by factor
                source = 'Injectables',  # Source method, 'all' for all methods
                dest   = 'None',  # Destination
                factor = 0.0,  # Factor by which to multiply existing probability
                keys   = ['<18', '18-20'],  # Which age keys to modify -- if not specified, all
            ),
            # Increase switching to injectables
            dict(
                source = 'None', # Source method, 'all' for all methods
                dest   = 'Injectables', # Destination
                factor = 5, # Factor by which to multiply existing probability
                keys   = ['>25'], # Which age keys to modify -- if not specified, all
            ),
        ]
    )

    uptake_scen3 = [uptake_scen1, uptake_scen2]


    #%% Create sims
    scens = fp.Scenarios(pars=p, repeats=2, scen_year=int_year)
    scens.add_scen(label='Baseline')
    scens.add_scen(uptake_scen1)
    scens.add_scen(uptake_scen2)
    scens.add_scen(uptake_scen3, label='Increased modern + increased injectables')

    # Run scenarios
    scens.run(serial=serial)

    # Ensure that everything is unique
    tfr = scens.results.df.tfr
    assert len(tfr) == len(tfr.unique()), 'Number of unique TFR values is less than the number of sims, could be unlucky or a bug'

    # Plot and print results
    if do_plot:
        scens.plot_sims()
        scens.plot_scens()

    return scens


if __name__ == '__main__':

    # run test suite
    with sc.timer():
        msim1 = test_update_methods_eff()
        msim2 = test_update_methods_probs()
        scens = test_scenarios()
