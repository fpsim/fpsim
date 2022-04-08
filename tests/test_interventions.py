"""
Run tests on the calibration object.
"""

import sciris as sc
import fpsim as fp
import fp_analyses as fa

do_plot = 0

def make_sim(n=500, **kwargs):
    '''
    Define a default simulation for testing the baseline.
    '''
    pars = fa.senegal_parameters.make_pars()
    pars['n'] = n
    pars['verbose'] = 0.1
    pars.update(kwargs)
    sim = fp.Sim(pars=pars)

    return sim


def test_interventions():
    ''' Test interventions '''
    sc.heading('Testing interventions...')

    def test_interv(sim):
        if sim.i == 100:
            print(f'Success on day {sim.t}/{sim.y}')

    pars = dict(
        interventions = [test_interv],
    )

    sim = make_sim(**pars)
    sim.run()

    return sim


def test_analyzers():
    ''' Test analyzers '''
    sc.heading('Testing analyzers...')

    pars = dict(
        analyzers = [fp.snapshot(timesteps=[100, 200])],
    )

    sim = make_sim(**pars)
    sim.run()

    return sim

def test_update_methods_eff():
    """
    Checks that fp.update_methods() properly updates sim.pars efficacies
    """
    low_eff = dict(dist='uniform', par1=0.80, par2=0.90)
    high_eff = dict(dist='uniform', par1=0.91, par2=0.95)

    pars_high_eff = fa.senegal_parameters.make_pars()
    pars_low_eff = fa.senegal_parameters.make_pars()

    scen_low_eff = sc.objdict(
        eff = {'Other modern':low_eff},
    )

    scen_high_eff = sc.objdict(
        eff = {'Other modern':high_eff}
    )

    low_eff = fp.update_methods(2005, scen_low_eff) 
    high_eff = fp.update_methods(2005, scen_high_eff)

    interventions = [low_eff, high_eff]

    simlist = []
    for index, pars in enumerate([pars_high_eff, pars_low_eff]):
        pars['n'] = 500
        pars.update(interventions=interventions[index])
        simlist.append(fp.Sim(pars=pars))

    msim = fp.MultiSim(sims=simlist)
    msim.run()

    low_eff_post_sim = msim.sims[0].pars['method_efficacy'][9]
    high_eff_post_sim = msim.sims[1].pars['method_efficacy'][9]

    assert high_eff_post_sim > low_eff_post_sim

def test_update_methods_probs():
    """
    Checks that fp.update_methods() function properly updates sim.pars for
    both the selected age keys, and the type (methods or postpartum_methods) of
    transition matrix
    """
    pars_no_keys_methods = fa.senegal_parameters.make_pars()
    pars_keys_methods = fa.senegal_parameters.make_pars()
    pars_no_keys_pp = fa.senegal_parameters.make_pars()
    pars_keys_pp = fa.senegal_parameters.make_pars()

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
                factor = None, # Factor by which to multiply existing probability
                value  = 0.2, # Alternatively, specify the absolute probability of switching to this method
                keys   = ['<18','18-20'], # Which age keys to modify -- if not specified, all
            ),
            dict(
                source = 'Other modern', # Source method, 'all' for all methods
                dest   = 'None', # Destination
                factor = None, # Factor by which to multiply existing probability
                value  = 0.8, # Alternatively, specify the absolute probability of switching to this method
                keys   = ['<18','18-20'], # Which age keys to modify -- if not specified, all
            )
        ]
    )


    uptake_no_keys_methods = fp.update_methods(2005, scen_no_keys, matrix='probs_matrix') # Create intervention
    uptake_keys_methods = fp.update_methods(2005, scen_keys, matrix='probs_matrix') # Create intervention
    uptake_no_keys_pp = fp.update_methods(2005, scen_no_keys, matrix='probs_matrix_1-6') # Create intervention
    uptake_keys_pp = fp.update_methods(2005, scen_keys, matrix='probs_matrix_1-6') # Create intervention

    interventions = [uptake_no_keys_methods, uptake_keys_methods, uptake_no_keys_pp, uptake_keys_pp]

    simlist = []
    for index, pars in enumerate([pars_no_keys_methods, pars_keys_methods, pars_no_keys_pp, pars_keys_pp]):
        pars['n'] = 500
        pars.update(interventions=interventions[index])
        simlist.append(fp.Sim(pars=pars))
    
    msim = fp.MultiSim(sims=simlist)
    msim.run()

    none_index = msim.sims[0].pars['methods']['map']['None']
    other_modern_index = msim.sims[0].pars['methods']['map']['Other modern']


    assert msim.sims[0].pars['methods']['probs_matrix']['21-25'][none_index][other_modern_index] != msim.sims[1].pars['methods']['probs_matrix']['21-25'][none_index][other_modern_index]
    assert msim.sims[0].pars['methods']['probs_matrix']['21-25'][none_index][other_modern_index] == 0.2
    assert msim.sims[1].pars['methods']['probs_matrix']['<18'][none_index][other_modern_index] == 0.2

    assert msim.sims[2].pars['methods_postpartum']['probs_matrix_1-6']['21-25'][none_index][other_modern_index] != msim.sims[3].pars['methods_postpartum']['probs_matrix_1-6']['21-25'][none_index][other_modern_index]
    assert msim.sims[2].pars['methods_postpartum']['probs_matrix_1-6']['21-25'][none_index][other_modern_index] == 0.2
    assert msim.sims[3].pars['methods_postpartum']['probs_matrix_1-6']['<18'][none_index][other_modern_index] == 0.2

    assert msim.sims[3].pars['methods_postpartum']['probs_matrix_1-6']['<18'][other_modern_index][none_index] == 0.8
    assert msim.sims[3].pars['methods_postpartum']['probs_matrix_1-6']['21-25'][other_modern_index][none_index] != 0.8

if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    T = sc.tic()

    sim1 = test_interventions()
    sim2 = test_analyzers()
    test_update_methods_probs()
    test_update_methods_eff()

    print('\n'*2)
    sc.toc(T)
    print('Done.')
