"""
Run tests on the Scenarios class.
"""

import sciris as sc
import fpsim as fp

# Global settings
n          = 100 # Population size
int_year   = 2005 # Year to start the interventions
start_year = 2000 # Start year of sims
end_year   = 2010 # End year of sims
serial     = True # Whether to run in serial


def make_sims(interventions):
    ''' Make simulations with paticular interventions '''
    simlist = sc.autolist()
    for intv in interventions:
        pars = fp.pars(n=n, start_year=start_year, end_year=end_year, interventions=intv)
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
    uptake_keys_pp         = fp.update_methods(int_year, scen_keys,     matrix='probs_matrix_1-6') # Create intervention

    # Make and runs ims
    simlist = make_sims([uptake_no_keys_methods, uptake_keys_methods, uptake_no_keys_pp, uptake_keys_pp])
    msim = fp.MultiSim(sims=simlist)
    msim.run(serial=serial)

    # Tests
    m0 = msim.sims[0].pars['methods']
    m1 = msim.sims[1].pars['methods']
    m2 = msim.sims[2].pars['methods_postpartum']
    m3 = msim.sims[3].pars['methods_postpartum']
    i_no  = m0['map']['None']
    i_oth = m0['map']['Other modern']

    assert m0['probs_matrix']['21-25'][i_no][i_oth] != m1['probs_matrix']['21-25'][i_no][i_oth], "update_methods did not change contraceptive matrix for key 21-25"
    assert m0['probs_matrix']['21-25'][i_no][i_oth] == target_prob1, "update_methods did not change contraceptive matrix 21-25 to spcified 0.2"
    assert m1['probs_matrix']['<18'][i_no][i_oth]   == target_prob1, "update_methods did not change contraceptive matrix <25 to spcified 0.2"

    assert m2['probs_matrix_1-6']['21-25'][i_no][i_oth] != m3['probs_matrix_1-6']['21-25'][i_no][i_oth], "update_methods did not change postpartum contraceptive matrix for key 21-25"
    assert m2['probs_matrix_1-6']['21-25'][i_no][i_oth] == target_prob1, "update_methods did not change postpartum contraceptive matrix for 21-25 to specified 0.2"
    assert m3['probs_matrix_1-6']['<18'][i_no][i_oth]   == target_prob1, "update_methods did not change postpartum contraceptive matrix for <18 to specified 0.2"

    assert m3['probs_matrix_1-6']['<18'][i_oth][i_no] != target_prob2, "After updating method switching postpartum for <18 for None to 0.8, value didn't change"
    assert m3['probs_matrix_1-6']['21-25'][i_oth][i_no] != target_prob2, "After updating method postpartum for 21-25, value is still 0.8"

    return msim


if __name__ == '__main__':

    # run test suite
    msim1 = test_update_methods_eff()
    msim2 = test_update_methods_probs()
