"""
Run tests on the Scenarios class.
"""

import numpy as np
import sciris as sc
import fpsim as fp
import pytest
from fpsim import defaults as fpd

# Global settings
int_year = 2002 # Year to start the interventions
serial   = 0 # Whether to run in serial (for debugging)
do_plot  = 0 # Whether to do plotting in interactive mode
do_plot_as = 0 # Whether or not to plot all age-specific channels
default_ages = list(fpd.method_age_map.keys())
sc.options(backend='agg') # Turn off interactive plots


def make_sims(interventions):
    ''' Make simulations with paticular interventions '''
    simlist = sc.autolist()
    for intv in interventions:
        pars = fp.pars('test', interventions=intv)
        simlist += fp.Sim(pars=pars)
    return simlist


def test_update_methods_eff():
    """
    Checks that fp.update_methods() properly updates sim.pars efficacies
    """

    sc.heading('Testing updating method efficacy...')

    method = 'Other modern'
    l = [0.80, 0.90]
    h = [0.90, 1.00]
    low_eff  = {method:dict(dist='uniform', par1=l[0], par2=l[1])}
    high_eff = {method:dict(dist='uniform', par1=h[0], par2=h[1])}

    low_eff  = fp.update_methods(year=int_year, eff=low_eff)
    high_eff = fp.update_methods(year=int_year, eff=high_eff)

    simlist = make_sims([low_eff, high_eff])
    msim = fp.MultiSim(sims=simlist)
    msim.run(serial=serial)

    low_eff_post_sim = msim.sims[0]['methods']['eff'][method]
    high_eff_post_sim = msim.sims[1]['methods']['eff'][method]

    msg = f"Expected efficacy {np.mean(h)} and got {high_eff_post_sim}, which is lower than expected {np.mean(l)} and got {low_eff_post_sim}"
    assert high_eff_post_sim > low_eff_post_sim, msg
    return msim


def test_update_methods_probs():
    """
    Checks that fp.update_methods() function properly updates sim.pars for
    both the selected age keys, and the type (methods or postpartum_methods) of
    transition matrix
    """

    sc.heading('Testing updating method probability...')

    target_prob1 = 0.2 # Specify the target contraceptive probability
    target_prob2 = 0.8

    scen_no_keys = dict(
        source = 'None', # Source method, 'all' for all methods
        dest   = 'Other modern', # Destination
        factor = None, # Factor by which to multiply existing probability
        value  = 0.2 # Alternatively, specify the absolute probability of switching to this method
    )

    scen_keys = [
        dict(
            source = 'None', # Source method, 'all' for all methods
            dest   = 'Other modern', # Destination
            value  = target_prob1, # Alternatively, specify the absolute probability of switching to this method
            ages   = default_ages[0:2], # Which age keys to modify -- if not specified, all
        ),
        dict(
            source = 'Other modern', # Source method, 'all' for all methods
            dest   = 'None', # Destination
            value  = target_prob2, # Alternatively, specify the absolute probability of switching to this method
            ages   = default_ages[0:2], # Which age keys to modify -- if not specified, all
        )
    ]

    # Make interventions
    uptake_no_keys_methods = fp.update_methods(int_year, probs=scen_no_keys, matrix='annual')
    uptake_keys_methods    = fp.update_methods(int_year, probs=scen_keys,    matrix='annual')
    uptake_no_keys_pp      = fp.update_methods(int_year, probs=scen_no_keys, matrix='pp0to1')
    uptake_keys_pp         = fp.update_methods(int_year, probs=scen_keys,    matrix='pp1to6')

    # Make and runs ims
    simlist = make_sims([uptake_no_keys_methods, uptake_keys_methods, uptake_no_keys_pp, uptake_keys_pp])
    msim = fp.MultiSim(sims=simlist)
    msim.run(serial=serial)

    # Tests
    mlist = []
    mmap = msim.sims[0].pars['methods']['map']
    for i in range(4):
        mlist.append(msim.sims[i].pars['methods']['raw'])
    m0, m1, m2, m3 = mlist
    none  = mmap['None']
    other = mmap['Other modern']

    targeted_age = default_ages[0]
    alt_targeted_age = default_ages[1]
    not_target_age = default_ages[-1]
    assert m0['annual'][not_target_age][none][other] != m1['annual'][not_target_age][none][other], f"update_methods did not change contraceptive matrix for key {not_target_age}"
    assert m0['annual'][not_target_age][none][other] == target_prob1, f"update_methods did not change contraceptive matrix {targeted_age} to specified {target_prob1}"
    assert m1['annual'][targeted_age][none][other]   == target_prob1, f"update_methods did not change contraceptive matrix {targeted_age} to specified {target_prob1}"

    assert m2['pp0to1'][not_target_age][other] != m3['pp0to1'][not_target_age][other], f"update_methods did not change postpartum contraceptive matrix for key {not_target_age}"
    assert m2['pp0to1'][not_target_age][other] == target_prob1, f"update_methods did not change postpartum contraceptive matrix for {not_target_age} to specified {target_prob1}"
    assert m3['pp1to6'][alt_targeted_age][none][other] == target_prob1, f"update_methods did not change postpartum contraceptive matrix for {alt_targeted_age} to specified {target_prob1}"

    assert m3['pp1to6'][alt_targeted_age][other][none] == target_prob2, f"After updating method switching postpartum for {alt_targeted_age} for None to {target_prob2}, value didn't change"
    assert m3['pp1to6'][not_target_age][other][none] != target_prob2, f"After updating method postpartum for {not_target_age}, value is still {target_prob2}"

    return msim

def test_scenarios():
    def run_scenario(scen, plot=do_plot, plot_as=do_plot_as):
        '''Runs simple scenario and returns Scenarios object'''
        scens = fp.Scenarios(location='test', scens=scen, start_year=int_year)
        scens.run(serial=serial)
        if plot:
            scens.plot()
            scens.plot(to_plot='method')
            scens.plot_sims()
        if plot_as:
            for plot_type in ['age_specific_tfr', 'age_specific_pregnancies', 'age_specific_imr', 'age_specific_mmr', 'age_specific_stillbirths', 'age_specific_births']:
                scens.plot(to_plot=plot_type)
        return scens

    output = {} # dictionary of scenarios output

    base_scenario = run_scenario(fp.make_scen(label="Baseline"))
    base_sim = base_scenario.msim.sims[0].pars['methods']['raw']
    default_age = default_ages[0]
    alt_default_age = default_ages[1]
    unchanged_age = default_ages[-1]
    none, other = (base_scenario.msim.sims[0].pars['methods']['map']['None'], base_scenario.msim.sims[0].pars['methods']['map']['Other modern'])

    output['Baseline'] = base_scenario

    '''Tests that Scenarios repeats sims corresponding to scenarios added as expected'''
    changed_value = 0.99
    scen = fp.make_scen(label='basic_efficacy_increase_lower', year=int_year, eff={'Injectables': changed_value})
    scens_repeat = fp.Scenarios(location='test', repeats=2, scens=scen, start_year=int_year)
    scens_repeat.run(serial=serial)

    assert len(scens_repeat.msim.sims) == 2, f"Should be {2} sims in scens object but found {len(scens.msim.sims)}"

    efficacy = scens_repeat.msim.sims[0].pars['methods']['eff']['Injectables']
    efficacy_copy = scens_repeat.msim.sims[0].pars['methods']['eff']['Injectables']

    output['Repeated'] = scens_repeat

    for efficacy in [efficacy, efficacy_copy]:
        assert efficacy == changed_value, f"Repeated efficacy scenarios do not match"

    '''Tests that Scenarios changes efficacy from dictionary as expected'''
    scen = fp.make_scen(label='basic_efficacy_increase', year=int_year, eff={'Injectables':changed_value})
    scens_efficacy = run_scenario(scen)

    efficacy = scens_efficacy.msim.sims[0].pars['methods']['eff']['Injectables']
    assert efficacy == changed_value, f"Effiacy of Injectables after efficacy scenario should be {changed_value} but is {efficacy}"

    for matrix in ['annual', 'pp1to6', 'pp0to1']:
        '''Tests that Scenarios handles changes to contraceptive switching as expected'''
        changed_value = 0.2
        increase_annual_uptake = sc.objdict(
            eff = {'Other modern':0.994},
            year = int_year,
            probs = [ 
                dict(
                    source = 'None',
                    dest   = 'Other modern',
                    value  = changed_value,
                    ages   = [default_age],
                    matrix = matrix
                ),
            ]
        )
        
        scen = fp.make_scen(label=f"increase_{matrix}_uptake", spec=increase_annual_uptake)
        output[f'{matrix}_matrix'] = run_scenario(scen)
        scens = output[f'{matrix}_matrix']
        changed_sim = scens.msim.sims[0].pars['methods']['raw']
        
        # Hard check that transition value is changed for methods
        no_change_message = f"Changing {matrix} matrix to {changed_value} for age {default_age} in Scenarios does not yield {changed_value}"
        wrong_targeting_message = f"Changing {matrix} matrix to {changed_value} for age {default_age} changed transition value but did not only target age key {default_age} for matrix {matrix}"
        if matrix != "pp0to1":
            # Expects 2D array for transition matrix
            assert changed_sim[matrix][default_age][none][other] == changed_value, no_change_message
            assert changed_sim[matrix][unchanged_age][none][other] == base_sim[matrix][unchanged_age][none][other], wrong_targeting_message
        else:
            # Expects 1D array representing transitions
            assert changed_sim[matrix][default_age][other] == changed_value, no_change_message
            assert changed_sim[matrix][unchanged_age][other] == base_sim['pp0to1'][unchanged_age][other], wrong_targeting_message

    '''Checks that inputting scenarios as list has same result as adding them separately'''
    changed_value2 = 0.95
    scen1 = fp.make_scen(label='basic_efficacy_increase', year=int_year, eff={'Injectables':changed_value})
    scen2 = fp.make_scen(label='basic_efficacy_increase_lower', year=int_year, eff={'Injectables':changed_value2})
    
    scens_scenario_list = run_scenario([scen1, scen2])

    efficacy1 = scens_scenario_list.msim.sims[0].pars['methods']['eff']['Injectables']
    efficacy2 = scens_scenario_list.msim.sims[1].pars['methods']['eff']['Injectables']

    output['Scenario_list'] = scens_scenario_list

    assert efficacy1 == changed_value, f"Effiacy of Injectables after efficacy scenarios passed in as list should be 0.99 but is {efficacy1}"
    assert efficacy2 == changed_value2, f"Effiacy of Injectables after efficacy scenarios passed in as list should be 0.95 but is {efficacy2}"

    '''Checks that Scenarios with custom method contains sims with appropriate parameters'''
    pars = fp.pars()
    new_method = 'new condoms'
    baseline_methods = len(base_scenario.msim.sims[0].pars['methods']['eff'])
    pars.add_method(name=new_method, eff=0.946)
    scens_custom_method = fp.Scenarios(location='test', pars=pars, scens=[fp.make_scen(label=f"custom_intervention"), fp.make_scen(label=f"custom_intervention2")])
    scens_custom_method.run()
    output['Custom_method'] = scens_custom_method
    for sim in scens_custom_method.msim.sims:
        assert len(sim.pars['methods']['eff']) == baseline_methods + 1, 'Method efficacies dict does not contain entry for {new_method}'
        assert new_method in sim.pars['methods']['map']

    '''Checks that we can't add invalid scenarios'''
    invalid_scen1 = dict(invalid_key='Should fail')
    invalid_scen2 = dict(probs=dict(invalid_key='Also should fail'))

    with pytest.raises(TypeError):
        invalid_scens1 = fp.Scenarios(location='test')
        invalid_scens1.add_scen(invalid_scen1)        

    with pytest.raises(ValueError):
        invalid_scens = fp.Scenarios(location='test')
        invalid_scens.add_scen(invalid_scen2)
        invalid_scens.run()

    '''Checks that scenarios can take varied keyword args'''
    scen = fp.make_scen(label="switch_with_keywords", year=int_year, source='None', dest='Other modern', factor=2, ages=[default_age, alt_default_age], matrix='pp1to6')
    scens_keywords = run_scenario(scen)
    sim_with_keyword_args = scens_keywords.msim.sims[0].pars['methods']['raw']

    switch_age_keywords = sim_with_keyword_args['pp1to6'][default_age][none][other]
    switch_age_default = base_sim['pp1to6'][default_age][none][other]
    switch_alt_age_keywords = sim_with_keyword_args['pp1to6'][alt_default_age][none][other]
    switch_alt_age_default = base_sim['pp1to6'][alt_default_age][none][other]

    output['Keyword_args'] = scens_keywords

    assert switch_age_keywords == 2 * switch_age_default, f"Changing matrix to by 2x in scenarios with keyword arguments should be {2 * switch_age_default} for {default_age} but is {switch_age_keywords}"
    assert switch_alt_age_keywords == 2 * switch_alt_age_default, f"Changing matrix to by 2x in scenarios with keyword arguments should be {2 * switch_alt_age_keywords} for {alt_default_age} but is {switch_alt_age_default}"
    assert sim_with_keyword_args['pp1to6'][unchanged_age][none][other] == base_sim['pp1to6'][unchanged_age][none][other], f"adding scenario to scenarios class with keywords did not sepcifically target {default_age} in pp1to6 matrix"
    
    changed_value = 0
    alt_scen = fp.make_scen(label="switch_with_alt_keywords", year=int_year, method='Other modern', discont_value=changed_value, ages=':', matrix='annual')
    scens_alt_keywords = run_scenario(alt_scen)
    alt_sim_with_keyword_args = scens_alt_keywords.msim.sims[0]['methods']['raw']

    output['Alt_keyword_args'] = scens_alt_keywords

    for age_group in default_ages:
        switch_value = alt_sim_with_keyword_args['annual'][age_group][other][none]
        assert switch_value == changed_value, f"Changing discontinuation value to 0 in scenarios with keyword arguments should be {0} for {age_group} but is {switch_value}"

    '''Checks that Scenarios.results against sim.results for each component sim'''
    scenario_df = base_scenario.results.df
    sim_results = base_scenario.msim.sims[0].results

    def compare_results(scenario_key, sim_key, is_sum=True):
        scenario_val = scenario_df[scenario_key][0]
        if is_sum:
            sim_val = sum(sim_results[sim_key])
        else:
            sim_val = np.mean(sim_results[sim_key])
            
        assert scenario_val == sim_val, f"From sim results {sim_key} is {sim_val} while in scenarios {scenario_key} is {scenario_val}"

    # check sums
    for keys in [("births", "births"), ("fails", "method_failures_over_year"), ("popsize", "pop_size"),
                    ("infant_deaths", "infant_deaths_over_year"), ("maternal_deaths", "maternal_deaths_over_year")]:
        compare_results(keys[0], keys[1])

    # check rates
    for keys in [("tfr", "tfr_rates"), ("mcpr", "mcpr")]:
        compare_results(keys[0], keys[1], is_sum=False) 

if __name__ == '__main__':

    sc.options(backend=None) # Turn on interactive plots
    with sc.timer():
        msim1  = test_update_methods_eff()
        msim2  = test_update_methods_probs()
        scenarios = test_scenarios() # returns a dict with schema {name: Scenarios}
