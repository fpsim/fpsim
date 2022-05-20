"""
Run tests on the Scenarios class.
"""

import sciris as sc
import fpsim as fp
import pytest

# Global settings
int_year = 2002 # Year to start the interventions
serial   = 1 # Whether to run in serial (for debugging)
do_plot  = 1 # Whether to do plotting in interactive mode
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
    low_eff  = {method:dict(dist='uniform', par1=0.80, par2=0.90)}
    high_eff = {method:dict(dist='uniform', par1=0.91, par2=0.95)}

    low_eff = fp.update_methods(year=int_year, eff=low_eff)
    high_eff = fp.update_methods(year=int_year, eff=high_eff)

    simlist = make_sims([low_eff, high_eff])
    msim = fp.MultiSim(sims=simlist)
    msim.run(serial=serial)

    low_eff_post_sim = msim.sims[0]['method_efficacy'][9]
    high_eff_post_sim = msim.sims[1]['method_efficacy'][9]

    msg = f"Method efficacy after updating to about .93 is {high_eff_post_sim} and after updating to about 0.85 is actually {low_eff_post_sim}"
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
            ages   = ['<18','18-20'], # Which age keys to modify -- if not specified, all
        ),
        dict(
            source = 'Other modern', # Source method, 'all' for all methods
            dest   = 'None', # Destination
            value  = target_prob2, # Alternatively, specify the absolute probability of switching to this method
            ages   = ['<18','18-20'], # Which age keys to modify -- if not specified, all
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

    assert m0['annual']['21-25'][none][other] != m1['annual']['21-25'][none][other], "update_methods did not change contraceptive matrix for key 21-25"
    assert m0['annual']['21-25'][none][other] == target_prob1, f"update_methods did not change contraceptive matrix 21-25 to specified {target_prob1}"
    assert m1['annual']['<18'][none][other]   == target_prob1, f"update_methods did not change contraceptive matrix <25 to specified {target_prob1}"

    assert m2['pp0to1']['21-25'][other] != m3['pp0to1']['21-25'][other], "update_methods did not change postpartum contraceptive matrix for key 21-25"
    assert m2['pp0to1']['21-25'][other] == target_prob1, f"update_methods did not change postpartum contraceptive matrix for 21-25 to specified {target_prob1}"
    assert m3['pp1to6']['<18'][none][other] == target_prob1, f"update_methods did not change postpartum contraceptive matrix for <18 to specified {target_prob1}"

    assert m3['pp1to6']['<18'][other][none]   == target_prob2, "After updating method switching postpartum for <18 for None to {target_prob2}, value didn't change"
    assert m3['pp1to6']['21-25'][other][none] != target_prob2, "After updating method postpartum for 21-25, value is still {target_prob2}"

    return msim


def test_scenarios(do_plot=do_plot):
    ''' Test the actual Scenarios object '''

    sc.heading('Testing scenarios...')

    # Increased uptake high efficacy
    uptake_scen1 = fp.make_scen(
        label = 'Increased modern',
        eff = {'Other modern':0.994}, # Co-opt an unused method and simulate a medium-efficacy method
        probs = dict( # Specify by value
            source = 'None', # Source method, 'all' for all methods
            dest   = 'Other modern', # Destination
            value  = 0.2, # Alternatively, specify the absolute probability of switching to this method
            ages   = ['>25'], # Which age keys to modify -- if not specified, all
        ),
    )

    uptake_scen2 = fp.make_scen(
        label = 'Increased injectables',
        eff = {'Injectables': 0.95},
        probs = [
            # Reduce switching from injectables
            dict( # Specify by factor
                source = 'Injectables',  # Source method, 'all' for all methods
                dest   = 'None',  # Destination
                factor = 0.0,  # Factor by which to multiply existing probability
                ages   = ['<18', '18-20'],  # Which age keys to modify -- if not specified, all
            ),
            # Increase switching to injectables
            dict(
                source = 'None', # Source method, 'all' for all methods
                dest   = 'Injectables', # Destination
                factor = 5, # Factor by which to multiply existing probability
                ages   = ['all'], # Which age keys to modify -- if not specified, all
            ),
        ]
    )

    uptake_scen3 = uptake_scen1 + uptake_scen2

    #%% Create sims
    scens = fp.Scenarios(location='test', n=200, repeats=2, scen_year=int_year)
    scens.add_scen(label='Baseline')
    scens.add_scen(uptake_scen1)
    scens.add_scen(uptake_scen2)
    scens.add_scen(uptake_scen3, label='Increased modern + increased injectables')

    # Run scenarios
    scens.run(serial=serial)

    # Ensure that everything is unique
    df = scens.results.df
    dfhash = df.births + df.fails + df.popsize + df.tfr # Construct a "hash" by summing column values -- at least one should differ
    assert len(dfhash) == len(dfhash.unique()), 'Number of unique output values is less than the number of sims, could be unlucky or a bug'

    # Check we can't add invalid scenarios
    invalid_scen1 = dict(invalid_key='Should fail')
    invalid_scen2 = dict(probs=dict(invalid_key='Also should fail'))
    with pytest.raises(ValueError):
        invalid_scens1 = fp.Scenarios(location='test')
        invalid_scens1.add_scen(invalid_scen1)
        invalid_scens1.run()
    with pytest.raises(ValueError):
        invalid_scens2 = fp.Scenarios(location='test')
        invalid_scens2.add_scen(invalid_scen2)
        invalid_scens2.run()

    # Plot and print results
    if do_plot:
        scens.plot_sims()
        scens.plot_scens()
        scens.plot_cpr()

    return scens


def test_make_scens():
    '''
    Test that the user-friendly scenarios API works
    '''

    sc.heading('Testing make_scens...')

    year   = 2002
    method = 'Injectables'

    # Create basic scenarios
    s = sc.objdict()
    s.eff   = fp.make_scen(year=year, eff={'Injectables':0.99}) # Basic efficacy scenario
    s.prob1 = fp.make_scen(year=year, source='None', dest='Injectables', factor=2) # Double rate of injectables initiation
    s.prob2 = fp.make_scen(year=year, method='Injectables', init_factor=2) # Double rate of injectables initiation -- alternate approach
    s.par   = fp.make_scen(par='exposure_correction', years=2005, vals=0.5) # Parameter scenario: halve exposure

    # More complex example: change condoms to injectables transition probability for 18-25 postpartum women
    s.complex = fp.make_scen(year=year, source='Condoms', dest='Injectables', value=0.5, ages='18-20', matrix='pp1to6')

    # Custom scenario
    def update_sim(sim): sim.updated = True
    s.custom = fp.make_scen(interventions=update_sim)

    # Combining multiple scenarios: increase injectables initiation and reduce exposure correction
    s.multi = fp.make_scen(
        dict(year=year, method=method, init_factor=2),
        dict(par='exposure_correction', years=2010, vals=0.5)
    )

    # Scenario addition
    s.sum = s.eff + s.prob1

    # More probability matrix options
    s.inj1 = fp.make_scen(year=year, method=method, init_factor=5, matrix='annual', ages=None)
    s.inj2 = fp.make_scen(year=year, method=method, discont_factor=0, matrix='annual', ages=':')
    s.inj3 = fp.make_scen(year=year, method=method, init_value=0.2, matrix='pp1to6', ages=None)
    s.inj4 = fp.make_scen(year=year, method=method, discont_value=0, matrix='pp1to6', ages=':')
    s.inj5 = fp.make_scen(year=year, source='None', dest='Injectables', factor=0.2, ages=['<18', '>25'])

    # Run scenarios
    scens = fp.Scenarios(location='test', n=100, repeats=1, scens=s.values())
    scens.run(serial=serial)

    return scens


if __name__ == '__main__':

    sc.options(backend=None) # Turn on interactive plots
    with sc.timer():
        # msim1  = test_update_methods_eff()
        # msim2  = test_update_methods_probs()
        scens1 = test_scenarios()
        scens2 = test_make_scens()
