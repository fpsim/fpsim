"""
Run tests on the interventions.
"""

import sciris as sc
import fpsim as fp
import pytest

serial   = 0 # Whether to run in serial (for debugging)
do_plot  = 1 # Whether to do plotting in interactive mode
sc.options(backend='agg') # Turn off interactive plots


def make_sim(**kwargs):
    '''
    Define a default simulation for testing the baseline.
    '''
    sim = fp.Sim(location='test', **kwargs)
    return sim


def test_intervention_fn():
    ''' Test interventions '''
    sc.heading('Testing interventions...')

    def test_interv(sim):
        if sim.i == 100:
            print(f'Success on day {sim.t}/{sim.y}')
            sim.intervention_applied = True

    sim = make_sim(interventions=test_interv)
    sim.run()
    assert sim.intervention_applied

    return sim


def test_change_par():
    ''' Testing that change_par() modifies sim results in expected ways '''
    sc.heading('Testing change_par()...')

    # Define exposure test
    verbose = True
    year = 2002
    ec = 0.01
    cp1 = fp.change_par(par='exposure_factor', years=year, vals=ec, verbose=verbose) # Reduce exposure factor
    s0 = make_sim(label='Baseline')
    s1 = make_sim(interventions=cp1, label='High exposure')

    # Define MCPR growth test
    sim_start = 2015
    mcpr_y1   = 2020 # When the MCPR projection starts
    mcpr_y2   = 2030 # Max MCPR
    mcpr_y3   = 2038 # MCPR should've declined
    sim_end   = 2040 # Reset at this point
    cp2 = fp.change_par('mcpr_growth_rate', vals={mcpr_y1:0.10, mcpr_y2:-0.10, mcpr_y3:'reset'}, verbose=verbose) # Set to large positive then negative growth
    s2 = make_sim(interventions=cp2, start_year=sim_start, end_year=sim_end, label='Changing MCPR')

    # Run
    m = fp.parallel(s0, s1, s2, serial=serial, compute_stats=False)
    s0, s1, s2 = m.sims[:] # Replace with run versions

    # Test exposure factor change
    base_births = s0.results['births'].sum()
    cp_births   = s1.results['births'].sum()
    assert s1['exposure_factor'] == ec, f'change_pars() did not change exposure factor to {ec}'
    assert cp_births < base_births, f'Reducing exposure factor should reduce births, but {cp_births} is not less than the baseline of {base_births}'

    # Test MCPR growth
    r = s2.results
    ind_y1 = sc.findnearest(r.t, mcpr_y1)
    ind_y2 = sc.findnearest(r.t, mcpr_y2)
    ind_y3 = sc.findnearest(r.t, mcpr_y3)
    assert ind_y1 != ind_y2 != ind_y3, f'Sim indices for years years {mcpr_y1}, {mcpr_y2}, {sim_end} should be different'
    assert r.mcpr[ind_y1] < r.mcpr[ind_y2], f'MCPR did not grow from {mcpr_y1} to {mcpr_y2}'
    assert r.mcpr[ind_y3] < r.mcpr[ind_y2], f'MCPR did not shrink from {mcpr_y2} to {sim_end}'
    assert s2['mcpr_growth_rate'] == s0['mcpr_growth_rate'], 'MCPR growth rate did not reset correctly'

    # Check user input validation
    with pytest.raises(ValueError): # Check that length of years and values match
        fp.change_par(par='test', years=[2002],vals=[1,2])
    with pytest.raises(ValueError): # Check invalid parameter
        make_sim(interventions=fp.change_par('not_a_parameter')).run()
    with pytest.raises(ValueError): # Check too early start year
        make_sim(interventions=fp.change_par('exposure_factor', years=1920, vals=1)).run()
    with pytest.raises(ValueError): # Check too late end year
        make_sim(interventions=fp.change_par('exposure_factor', years=2120, vals=1)).run()

    if do_plot:
        m.plot()

    return m


def test_plot():
    sc.heading('Testing intervention plotting...')

    cp = fp.change_par(par='exposure_factor', years=2002, vals=2.0) # Reduce exposure factor
    um1 = fp.update_methods(year=2005, eff={'Injectables':1.0})
    um2 = fp.update_methods(year=2008, probs=dict(source='None', dest='Injectables', value=0.5))
    sim = make_sim(interventions=[cp, um1, um2]).run()

    if do_plot:
        sim.plot()
    return sim

if __name__ == '__main__':
    sc.options(backend=None) # Turn on interactive plots
    with sc.timer():
        isim   = test_intervention_fn()
        cpmsim = test_change_par()
        sim  = test_plot()
