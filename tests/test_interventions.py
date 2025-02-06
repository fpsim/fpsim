"""
Run tests on the interventions.
"""

import sciris as sc
import fpsim as fp
import numpy as np
import pytest

serial   = 0 # Whether to run in serial (for debugging)
do_plot  = 1 # Whether to do plotting in interactive mode
# sc.options(backend='agg') # Turn off interactive plots


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
        if sim.ti == 100:
            print(f'Success on day {sim.ti}')
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
    cp2 = fp.change_par(par='exposure_factor', years=year+5, vals=['reset'], verbose=verbose)  # Reset exposure factor
    s0 = make_sim(label='Baseline')
    s1 = make_sim(interventions=cp1, label='Low exposure')
    s2 = make_sim(interventions=[cp1, cp2], label='Low exposure, reset')

    # Run
    m = fp.parallel(s0, s1, s2, serial=serial, compute_stats=False)
    s0, s1, s2 = m.sims[:] # Replace with run versions

    # Test exposure factor change
    base_births = s0.results['births'].sum()
    cp1_births   = s1.results['births'].sum()
    cp2_births  = s2.results['births'].sum()
    assert s1['exposure_factor'] == ec, f'change_pars() did not change exposure factor to {ec}'
    assert cp1_births < base_births, f'Reducing exposure factor should reduce births, but {cp1_births} is not less than the baseline of {base_births}'

    assert s2['exposure_factor'] == 1.0, f'Exposure factor should be reset back to 1.0, but it is {s2["exposure_factor"]}'
    assert cp2_births < base_births, f'Reducing exposure factor temporarily should reduce births, but {cp2_births} is not less than the baseline of {base_births}'

    # Check user input validation
    with pytest.raises(ValueError): # Check that length of years and values match
        fp.change_par(par='test', years=[2002],vals=[1,2])
    with pytest.raises(ValueError): # Check invalid parameter
        make_sim(interventions=fp.change_par('not_a_parameter')).run()
    with pytest.raises(ValueError): # Check too early start year
        make_sim(interventions=fp.change_par('exposure_factor', years=1920, vals=1)).run()
    with pytest.raises(ValueError): # Check too late end year
        make_sim(interventions=fp.change_par('exposure_factor', years=2120, vals=1)).run()
    with pytest.raises(ValueError): # Check invalid year type
        make_sim(interventions=fp.change_par('exposure_factor', years=None, vals=-1)).run()

    if do_plot:
        m.plot()

    return m


def test_plot():
    sc.heading('Testing intervention plotting...')

    cp = fp.change_par(par='exposure_factor', years=2002, vals=2.0) # Reduce exposure factor
    um1 = fp.update_methods(year=2005, eff={'Injectables': 1.0})
    um2 = fp.update_methods(year=2008, p_use=0.5)
    um3 = fp.update_methods(year=2010, method_mix=[0.9, 0.1, 0, 0, 0, 0, 0, 0, 0])
    sim = make_sim(interventions=[cp, um1, um2, um3]).run()

    if do_plot:
        sim.plot()
    return sim


def test_change_people_state(emp=False):
    """ Testing that change_people_state() modifies sim results in expected ways """
    sc.heading('Testing change_people_state()...')

    def intv_eligible(sim):
        return ((sim.people.is_female) &
                (sim.people.alive) &
                (sim.people.age >= 15) &
                (sim.people.age < 50) &
                ~sim.people.has_fin_knowl)

    fin_know = fp.change_people_state('has_fin_knowl', years=2019, new_val=True, eligibility=intv_eligible, prop=0.1, annual=True)

    par_kwargs = dict(n_agents=500, start_year=2000, end_year=2020, seed=1, verbose=1)
    pars = fp.pars(location='kenya', **par_kwargs)

    # Create modules
    if not emp:
        ms = fp.SimpleChoice(location='kenya')
        sim_kwargs = dict(contraception_module=ms)
    else:
        ms = fp.EmpoweredChoice(location='kenya')
        emp = fp.Empowerment(location='kenya')
        edu = fp.Education(location='kenya')
        sim_kwargs = dict(contraception_module=ms, empowerment_module=emp, education_module=edu)

    # Make and run sim
    s0 = fp.Sim(pars, **sim_kwargs, label="Baseline")
    s1 = fp.Sim(pars, **sim_kwargs, interventions=fin_know, label="Fin_Knowl")
    s2 = fp.Sim(pars, **sim_kwargs, interventions=fp.change_people_state('has_fin_knowl', years=2019, new_val=False, eligibility=np.arange(500), prop=1, annual=True), label="No_Fin_Knowl 500")
    s0.run()
    s1.run()
    s2.run()

    # Test people state change
    s0_has_fin_knowl = s0['has_fin_knowl'].count(True)
    s1_has_fin_knowl = s1['has_fin_knowl'].count(True)
    s2_500_has_fin_knowl = s2['has_fin_knowl'][0:500].count(True)

    assert s1_has_fin_knowl > s0_has_fin_knowl, f'Changing people state should increase the number of people with financial knowledge, but {s1_has_fin_knowl} is not greater than the baseline of {s0_has_fin_knowl}'
    assert s2_500_has_fin_knowl == 0, f'Changing people state should set the financial knowledge of the first 500 agents to 0, but {s2_500_has_fin_knowl} is not 0'

    # Check user input validation
    with pytest.raises(ValueError):  # Check invalid parameter
        make_sim(interventions=fp.change_people_state('not_a_parameter')).run()
    with pytest.raises(ValueError):  # Check bad value
        make_sim(interventions=fp.change_people_state('has_fin_know', new_val=None)).run()
    with pytest.raises(ValueError):  # Check too late end year
        make_sim(interventions=fp.change_people_state('has_fin_know', years=2120, new_val=True)).run()
    with pytest.raises(ValueError):  # Check invalid year type
        make_sim(interventions=fp.change_people_state('has_fin_know', years=None, new_val=True)).run()
    with pytest.raises(ValueError):  # Check invalid eligible
        make_sim(interventions=fp.change_people_state('has_fin_know', years=2005, new_val=True, eligibility="")).run()





    # Check with plot
    if do_plot:
        import pylab as pl
        t = s0.results['t']
        y0 = s0.results['has_fin_knowl']
        y1 = s1.results['has_fin_knowl']
        pl.figure()
        pl.plot(t, y0, label='Baseline')
        pl.plot(t, y1, label='Improved financial knowledge')
        pl.legend()
        pl.show()

    return s0, s1


if __name__ == '__main__':
    isim   = test_intervention_fn()
    cpmsim = test_change_par()
    sim  = test_plot()
    s0, s1 = test_change_people_state(emp=True)
    s2, s3 = test_change_people_state(emp=False)


