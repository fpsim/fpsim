"""
Run tests on the interventions.
"""

import sciris as sc
import pylab as pl
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
    """ Test defining an intervention as a function """
    sc.heading('Testing intervention can be defined as a function...')

    def test_interv(sim):
        if sim.ti == 100:
            print(f'Success on day {sim.ti}')
            sim.intervention_applied = True

    sim = make_sim(interventions=test_interv)
    sim.run()
    assert sim.intervention_applied
    print(f'✓ (functions intervention ok)')

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


def test_change_people_state():
    """ Testing that change_people_state() modifies sim results in expected ways """
    sc.heading('Testing change_people_state()...')

    par_kwargs = dict(n_agents=500, start_year=2000, end_year=2020, seed=1, verbose=1)
    pars = fp.pars(location='kenya', **par_kwargs)
    ms = fp.SimpleChoice(location='kenya')
    sim_kwargs = dict(contraception_module=ms)

    # Change ever user
    prior_use_lift = fp.change_people_state('ever_used_contra', years=2019, new_val=True, eligibility=np.arange(500), prop=1, annual=False)
    prior_use_gone = fp.change_people_state('ever_used_contra', years=2020, new_val=False, eligibility=np.arange(500), prop=1, annual=False)

    # Make and run sim
    s0 = fp.Sim(pars, **sim_kwargs, label="Baseline")
    s1 = fp.Sim(pars, **sim_kwargs, interventions=prior_use_lift, label="All prior_use set to True")
    s2 = fp.Sim(pars, **sim_kwargs, interventions=prior_use_gone, label="Prior use removed from 500 people")
    msim = fp.parallel(s0, s1, s2)
    s0, s1, s2 = msim.sims

    # Test people state change
    s0_used_contra = np.sum(s0.people['ever_used_contra'])
    s1_used_contra = np.sum(s1.people['ever_used_contra'])
    s2_500_used_contra = np.sum(s2.people['ever_used_contra'][0:500])

    # print(f"Checking change_state CPR trends ... ")
    # assert s1_used_contra > s0_used_contra, f'Increasing prior use should increase the number of people with who have used contraception, but {s1_used_contra} is not greater than the baseline of {s0_used_contra}'
    # assert s2_500_used_contra == 0, f'Changing people state should set prior use to False for the first 500 agents, but {s2_500_used_contra} is not 0'
    # print(f"✓ ({s1_used_contra} > {s0_used_contra})")

    # Check user input validation
    with pytest.raises(ValueError):  # Check invalid parameter
        make_sim(interventions=fp.change_people_state('not_a_parameter', new_val=True)).run()
    with pytest.raises(ValueError):  # Check bad value
        make_sim(interventions=fp.change_people_state('ever_used_contra', new_val=None)).run()
    with pytest.raises(ValueError):  # Check too late end year
        make_sim(interventions=fp.change_people_state('ever_used_contra', years=2120, new_val=True)).run()
    with pytest.raises(ValueError):  # Check invalid year type
        make_sim(interventions=fp.change_people_state('ever_used_contra', years=None, new_val=True)).run()
    with pytest.raises(ValueError):  # Check invalid eligible
        make_sim(interventions=fp.change_people_state('ever_used_contra', years=2005, new_val=True, eligibility="")).run()

    # Check with plot
    if do_plot:
        import pylab as pl
        t = s0.results['t']
        y0 = s0.results['ever_used_contra']
        y1 = s1.results['ever_used_contra']
        y2 = s2.results['ever_used_contra']
        pl.figure()
        pl.plot(t, y0, label='Baseline')
        pl.plot(t, y1, label='Higher prior use')
        pl.plot(t, y2, label='Stop prior use')
        pl.legend()
        pl.show()

    return s0, s1, s2


def test_education():
    """ Testing that increasing education has expected effects """
    par_kwargs = dict(n_agents=500, start_year=2000, end_year=2020, seed=1, verbose=1)
    pars = fp.pars(location='kenya', **par_kwargs)

    def select_undereducated(sim):
        """ Select women who want education but have attained less than their goals """
        is_eligible = ((sim.people.is_female) &
                       (sim.people.alive)     &
                       # (sim.people.edu_attainment < sim.people.edu_objective) &
                       (sim.people.edu_objective > 0))
        return is_eligible

    edu = fp.Education()
    s0 = fp.Sim(pars=pars, education_module=edu, label='Baseline')

    change_education = fp.change_people_state(
                            'edu_attainment',
                            eligibility=select_undereducated,
                            years=2010.0,
                            new_val=15,  # Give all selected women 15 years of education
                        )
    edu = fp.Education()
    s1 = fp.Sim(pars=pars,
                education_module=edu,
                interventions=change_education,
                label='Increased education')

    s0.run()
    s1.run()

    pl.plot(s0.results.t, s0.results.edu_attainment, label=s0.label)
    pl.plot(s1.results.t, s1.results.edu_attainment, label=s1.label)
    pl.xlim([2005, 2012])
    pl.ylabel('Average years of education among women ')
    pl.xlabel('Year')
    pl.legend()
    pl.show()

    return s0, s1


if __name__ == '__main__':
    # s0 = test_intervention_fn()
    # s1 = test_change_par()
    # s3 = test_plot()
    # s4, s5, s6 = test_change_people_state()
    s7 = test_education()
    # s7, s8 = test_education()

    print('Done.')


