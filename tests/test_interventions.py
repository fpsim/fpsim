"""
Run tests on the interventions.
"""

import sciris as sc
import pylab as pl
import fpsim as fp
import numpy as np

serial   = 0 # Whether to run in serial (for debugging)
do_plot  = 1 # Whether to do plotting in interactive mode
# sc.options(backend='agg') # Turn off interactive plots


def make_sim(**kwargs):
    '''
    Define a default simulation for testing the baseline.
    '''
    sim = fp.Sim(location='test', **kwargs)
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
    assert cp2_births <= base_births, f'Reducing exposure factor temporarily should reduce births, but {cp2_births} is not less than the baseline of {base_births}'

    return m


def test_plot():
    sc.heading('Testing intervention plotting...')

    cp = fp.change_par(par='exposure_factor', years=2002, vals=2.0) # Reduce exposure factor
    um1 = fp.update_methods(year=2005, eff={'Injectables': 1.0})
    um2 = fp.update_methods(year=2008, p_use=0.5)
    um3 = fp.update_methods(year=2010, method_mix=[0.9, 0.1, 0, 0, 0, 0, 0, 0, 0])
    sim = make_sim(contraception_module=fp.RandomChoice(), interventions=[cp, um1, um2, um3]).run()

    return sim


def test_change_people_state():
    """ Testing that change_people_state() modifies sim results in expected ways """
    sc.heading('Testing change_people_state()...')

    par_kwargs = dict(n_agents=500, start_year=2000, end_year=2020, seed=1, verbose=-1)
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

    print(f"Checking change_state CPR trends ... ")
    assert s1_used_contra > s0_used_contra, f'Increasing prior use should increase the number of people with who have used contraception, but {s1_used_contra} is not greater than the baseline of {s0_used_contra}'
    assert s2_500_used_contra == 0, f'Changing people state should set prior use to False for the first 500 agents, but {s2_500_used_contra} is not 0'
    print(f"✓ ({s1_used_contra} > {s0_used_contra})")

    return s0, s1, s2


if __name__ == '__main__':
    s1 = test_change_par()
    s3 = test_plot()
    s4, s5, s6 = test_change_people_state()

    print('Done.')


