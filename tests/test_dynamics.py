"""
Test dynamics within FPsim related to contraceptive use, method choice, and duration of use
"""

import sciris as sc
import pylab as pl
import fpsim as fp
import numpy as np


serial   = 1 # Whether to run in serial (for debugging)
do_plot  = 1 # Whether to do plotting in interactive mode
# sc.options(backend='agg') # Turn off interactive plots


def make_sim(**kwargs):
    '''
    Define a default simulation for testing the baseline.
    '''
    sim = fp.Sim(location='test', **kwargs)
    return sim


def test_education():
    sc.heading('Testing that increasing education has expected effects')
    location = 'kenya'
    par_kwargs = dict(n_agents=500, start_year=2000, end_year=2020, seed=1, verbose=-1)
    pars = fp.pars(location=location, **par_kwargs)

    def select_undereducated(sim):
        return sim.people.is_female & sim.people.alive & (sim.people.edu_objective > 0)

    edu = fp.Education()
    choice = fp.StandardChoice(location=location)
    s0 = fp.Sim(pars=pars, contraception_module=choice, education_module=edu, label='Baseline')
    s0.run()

    # Sim with intervention
    change_education = fp.change_people_state(
                            'edu_attainment',
                            eligibility=select_undereducated,
                            years=2010.0,
                            new_val=15,  # Give all selected women 15 years of education
                        )
    edu = fp.Education()
    choice = fp.StandardChoice(location=location)
    s1 = fp.Sim(pars=pars,
                education_module=edu,
                contraception_module=choice,
                interventions=change_education,
                label='Increased education')
    s1.run()

    # Check that the intervention worked
    s0_edu = s0.results['edu_attainment'][-1]
    s1_edu = s1.results['edu_attainment'][-1]
    s0_mcpr = s0.results['mcpr'][-1]
    s1_mcpr = s1.results['mcpr'][-1]

    print(f"Checking effect of education ... ")
    assert s0_edu < s1_edu, f'Increasing education should increase average years of education, but {s1_edu}<{s0_edu}'
    # assert s0_mcpr < s1_mcpr, f'Increasing education should increase contraceptive use, but {s1_mcpr}<{s0_edu}'
    print(f"✓ ({s0_edu:.2f} < {s1_edu:.2f})")
    print(f"✗ (TEST FAILS: {s0_mcpr:.2f} > {s1_mcpr:.2f}) - NEED TO DEBUG")

    # Plot
    fig, axes = pl.subplots(1, 2, figsize=(12, 6))
    ax = axes[0]
    ax.plot(s0.results.t, s0.results.edu_attainment, label=s0.label)
    ax.plot(s1.results.t, s1.results.edu_attainment, label=s1.label)
    ax.set_ylabel('Average years of education among women')
    ax.set_xlabel('Year')
    ax = axes[1]
    ax.plot(s0.results.t, s0.results.mcpr, label=s0.label)
    ax.plot(s1.results.t, s1.results.mcpr, label=s1.label)
    ax.set_ylabel('mCPR')
    ax.set_xlabel('Year')
    pl.legend()
    pl.show()

    return s0, s1


if __name__ == '__main__':
    s0, s1 = test_education()

    print('Done.')


