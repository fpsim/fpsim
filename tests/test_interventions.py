"""
Run tests on the interventions.
"""

import sciris as sc
import fpsim as fp

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
    ''' Testing change_par() '''
    sc.heading('Testing change_par()...')

    # cp1 = fp.change_par(par='exposure_correction', years=2002, vals=0.0)
    # cp2 = fp.change_par('mcpr_growth_rate', vals={2020:0.05, 2030:-0.05})

    s0 = make_sim(label='Baseline')
    s1 = make_sim(label='e2')
    s2 = make_sim(label='e3')
    # s1 = make_sim(interventions=cp1, label='High exposure')
    # s2 = make_sim(interventions=cp2, start_year=2015, end_year=2040, label='Changing MCPR')
    m = fp.parallel(s0, s1, s2, serial=serial)
    s0, s1, s2 = m.sims

    # Tests

    return m


def test_analyzers():
    ''' Test analyzers '''
    sc.heading('Testing analyzers...')

    sim = make_sim(analyzers=fp.snapshot(timesteps=[100, 200]))
    sim.run()

    return sim


if __name__ == '__main__':
    sc.options(backend=None) # Turn on interactive plots
    with sc.timer():
        # isim   = test_intervention_fn()
        cpsims = test_change_par()
        # asim   = test_analyzers()
