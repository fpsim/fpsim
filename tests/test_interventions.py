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

    # Define exposure test
    cp1 = fp.change_par(par='exposure_correction', years=2002, vals=0.0) # Reduce exposure correction
    s0 = make_sim(label='Baseline')
    s1 = make_sim(interventions=cp1, label='High exposure')

    # Define MCPR growth test
    sim_start = 2015
    mcpr_y1   = 2020
    mcpr_y2   = 2030
    sim_end   = 2040
    cp2 = fp.change_par('mcpr_growth_rate', vals={mcpr_y1:0.10, mcpr_y2:-0.10}) # Set to large positive then negative growth
    s2 = make_sim(interventions=cp2, start_year=sim_start, end_year=sim_end, label='Changing MCPR')

    # Run
    m = fp.parallel(s0, s1, s2, serial=serial)

    # Test exposure correction change
    base_births = m.sims[0].results['births'].sum()
    cp_births   = m.sims[1].results['births'].sum()
    assert cp_births < base_births, f'Reducing exposure correction should reduce births, but {cp_births} is not less than the baseline of {base_births}'

    # Test MCPR growth
    r = m.sims[2].results
    ind_y1 = sc.findnearest(r.t, mcpr_y1)
    ind_y2 = sc.findnearest(r.t, mcpr_y2)
    ind_y3 = sc.findnearest(r.t, sim_end)
    assert ind_y1 != ind_y2 != ind_y3, f'Sim indices for years years {mcpr_y1}, {mcpr_y2}, {sim_end} should be different'
    assert r.mcpr[ind_y1] < r.mcpr[ind_y2], f'MCPR did not grow from {mcpr_y1} to {mcpr_y2}'
    assert r.mcpr[sim_end] < r.mcpr[ind_y2], f'MCPR did not shrink from {mcpr_y2} to {sim_end}'

    if do_plot:
        m.plot()

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
        cpmsim = test_change_par()
        # asim   = test_analyzers()
