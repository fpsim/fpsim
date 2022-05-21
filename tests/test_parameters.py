"""
Run tests on individual parameters.
"""

import numpy as np
import sciris as sc
import fpsim as fp
import pytest

do_plot = True
sc.options(backend='agg') # Turn off interactive plots


def test_null(do_plot=do_plot):
    sc.heading('Testing no births, no deaths...')

    pars = fp.pars('test') # For default pars

    # Set things to zero
    for key in ['exposure_factor', 'high_parity_nonuse']:
        pars[key] = 0

    for key in ['f', 'm']:
        pars['age_mortality'][key] *= 0

    for key in ['age_mortality', 'maternal_mortality', 'infant_mortality']:
        pars[key]['probs'] *= 0

    sim = fp.Sim(pars)
    sim.run()

    if do_plot:
        sim.plot()

    return sim


def test_method_timestep():
    sc.heading('Test sim speed')

    pars1 = fp.pars(location='test', method_timestep=1)
    pars2 = fp.pars(location='test', method_timestep=6)
    sim1 = fp.Sim(pars1)
    sim2 = fp.Sim(pars2)

    T = sc.timer()

    sim1.run()
    t1 = T.tt(output=True)

    sim2.run()
    t2 = T.tt(output=True)

    assert t2 < t1, 'Expecting runtime to be less with a larger method timestep'

    return [t1, t2]


def test_mcpr_growth():
    sc.heading('Test MCPR growth assumptions')

    pars = dict(
        start_year = 2010,
        end_year   = 2030, # Should be after last MCPR data year
    )

    pars1 = fp.pars(location='test', mcpr_growth_rate=-0.05, **pars)
    pars2 = fp.pars(location='test', mcpr_growth_rate=0.05, **pars)
    sim1 = fp.Sim(pars1)
    sim2 = fp.Sim(pars2)

    msim = fp.MultiSim([sim1, sim2]).run()
    s1 = msim.sims[0]
    s2 = msim.sims[1]

    mcpr_last = pars1['methods']['mcpr_rates'][-1] # Last MCPR data point
    decreasing = s1.results['mcpr'][-1]
    increasing = s2.results['mcpr'][-1]

    assert mcpr_last > decreasing, f'Negative MCPR growth did not reduce MCPR ({decreasing} ≥ {mcpr_last})'
    assert mcpr_last < increasing, f'Positive MCPR growth did not increase MCPR ({increasing} ≤ {mcpr_last})'

    return [s1, s2]


def test_scale():
    sc.heading('Test scale factor')

    # Test settings
    orig_pop = 100
    scale = 2

    # Make and run sims
    pars = fp.pars('test')
    s1 = fp.Sim(pars, scaled_pop=orig_pop)
    s2 = fp.Sim(pars, scaled_pop=scale*orig_pop)
    msim = fp.parallel(s1, s2)
    s1, s2 = msim.sims

    # Tests
    assert np.array_equal(s1.results.mcpr, s2.results.mcpr), 'Scale factor should not change MCPR'
    assert scale*s1.results.total_births.sum() == s2.results.total_births.sum(), 'Total births should scale exactly with scale factor'

    return [s1, s2]


def test_validation():
    sc.heading('Test parameter validation')

    pars = fp.pars()

    # Extra value not allowed
    with pytest.raises(ValueError):
        fp.pars(not_a_par=4)

    # Missing value not allowed
    with pytest.raises(ValueError):
        p = sc.dcp(pars)
        p.pop('exposure_factor')
        p.validate()

    # Wrong matrix keys
    with pytest.raises(ValueError):
        p = sc.dcp(pars)
        p['methods']['raw']['annual'].pop('<18')
        p.validate()

    # Wrong matrix shape
    with pytest.raises(ValueError):
        p = sc.dcp(pars)
        matrix = p['methods']['raw']['annual']['<18']
        np.insert(matrix, (0,0), matrix[:,0])
        p.validate()

    return pars


if __name__ == '__main__':

    sc.options(backend=None) # Turn on interactive plots
    with sc.timer():
        null    = test_null(do_plot=do_plot)
        timings = test_method_timestep()
        mcpr    = test_mcpr_growth()
        scale   = test_scale()
        pars    = test_validation()
