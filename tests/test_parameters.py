"""
Run tests on individual parameters.
"""

import numpy as np
import sciris as sc
import fpsim as fp
import pylab as pl
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

    # Tests
    for key in ['births', 'deaths']:
        n = sim.results[key].sum()
        assert n == 0, f'Expecting {key} to be 0, not {n}'

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

    assert t2 < t1, f'Expecting runtime to be less with a larger method timestep, but {t2:0.3f} > {t1:0.3f}'
    print(f'Larger method timestep reduced runtime from {t1:0.3f} s to {t2:0.3f} s')

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

    assert mcpr_last > decreasing, f'Negative MCPR growth did not reduce MCPR ({decreasing:0.3f} ≥ {mcpr_last:0.3f})'
    assert mcpr_last < increasing, f'Positive MCPR growth did not increase MCPR ({increasing:0.3f} ≤ {mcpr_last:0.3f})'
    print(f'MCPR changed as expected: {decreasing:0.3f} < {mcpr_last:0.3f} < {increasing:0.3f}')

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


def test_matrix_methods():
    sc.heading('Test matrix methods')

    pars = fp.pars('test')
    n = len(pars['methods']['map'])

    # Test add method
    p1 = pars.copy()
    name = 'New method'
    p1.add_method(name=name, eff=1.0)
    s1 = fp.Sim(pars=p1)
    s1.run()
    assert s1.pars['methods']['map'][name] == n, 'Last entry does not have expected shape'

    # Test remove method
    p2 = pars.copy()
    p2.rm_method(name='Injectables')
    s2 = fp.Sim(pars=p2)
    s2.run()
    assert len(s2.pars['methods']['map']) == n-1, 'Methods do not have expected shape'

    # Test reorder methods
    p3 = pars.copy()
    reverse = list(p3['methods']['map'].values())[::-1]
    p3.reorder_methods(reverse)
    s3 = fp.Sim(pars=p3)
    s3.run()

    # Test copy method
    p4 = pars.copy()
    new_name = 'New method'
    orig_name = 'Injectables'
    p4.add_method(name=new_name, eff=1.0)
    p4.update_method_prob(dest=new_name, copy_from=orig_name, matrix='annual')

    # Do tests
    new_ind = fp.defaults.method_map[new_name]
    orig_ind = fp.defaults.method_map[orig_name]
    nestkeys = ['methods', 'raw', 'annual', '>25']
    pars_arr = sc.getnested(pars, nestkeys)
    p4_arr = sc.getnested(p4, nestkeys)
    assert p4_arr[0, new_ind] == pars_arr[0, orig_ind], 'Copied method has different initiation rate'
    if do_plot:
        pl.figure()
        pl.subplot(2,1,1)
        pl.pcolor(pars['methods']['raw']['annual']['>25'])
        pl.title('Original')
        pl.subplot(2,1,2)
        pl.pcolor(p4['methods']['raw']['annual']['>25'])
        pl.title('With new method')

    return [s1, s2, s3, p4]


def test_validation():
    sc.heading('Test parameter validation')

    pars = fp.pars('test') # Don't really need "test" since not running

    # Extra value not allowed
    with pytest.raises(ValueError):
        fp.pars(not_a_par=4)

    # Equivalent implementation
    with pytest.raises(ValueError):
        p = sc.dcp(pars)
        p['not_a_par'] = 4
        p.validate()

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
        # null    = test_null(do_plot=do_plot)
        # timings = test_method_timestep()
        # mcpr    = test_mcpr_growth()
        # scale   = test_scale()
        meths   = test_matrix_methods()
        pars    = test_validation()
