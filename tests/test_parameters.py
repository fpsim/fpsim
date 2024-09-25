"""
Run tests on individual parameters.
"""

import os
import numpy as np
import sciris as sc
import fpsim as fp
import pylab as pl
import pytest

do_plot = True
sc.options(backend='agg') # Turn off interactive plots

def ok(string):
    ''' Print out a successful test nicely '''
    return sc.printgreen(f'✓ {string}\n')


def test_null(do_plot=do_plot):
    sc.heading('Testing no births, no deaths...')

    pars = fp.pars('test')  # For default pars

    # Set things to zero
    for key in ['exposure_factor']:
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
        ok(f'{key} was 0, as expected')

    if do_plot:
        sim.plot()

    return sim


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
    orig = s1.results.total_births.sum()
    expected = scale*orig
    actual = s2.results.total_births.sum()
    assert expected == actual, 'Total births should scale exactly with scale factor'
    assert np.array_equal(s1.results.mcpr, s2.results.mcpr), 'Scale factor should not change MCPR'
    ok(f'{actual} births = {scale}*{orig} as expected')

    return [s1, s2]


def test_method_changes():
    sc.heading('Test changing methods')

    # Test adding method
    choice = fp.RandomChoice()
    n = len(choice.methods)
    new_method = fp.Method(
        name='new',
        efficacy=1,
        modern=True,
        dur_use=dict(dist='lognormal', par1=10, par2=3),
        label='New method')
    choice.add_method(new_method)
    s1 = fp.Sim(location='test', contraception_module=choice)
    s1.run()
    assert len(s1.contraception_module.methods) == n+1, 'Method was not added'
    ok(f'Methods had expected length after addition ({n+1})')

    # Test remove method
    choice.remove_method('Injectables')
    s2 = fp.Sim(location='test', contraception_module=choice)
    s2.run()
    assert len(s2.contraception_module.methods) == n, 'Methods was not removed'
    ok(f'Methods have expected length after removal ({n})')

    # Test method efficacy
    methods = sc.dcp(fp.Methods)
    for method in methods.values():
        if method.name != 0: method.efficacy = 1  # Make all methods totally effective
    choice = fp.RandomChoice(pars=dict(p_use=1), methods=methods)
    s3 = fp.Sim(location='test', contraception_module=choice)
    s3.run()
    assert s3.results.births.sum() == 0, f'Expecting births to be 0, not {n}'
    ok(f'No births with completely effective contraception, as expected')


def test_validation():
    sc.heading('Test parameter validation')

    pars = fp.pars('test') # Don't really need "test" since not running

    # Extra value not allowed
    with pytest.raises(ValueError):
        fp.pars(not_a_par=4)
    ok('Invalid parameter name was caught')

    # Equivalent implementation
    with pytest.raises(ValueError):
        p = sc.dcp(pars)
        p['not_a_par'] = 4
        p.validate()
    ok('Invalid name was caught by validation')

    # Missing value not allowed
    with pytest.raises(ValueError):
        p = sc.dcp(pars)
        p.pop('exposure_factor')
        p.validate()
    ok('Missing parameter was caught by validation')

    return pars



def test_save_load():
    sc.heading('Testing saving and loading...')
    filename = 'tmp_pars.json'

    pars = fp.pars()
    pars.to_json(filename)
    assert os.path.exists(filename), 'Did not write file to disk'
    pars.from_json(filename)
    os.remove(filename)
    ok('pars.from_json() and pars.to_json() work')

    return pars


def test_long_params():
    sc.heading('Test longitudinal params')
    # Define pars
    pars = fp.pars(location='kenya')

    # Make and run sim
    s = fp.Sim(pars)
    s.run()

    expected_rows = len(s.people)
    expected_cols = s.tperyear

    for key in s.people.keys():
        if key.endswith('prev'):
            df = s.people[key]
            assert df.shape == (expected_rows, expected_cols), f"Expected {key} to have dimensions ({expected_rows}, {expected_cols}), but got {df.shape}"
            curr_data_key = key.removesuffix("_prev")
            curr_year_index = s.ti % s.tperyear
            assert (df[:, curr_year_index] == s.people[curr_data_key]).all(), f"Expected column {curr_year_index} to have same data as {curr_data_key} but it does not."


if __name__ == '__main__':

    sc.options(backend=None) # Turn on interactive plots
    with sc.timer():
        null    = test_null(do_plot=do_plot)
        scale   = test_scale()
        meths   = test_method_changes()
        pars    = test_validation()
        p2      = test_save_load()
