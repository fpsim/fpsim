"""
Run tests on individual parameters.
"""

import os
import numpy as np
import sciris as sc
import fpsim as fp
import pytest
import types
import fpsim.defaults as fpd

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

    return sim


def test_timestep():
    pars = fp.pars('test')

    # Set options
    pars['n_agents'] = 500   # Small population size
    pars['end_year'] = 2020  # 1961 - 2020 is the normal date range
    pars['exposure_factor'] = 0.5  # Overall scale factor on probability of becoming pregnant

    for timestep in range(1, 13):
        pars['timestep'] = timestep
        sim = fp.Sim(pars=pars)
        sim.run()
        ok(f'simulation ran for timestep {timestep}')

    return


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
    methods = sc.dcp(fp.make_methods().Methods) # TEMP
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
    pars = fp.pars(location='senegal')

    # Make and run sim
    s = fp.Sim(pars)
    s.run()

    expected_rows = len(s.people)
    expected_cols = s.tiperyear

    for key in s.people.longitude.keys():
        df = s.people.longitude[key]
        assert df.shape == (expected_rows, expected_cols), f"Expected {key} to have dimensions ({expected_rows}, {expected_cols}), but got {df.shape}"
        curr_year_index = s.ti % s.tiperyear
        assert (df[:, curr_year_index] == s.people[key]).all(), f"Expected column {curr_year_index} to have same longitudinal data as {key} but it does not."


def test_register_custom_location():
    sc.heading('Testing ability to register a custom location')

    # Create a dummy location module
    dummy_module = types.SimpleNamespace()

    # Add a fake make_pars function
    def make_pars(seed=None):
        return {'location': 'dummy', 'seed': seed}

    # Optionally add fake data_utils
    class DummyDataUtils:
        @staticmethod
        def process_contra_use(use_type, location):
            return f"Processed {use_type} for {location}"

    dummy_module.make_pars = make_pars
    dummy_module.data_utils = DummyDataUtils()

    # Register the custom location
    fpd.register_location('dummy', dummy_module)

    # Retrieve it and test
    assert 'dummy' in fpd.location_registry
    mod = fpd.location_registry['dummy']
    assert mod.make_pars(seed=42)['location'] == 'dummy'
    assert mod.data_utils.process_contra_use('simple', 'dummy') == "Processed simple for dummy"


if __name__ == '__main__':

    sc.options(backend=None) # Turn on interactive plots
    with sc.timer():
        null    = test_null(do_plot=do_plot)
        scale   = test_scale()
        meths   = test_method_changes()
        pars    = test_validation()
        p2      = test_save_load()
        long    = test_long_params()
        custom_loc = test_register_custom_location()
