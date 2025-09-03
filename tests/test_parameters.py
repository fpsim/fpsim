"""
Run tests on individual parameters.
"""

import os
import numpy as np
import sciris as sc
import fpsim as fp
import pytest
import starsim as ss
import types
import fpsim.defaults as fpd

do_plot = True
sc.options(backend='agg') # Turn off interactive plots

def ok(string):
    ''' Print out a successful test nicely '''
    return sc.printgreen(f'✓ {string}\n')


def test_null(do_plot=do_plot):
    sc.heading('Testing no births, no deaths...')

    fp_pars = fp.make_fp_pars()  # For default pars
    fp_pars.update_location('senegal')

    # Set things to zero
    for key in ['exposure_factor']:
        fp_pars[key] = 0

    for key in ['f', 'm']:
        fp_pars['age_mortality'][key] *= 0

    for key in ['age_mortality', 'maternal_mortality', 'infant_mortality']:
        fp_pars[key]['probs'] *= 0

    sim = fp.Sim(test=True, fp_pars=fp_pars)
    sim.run()

    # Tests
    n = sim.results.fp.births.sum()
    assert n == 0, f'Expecting 0 births, not {n}'
    n = sim.results.new_deaths.sum()
    assert n == 0, f'Expecting 0 deaths, not {n}'
    ok(f'Births and deaths are 0, as expected')

    return sim


def test_scale():
    sc.heading('Test scale factor')

    # Test settings
    scale = 2

    # Make and run sims
    pars = dict(test=True)
    s1 = fp.Sim(pars=pars)
    s2 = fp.Sim(pars=pars, pop_scale=scale)
    msim = ss.parallel([s1, s2], shrink=False)
    s1, s2 = msim.sims

    # Tests
    orig = s1.results.fp.total_births.sum()
    expected = scale*orig
    actual = s2.results.fp.total_births.sum()
    assert expected == actual, 'Total births should scale exactly with scale factor'
    assert np.array_equal(s1.results.contraception.mcpr, s2.results.contraception.mcpr), 'Scale factor should not change MCPR'
    ok(f'{actual} births = {scale}*{orig} as expected')

    return [s1, s2]


def test_method_changes():
    sc.heading('Test changing methods')

    # # Test adding method
    choice = fp.RandomChoice()
    n = len(choice.methods)
    new_method = fp.Method(
        name='new',
        efficacy=1,
        modern=True,
        dur_use=dict(dist='lognormal', par1=10, par2=3),
        label='New method')
    choice.add_method(new_method)
    s1 = fp.Sim(test=True, contraception_module=choice)
    s1.run()
    assert len(s1.connectors.contraception.methods) == n+1, 'Method was not added'
    ok(f'Methods had expected length after addition ({n+1})')

    # Test remove method
    methods = [m for m in fp.make_method_list() if m.label != 'Injectables']
    choice = fp.RandomChoice(methods=methods)
    s2 = fp.Sim(test=True, contraception_module=choice)
    s2.run()
    assert len(s2.connectors.contraception.methods) == len(methods), 'Methods was not removed'
    ok(f'Methods have expected length after removal ({n})')

    # Test method efficacy
    methods = fp.make_method_list()
    for method in methods: method.efficacy = 1  # Make all methods totally effective
    choice = fp.RandomChoice(pars=dict(p_use=1), methods=methods)
    s3 = fp.Sim(test=True, contraception_module=choice)
    s3.run()
    assert s3.results.fp.births.sum() == 0, f'Expecting births to be 0, not {n}'
    ok(f'No births with completely effective contraception, as expected')


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

    sc.options(backend=None)  # Turn on interactive plots
    with sc.timer():
        null    = test_null(do_plot=do_plot)
        scale   = test_scale()
        meths   = test_method_changes()
        custom_loc = test_register_custom_location()
