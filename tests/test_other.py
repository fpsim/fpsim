"""
Test other things not covered in other tests.
"""

import os
import numpy as np
import sciris as sc
import fpsim as fp
import pytest


do_plot  = 1 # Whether to do plotting in interactive mode
sc.options(backend='agg') # Turn off interactive plots

def ok(string, newline=True):
    ''' Print out a successful test nicely '''
    return sc.printgreen(f'✓ {string}' + '\n'*newline)


def test_options():
    sc.heading('Testing options...')

    d = fp.options.to_dict()
    assert isinstance(d, dict), 'Expected a dict'
    ok('Options to_dict() works')

    with fp.options.context(jupyter=True):
        assert fp.options.returnfig == False, 'Jupyter should disable returnfig'
        ok('Options as context works (note: may raise warning)')

    fp.options.disp()
    ok('Options disp() works')

    filename = 'tmp_settings.json'
    fp.options.save(filename)
    assert os.path.exists(filename), 'Did not write file to disk'
    fp.options.load(filename)
    os.remove(filename)
    ok('Options load() and save() work')

    return sc.dcp(fp.options)


def test_to_df():
    sc.heading('Testing other sim methods...')

    sim = fp.Sim(location='test').run()
    sim.brief()
    ok('sim.brief() worked')

    df = sim.to_df()
    births = df.births.sum()
    last = df.t.values[-1]
    assert last == sim['end_year'], 'Last years do not match'
    assert births > 0, 'Expected births'
    ok(f'to_df() worked to capture {births} births and final year {last}')

    return df


def test_plot_people():
    sc.heading('Test plotting people...')

    sim = fp.Sim(location='test').run()

    if do_plot:
        sim.people.plot()

    return sim.people


def test_samples(do_plot=False, verbose=True):
    sc.heading('Samples distribution')

    n = 200_000

    # Warning, must match utils.py!
    choices = [
        'uniform',
        'normal',
        'lognormal',
        'normal_pos',
        'normal_int',
        'lognormal_int',
    ]

    # Run the samples
    nchoices = len(choices)
    nsqr, _ = sc.get_rows_cols(nchoices)
    results = sc.objdict()
    mean = 11
    std = 7
    low = 3
    high = 9
    normal_dists = ['normal', 'normal_pos', 'normal_int', 'lognormal', 'lognormal_int']
    for c,choice in enumerate(choices):
        kw = {}
        if choice in normal_dists:
            par1 = mean
            par2 = std
        elif choice == 'neg_binomial':
            par1 = mean
            par2 = 1.2
            kw['step'] = 0.1
        elif choice == 'poisson':
            par1 = mean
            par2 = 0
        elif choice == 'uniform':
            par1 = low
            par2 = high
        else:
            errormsg = f'Choice "{choice}" not implemented'
            raise NotImplementedError(errormsg)

        # Compute
        results[choice] = fp.sample(dist=choice, par1=par1, par2=par2, size=n, **kw)

    with pytest.raises(NotImplementedError):
        fp.sample(dist='not_found')

    # Do statistical tests
    tol = 1/np.sqrt(n/50/len(choices)) # Define acceptable tolerance -- broad to avoid false positives

    def isclose(choice, tol=tol, **kwargs):
        key = list(kwargs.keys())[0]
        ref = list(kwargs.values())[0]
        npfunc = getattr(np, key)
        value = npfunc(results[choice])
        msg = f'Test for {choice:14s}: expecting {key:4s} = {ref:5.2f} ± {tol*ref:4.2f} and got {value:5.2f}'
        if verbose:
            ok(msg, newline=False)
        assert np.isclose(value, ref, rtol=tol), msg
        return True

    # Normal
    for choice in normal_dists:
        isclose(choice, mean=mean)
        if all([k not in choice for k in ['_pos', '_int']]): # These change the variance
            isclose(choice, std=std)

    # Uniform
    isclose('uniform', mean=(low+high)/2)

    return results

def test_method_usage():
    '''Test that method usage proportions add to 1 and correspond to population'''
    sim = fp.Sim(location='test')
    sim.run() 
    for timestep, proportions in enumerate(sim.results['method_usage']):
        assert np.isclose(sum(proportions), 1, atol=0.0001)
        pop = sim.results['pop_size'][timestep]

        # Checking that proportion isn't calculated from a larger population than expected
        for proportion in proportions:
            if proportion > 0:
                assert (proportion * pop) > 1, "Method usage proportions drawing from a larger population than expected"
    
    return sim

# Run all tests
if __name__ == '__main__':

    sc.options(backend=None) # Turn on interactive plots

    with sc.timer():
        opts = test_options()
        df   = test_to_df()
        ppl  = test_plot_people()
        res  = test_samples()
        method = test_method_usage()