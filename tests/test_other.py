"""
Test other things not covered in other tests.
"""

import os
import sciris as sc
import fpsim as fp


do_plot  = 1 # Whether to do plotting in interactive mode
sc.options(backend='agg') # Turn off interactive plots

def ok(string):
    ''' Print out a successful test nicely '''
    return sc.printgreen(f'✓ {string}\n')


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


# Run all tests
if __name__ == '__main__':

    sc.options(backend=None) # Turn on interactive plots

    with sc.timer():
        opts = test_options()
        df   = test_to_df()
        ppl  = test_plot_people()