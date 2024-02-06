"""
Run tests on the analyzers, including calibration.
"""

import sciris as sc
import fpsim as fp
import os
import numpy as np


do_plot = 1
sc.options(backend='agg') # Turn off interactive plots
max_pregnancy_loss = 0.5 # Maximum allowed fraction of pregnancies to allow to not end in birth (including stillbirths)


def ok(string):
    ''' Print out a successful test nicely '''
    return sc.printgreen(f'✓ {string}\n')


def make_analyzer(analyzer):
    ''' Create a sim with a single analyzer '''
    sim = fp.Sim(location='test', analyzers=analyzer).run()
    an = sim.get_analyzer()
    return an


def test_calibration(n_trials=5):
    ''' Compare the current default sim against the saved baseline '''
    sc.heading('Testing calibration...')

    calib_pars = dict(
        exposure_factor = [1.5, 1.4, 1.6],
    )

    # Calculate calibration
    pars = fp.pars('test', n_agents=200)
    calib = fp.Calibration(pars=pars)
    calib.calibrate(calib_pars=calib_pars, n_trials=n_trials, n_workers=2)
    before,after = calib.summarize()

    assert after <= before, 'Expect calibration to not make fit worse'
    ok(f'Calibration improved fit ({after:n} < {before:n})')

    if do_plot:
        calib.after.plot()
        calib.after.fit.plot()

    return calib


def test_snapshot():
    ''' Test snapshot analyzer '''
    sc.heading('Testing snapshot analyzer...')

    timesteps = [50, 100]
    snap = make_analyzer(fp.snapshot(timesteps=timesteps))
    shots = snap.snapshots
    assert len(shots) == len(timesteps), 'Wrong number of snapshots'
    ok(f'Took {len(timesteps)} snapshots')
    pop0 = len(shots[0])
    pop1 = len(shots[1])
    assert pop1 > pop0, 'Expected population to grow'
    ok(f'Population grew ({pop1} > {pop0})')

    return snap


def test_timeseries_recorder():
    sc.heading('Testing timeseries recorder...')

    tsr = make_analyzer(fp.timeseries_recorder())

    if do_plot:
        tsr.plot()

    return tsr


def test_age_pyramids():
    sc.heading('Testing age pyramids...')

    ap = make_analyzer(fp.age_pyramids())

    if do_plot:
        ap.plot()

    return ap

def test_verbose_sim():
    '''Test main verbose_sim functions'''
    sc.heading('Testing verbose sim ...')
    checked_channel = "gestation"
    verbose_sim = make_analyzer(fp.verbose_sim())
    verbose_sim.save(to_csv=True, to_json=True, custom_csv_tables=[checked_channel])
    
    total_results = verbose_sim.total_results
    csv_filepath = f"sim_output/{checked_channel}_state.csv"
    json_filepath = "sim_output/total_results.json"
    
    assert os.path.isfile(csv_filepath)
    assert os.path.isfile(json_filepath)
    
    os.remove(csv_filepath)
    os.remove(json_filepath)

    ok('verbose_sim.save() succeeded')

    assert all([len(channel) > 0 for channel in total_results[min(total_results.keys())]])
    ok('no empty channels in verbose_sim')

    verbose_sim.story(1)
    ok('verbose_sim.story() succeeded')

    return verbose_sim

def test_verbose_channels():
    ''' Test verbose_sim calculated channels '''
    sc.heading('Testing verbose sim added channels...')
    sim = fp.Sim(location='test', analyzers=fp.verbose_sim())
    sim.run()
    sim_verbose = sim['analyzers']
    events = sim_verbose.events
    channels =  sim_verbose.channels

    # Checks that none of the channels from events contain no entries.
    for channel in channels:
        if channel != "Deaths":
            maxval = 0
            for timestep in events:
                if len(events[timestep][channel]) > maxval:
                    maxval = len(events[timestep][channel])

        assert maxval > 0, f"Detected empty channel: {channel}"
    ok('No empty channels')

    miscarriages = 0
    births = 0
    conceptions = 0
    deaths = 0

    sim_births = 0
    sim_deaths = 0

    # Since analyzers are applied before people is updated for a timestep
    # we will skip the last timestep
    for timestep in events:
        births += len(events[timestep]['Births'])
        conceptions = conceptions + len(events[timestep]['Conceptions'])
        miscarriages += len(events[timestep]['Miscarriages'])
        deaths += len(events[timestep]['Deaths'])

    sim_births = sum(sim.results['births'][:-1])
    sim_deaths = sum(sim.results['deaths'][:-1])
    assert births == sim_births, f"sim.results births is {sim_births} and births channel is {births} on timestep {timestep}"
    assert deaths == sim_deaths, f"sim.results deaths is {sim_deaths} and deaths channel is {deaths} on timestep {timestep}"
    
    ok(f'No mismatch for births ({births} == {sim_births})')
    ok(f'No mismatch for deaths ({deaths} == {sim_deaths})')

    # Checks that conceptions is approximately births, and that conceptions is greater than the number of births.
    # We wouldn't expect more than a quarter of conceptions to end in miscarriages
    assert np.isclose(births, conceptions, atol=max_pregnancy_loss*births), "Less than 75 percent of conceptions result in births"
    assert conceptions > births, "Number of conceptions not greater than recorded births"
    ok(f'No mismatch for conceptions > miscarriages ({conceptions} > {miscarriages})')

    # Checks that miscarriages < difference between conceptions and births
    assert conceptions - births > miscarriages, "The number of miscarriages is greater than the differences between conceptions and births"
    ok(f'No mismatch for conceptions - births > miscarriages ({conceptions} - {births} > {miscarriages})')


if __name__ == '__main__':

    sc.options(backend=None) # Turn on interactive plots
    with sc.timer():
        calib = test_calibration()
        snap  = test_snapshot()
        tsr   = test_timeseries_recorder()
        ap    = test_age_pyramids()
        vs    = test_verbose_sim()
        nc    = test_verbose_channels()
