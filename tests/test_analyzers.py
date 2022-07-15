"""
Run tests on the analyzers, including calibration.
"""

import sciris as sc
import fpsim as fp


do_plot = 1
sc.options(backend='agg') # Turn off interactive plots


def ok(string):
    ''' Print out a successful test nicely '''
    return sc.printgreen(f'✓ {string}\n')


def make_analyzer(analyzer):
    ''' Create a sim with a single analyzer '''
    sim = fp.Sim(location='test', analyzers=analyzer).run()
    an = sim.get_analyzer()
    return an


def test_calibration(n_trials=3):
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


if __name__ == '__main__':

    sc.options(backend=None) # Turn on interactive plots
    with sc.timer():
        calib = test_calibration()
        snap  = test_snapshot()
        tsr   = test_timeseries_recorder()
        ap    = test_age_pyramids()
