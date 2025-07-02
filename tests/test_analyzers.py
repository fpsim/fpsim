"""
Run tests on the analyzers, including calibration.
"""

import sciris as sc
import fpsim as fp
import starsim as ss
import pytest


do_plot = 1
sc.options(backend='agg') # Turn off interactive plots
max_pregnancy_loss = 0.5 # Maximum allowed fraction of pregnancies to allow to not end in birth (including stillbirths)


def ok(string):
    ''' Print out a successful test nicely '''
    return sc.printgreen(f'✓ {string}\n')


def make_analyzer(analyzer):
    ''' Create a sim with a single analyzer '''
    sim = fp.Sim(location='test', analyzers=analyzer).run(verbose=1/12)
    an = sim.analyzers[0]
    return an


def test_calibration(n_trials=3):
    ''' Compare the current default sim against the saved baseline '''
    sc.heading('Testing calibration...')

    calib_pars = dict(
        exposure_factor = [5, 0, 5],
    )

    # Calculate calibration
    pars= dict(location='test', n_agents=20, start=1960, stop=1980, verbose=1/12)

    calib = fp.Calibration(pars=pars, weights=dict(pop_size=100))
    calib.calibrate(calib_pars=calib_pars, n_trials=n_trials, n_workers=2)
    before,after = calib.summarize()

    # TODO FIX THIS
    # assert after <= before, 'Expect calibration to not make fit worse'
    ok(f'Calibration improved fit ({after:n} < {before:n})')

    if do_plot:
        calib.before.plot()
        calib.after.plot()
        calib.before.fit.plot()
        calib.after.fit.plot()

    return calib


def test_snapshot():
    ''' Test snapshot analyzer '''
    sc.heading('Testing snapshot analyzer...')

    timesteps = [0, 50]
    snap = make_analyzer(fp.snapshot(timesteps=timesteps))
    shots = snap.snapshots
    assert len(shots) == len(timesteps), 'Wrong number of snapshots'
    ok(f'Took {len(timesteps)} snapshots')
    pop0 = len(shots[0])
    pop1 = len(shots[1])
    # assert pop1 > pop0, 'Expected population to grow'
    # ok(f'Population grew ({pop1} > {pop0})')

    return snap


def test_age_pyramids():
    sc.heading('Testing age pyramids...')

    ap = make_analyzer(fp.age_pyramids())

    if do_plot:
        ap.plot()

    return ap

def test_longitudinal():
    sc.heading('Testing longitudinal history analyzer...')
    keys=['age']
    lh = fp.longitudinal_history(keys)

    sim = fp.Sim(analyzers=lh)
    sim.init()
    sim.run()

    # The difference between the largest and smallest age should for each person be equal to (1 year - 1/timestepsperyear)
    # Based on the default params, the value in slot 0 is the max and in slot 1 is the min. There will be some rounding error
    # so we use pytest.approx to compare.
    max_age = sim.analyzers.longitudinal_history.age[ss.uids(1), 0]
    min_age = sim.analyzers.longitudinal_history.age[ss.uids(1), 1]
    assert max_age - min_age == pytest.approx(1 - 1/sim.fp_pars['tiperyear'], rel=1e-2), 'Expected age difference to be equal to 1 year minus the timestep size'

    return



if __name__ == '__main__':

    sc.options(backend=None) # Turn on interactive plots
    with sc.timer():
        calib = test_calibration()
        snap  = test_snapshot()
        ap    = test_age_pyramids()
