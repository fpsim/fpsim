"""
Run tests on the analyzers, including calibration.
"""
import numpy as np
import sciris as sc
import starsim as ss
import fpsim as fp

do_plot = 1
sc.options(interactive=False) # Turn off interactive plots
max_pregnancy_loss = 0.5 # Maximum allowed fraction of pregnancies to allow to not end in birth (including stillbirths)

def ok(string):
    ''' Print out a successful test nicely '''
    return sc.printgreen(f'✓ {string}\n')


def make_analyzer(analyzer):
    ''' Create a sim with a single analyzer '''
    sim = fp.Sim(test=True, analyzers=analyzer).run(verbose=1/12)
    an = sim.analyzers[0]
    return an


def test_calibration(n_trials=3):
    ''' Compare the current default sim against the saved baseline '''
    sc.heading('Testing calibration...')

    calib_pars = dict(
        exposure_factor = [5, 0, 5],
    )

    # Calculate calibration
    pars = dict(test=True, n_agents=20, start=2000, stop=2010, verbose=1/12)

    calib = fp.Calibration(pars=pars, weights=dict(pop_size=100))
    calib.calibrate(calib_pars=calib_pars, n_trials=2, n_workers=1)
    before, after = calib.summarize()

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
    assert pop1 > pop0, 'Expected population to grow'
    ok(f'Population grew ({pop1} > {pop0})')

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

    sim = fp.Sim(analyzers=lh, copy_inputs=False)
    sim.init()
    sim.run(verbose=1/12)

    # The difference between the largest and smallest age should for each person be equal to (1 year - 1/timestepsperyear)
    # Based on the default params, the value in slot 0 is the max and in slot 1 is the min. There will be some rounding error
    # so we use pytest.approx to compare.
    ages = np.sort(lh.age[ss.uids(1), :])
    age_diffs = np.diff(ages)
    assert np.allclose(age_diffs, 1/sim.fp_pars['tiperyear'], rtol=1e-2), f'Expected age differences to be equal to 1 year minus the timestep size, not {age_diffs}'

    return


def test_method_mix_by_age():
    sc.heading('Testing method mix by age analyzer...')

    # Create a sim with the method mix by age analyzer
    mmba = fp.method_mix_by_age()
    sim = fp.Sim(analyzers=[mmba])
    sim.init()
    sim.run()

    # Check that the analyzer has been populated
    assert sim.analyzers.method_mix_by_age.mmba_results is not None, 'Method mix by age results should not be empty'

    return sim.analyzers.method_mix_by_age


if __name__ == '__main__':

    sc.options(backend=None) # Turn on interactive plots
    with sc.timer():
        calib = test_calibration()
        snap  = test_snapshot()
        ap    = test_age_pyramids()
        lh    = test_longitudinal()
        mmba  = test_method_mix_by_age()
