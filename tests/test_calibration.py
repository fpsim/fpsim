"""
Run tests on the calibration object.
"""

import sciris as sc
import fpsim as fp


do_plot = 1
sc.options(backend='agg') # Turn off interactive plots


def ok(string):
    ''' Print out a successful test nicely '''
    return sc.printgreen(f'✓ {string}\n')


def make_calib():
    '''
    Define a default simulation for testing the baseline.
    '''
    pars = fp.pars('test', n_agents=200)
    calib = fp.Calibration(pars=pars)
    return calib


def test_calibration(n_trials=3):
    ''' Compare the current default sim against the saved baseline '''
    sc.heading('Testing calibration...')

    calib_pars = dict(
        exposure_factor = [1.5, 1.4, 1.6],
    )

    # Calculate calibration
    calib = make_calib()
    calib.calibrate(calib_pars=calib_pars, n_trials=n_trials, n_workers=2)
    before,after = calib.summarize()

    assert after <= before, 'Expect calibration to not make fit worse'
    ok(f'Calibration improved fit ({after:n} < {before:n})')

    if do_plot:
        calib.after.plot()
        calib.after.fit.plot()

    return calib


if __name__ == '__main__':

    sc.options(backend=None) # Turn on interactive plots
    with sc.timer():
        calib = test_calibration()
