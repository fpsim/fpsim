"""
Run tests on the calibration object.
"""

import sciris as sc
import fpsim as fp


do_plot = 0

def make_calib():
    '''
    Define a default simulation for testing the baseline.
    '''
    pars = fp.pars('test', n=200)
    calib = fp.Calibration(pars=pars)
    return calib


def test_calibration(n_trials=3, do_plot=do_plot):
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

    if do_plot:
        calib.before.plot()
        calib.after.plot()
        calib.before.fit.plot()
        calib.after.fit.plot()

    return calib


if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    T = sc.tic()

    calib = test_calibration(do_plot=do_plot)

    print('\n'*2)
    sc.toc(T)
    print('Done.')
