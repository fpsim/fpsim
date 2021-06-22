"""
Run tests on the calibration object.
"""

import sciris as sc
import fpsim as fp
import fp_analyses as fa

do_plot = 1

def make_calib(n=500):
    '''
    Define a default simulation for testing the baseline.
    '''
    pars = fa.senegal_parameters.make_pars()
    pars['n'] = n
    pars['verbose'] = 0
    calib = fp.Calibration(pars=pars)

    return calib


def test_calibration(n_trials=5, do_plot=False):
    ''' Compare the current default sim against the saved baseline '''
    sc.heading('Testing calibration...')

    calib_pars = dict(
        exposure_correction = [1.0, 0.5, 8.0],
    )

    # Calculate calibration
    calib = make_calib()
    calib.calibrate(calib_pars=calib_pars, n_trials=n_trials)
    before,after = calib.summarize()

    assert before > after

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