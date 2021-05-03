"""
Run tests on the calibration object.
"""

import sciris as sc
import fpsim as fp
import fp_analyses as fa

do_plot = 0

def make_calib(n=500):
    '''
    Define a default simulation for testing the baseline.
    '''
    pars = fa.senegal_parameters.make_pars()
    pars['n'] = n
    calib = fp.Calibration(pars=pars)

    return calib


def test_calibration(max_iters=10, do_plot=False):
    ''' Compare the current default sim against the saved baseline '''
    sc.heading('Testing calibration...')

    calib_pars = dict(
        exposure_correction = [0.5, 2.0],
    )

    # Calculate calibration
    calib = make_calib()
    calib.calibrate(calib_pars=calib_pars, max_iters=max_iters)
    print(calib.results)

    if do_plot:
        raise NotImplementedError

    return calib


if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    T = sc.tic()

    calib = test_calibration(do_plot=do_plot)

    print('\n'*2)
    sc.toc(T)
    print('Done.')