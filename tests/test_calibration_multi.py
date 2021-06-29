"""
Run tests on the calibration object.
"""

import sciris as sc
import fpsim as fp
import fp_analyses as fa

do_plot = 1

def make_calib(n=5000):
    '''
    Define a default simulation for testing the baseline.
    '''
    pars = fa.senegal_parameters.make_pars()
    pars['n'] = n
    pars['verbose'] = 0.1
    calib = fp.Calibration(pars=pars)

    return calib


def test_calibration(n_trials=25, do_plot=False):
    ''' Compare the current default sim against the saved baseline '''
    sc.heading('Testing calibration...')

    calib_pars = dict(
        exposure_correction = [1.0, 0.9, 1.1],
        fecundity_variation_low = [0.4, 0.1, 0.9],
        fecundity_variation_high = [1.4, 1.1, 1.9],
        maternal_mortality_multiplier = [1, 0.75, 3.0],
        abortion_prob = [0.086, 0.017, 0.1]
    )

    # Calculate calibration
    calib = make_calib()
    calib.calibrate(calib_pars=calib_pars, n_trials=n_trials)
    before,after = calib.summarize()

    assert before > after

    return calib


if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    T = sc.tic()

    calib = test_calibration(do_plot=do_plot)

    print('\n'*2)
    sc.toc(T)
    print('Done.')