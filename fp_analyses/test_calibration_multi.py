"""
Run tests on the calibration object.
"""

import sciris as sc
import fpsim as fp
import fp_analyses as fa

n_trials = 50


# Set parameters
pars = fa.senegal_parameters.make_pars()
pars['n'] = 1000
pars['verbose'] = 0.1
calib = fp.Calibration(pars=pars)

calib_pars = dict(
    exposure_correction = [1.0, 0.5, 1.5],
    fecundity_variation_low = [0.4, 0.1, 0.9],
    fecundity_variation_high = [1.4, 1.1, 1.9],
    maternal_mortality_multiplier = [1, 0.75, 3.0],
    abortion_prob = [0.086, 0.017, 0.1]
)


if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    T = sc.tic()

    # Calculate calibration
    calib.calibrate(calib_pars=calib_pars, n_trials=n_trials)
    before,after = calib.summarize()

    # Wrap up
    print('\n'*2)
    sc.toc(T)
    print('Done.')