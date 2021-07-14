"""
Run tests on the calibration object.
"""

import sciris as sc
import fpsim as fp
import fp_analyses as fa

do_plot = 1
total_trials = 100

n_workers = sc.cpu_count()
n_trials = int(total_trials/n_workers)


# Set parameters
pars = fa.senegal_parameters.make_pars()
pars['n'] = 100
pars['verbose'] = 0.1
calib = fp.Calibration(pars=pars, n_workers=n_workers, n_trials=n_trials)

calib_pars = dict(
    exposure_correction = [1.0, 0.9, 1.1],
    fecundity_variation_low = [0.4, 0.1, 0.9],
    fecundity_variation_high = [1.4, 1.1, 1.9],
    maternal_mortality_multiplier = [1, 0.75, 3.0],
    abortion_prob = [0.086, 0.017, 0.1]
)

weights = dict(
    maternal_mortality_ratio = 0.0,
)


if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    T = sc.tic()

    # Calculate calibration
    calib.calibrate(calib_pars=calib_pars, n_trials=n_trials, weights=weights)
    before,after = calib.summarize()

    if do_plot:
        calib.before.plot()
        calib.after.plot()
        calib.before.fit.plot()
        calib.after.fit.plot()

    # Wrap up
    print('\n'*2)
    sc.toc(T)
    print('Done.')
