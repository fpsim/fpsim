'''
Demonstrate multiparameter calibration with plotting.
'''

import fpsim as fp
import fp_analyses as fa

pars = fa.senegal_parameters.make_pars()
pars['n'] = 500
pars['verbose'] = 0
calib = fp.Calibration(pars=pars)

calib_pars = dict(
    exposure_correction = [1.0, 0.5, 1.5],
    maternal_mortality_multiplier = [1, 0.75, 3.0],
)

if __name__ == '__main__':

    calib.calibrate(calib_pars=calib_pars, n_trials=50)
    calib.plot_all()
    calib.plot_trend()