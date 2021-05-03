'''
Simple example usage for FPsim
'''

import sciris as sc
import fpsim as fp
import fp_analyses.senegal_parameters as sp

# Settings
doplot = True
dosave = False
pars = sp.make_pars()
pars['n'] = 500
pars['verbose'] = 0

# Define calibration parameters -- best, low, high
calib_pars = dict(
    exposure_correction = [1.0, 0.5, 8.0],
)

# Run
sc.tic()
calib = fp.Calibration(pars=pars, n_trials=10)
calib.calibrate(calib_pars=calib_pars)
calib.summarize()
sc.toc()

print('Done.')