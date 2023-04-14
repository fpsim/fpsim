'''
Simple example usage for FPsim.
An automatic calibration algorithm that takes in any parameter for fpsim and optimizes it for a mismatch
between a sim and data based on the Experiment class
'''

import sciris as sc
import fpsim as fp
import pylab as pl

# Settings
do_plot = True
do_save = True
pars = fp.pars(location='senegal')
pars['n'] = 500
pars['verbose'] = 0

# Define calibration parameters -- best, low, high
#calib_pars = dict(
    #exposure_factor = [1.0, 0.5, 8.0],
#)

calib_pars = dict(
    high_parity=dict(best=4, low=1, high=8),
    high_parity_nonuse=dict(best=0.5, low=0.1, high=10)
)

# Run
T = sc.tic()
calib = fp.Calibration(pars=pars, n_trials=100)
calib.calibrate(calib_pars=calib_pars)
calib.summarize()

fig = calib.plot_best()
if do_save:
    pl.savefig("calib_best.png", bbox_inches='tight', dpi=100)

fig = calib.plot_trend()
if do_save:
    pl.savefig("calib_trend.png", )

sc.toc(T)

print('Done.')