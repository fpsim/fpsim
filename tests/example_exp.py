'''
Simple example usage for FPsim
'''

import sciris as sc
import fpsim as fp
import fp_analyses.senegal_parameters as sp

doplot = True
dosave = False

sc.tic()
pars = sp.make_pars()
pars['n'] = 5000 # Only do a partial run
# pars['end_year'] = 2020
pars['exposure_correction'] = 10
pars['verbose'] = 0
exp = fp.Experiment(pars=pars)
exp.run()
if doplot:
    exp.plot()
    exp.fit.plot()

sc.toc()
print('Done.')


