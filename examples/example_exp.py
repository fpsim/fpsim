'''
Simple example usage for FPsim
'''

import sciris as sc
import fpsim as fp
import fp_analyses.senegal_parameters as sp

# Set options
do_plot = True
pars = sp.make_pars()
pars['n'] = 500 # Small population size
pars['end_year'] = 2020 # 1961 - 2020 is the normal date range
pars['exposure_correction'] = 1.0 # Overall scale factor on probability of becoming pregnant -- 3.0 gives the best match to data, but it's not very good!

sc.tic()
exp = fp.Experiment(pars=pars)
exp.run()
sc.toc()

if do_plot:
    exp.plot()
    exp.fit.plot()

print('Done.')


