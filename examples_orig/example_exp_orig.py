'''
Simple example usage for FPsim
'''

import sciris as sc
import fpsim_orig as fp
import fp_analyses_orig.senegal_parameters as sp

# Set options
do_plot = True
pars = sp.make_pars()
pars['n'] = 500 # Small population size
pars['end_year'] = 2020 # 1961 - 2020 is the normal date range
# pars['exposure_correction'] = 3.0 # Not implemented for original version

sc.tic()
exp = fp.Calibration(pars=pars) # In the original version, an Experiment was named a Calibration
exp.run()
sc.toc()

if do_plot:
    exp.plot()
    # exp.fit.plot() # Not implemented for original version

print('Done.')


