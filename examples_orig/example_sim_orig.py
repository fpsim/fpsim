'''
Simple example usage for FPsim
'''

import sciris as sc
import fpsim_orig as fp
import fp_analyses_orig.senegal_parameters as sp

# Set options
do_plot = True
pars = sp.make_pars()
pars['n'] = 1000 # Small population size
pars['end_year'] = 1970 # 1961 - 2020 is the normal date range
# pars['exposure_correction'] = 1.0 # Overall scale factor on probability of becoming pregnant -- not implemented for original version

sc.tic()
sim = fp.Sim(pars=pars)
sim.run()

if do_plot:
    sim.plot()

sc.toc()
print('Done.')
