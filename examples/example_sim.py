'''
Simple example usage for FPsim
'''

import sciris as sc
import fpsim as fp

# Set options
do_plot = True
pars = fp.pars(location='kenya')
pars['n_agents'] = 500 # Small population size
pars['end_year'] = 2020 # 1961 - 2020 is the normal date range
pars['exposure_correction'] = 1.0 # Overall scale factor on probability of becoming pregnant

sc.tic()
sim = fp.Sim(pars=pars)
sim.run()

if do_plot:
    sim.plot()

sc.toc()
print('Done.')
