'''
Simple simulation using the new empowement related attributes
'''

import sciris as sc
import fpsim as fp

# Set options
do_plot = True
pars = fp.pars(location='kenya')
pars['n_agents'] = 500 # Small population size

sc.tic()
sim = fp.Sim(pars=pars)
sim.run()

if do_plot:
    sim.plot()

sc.toc()
print('Done.')
