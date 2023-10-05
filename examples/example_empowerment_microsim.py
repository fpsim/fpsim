'''
Simple simulation using the new empowement related attributes
'''

import sciris as sc
import fpsim as fp
import matplotlib.pyplot as plt

# Set options
do_plot = True
pars = fp.pars(location='kenya')
pars['n_agents'] = 500 # Small population size



sc.tic()
sim = fp.Sim(pars=pars, analyzers=fp.empowerment_recorder(bins=[0, 15, 20, 40, 50, 100]))
sim.run()

if do_plot:
    sim.plot()
    empwr_analyzer= sim.get_analyzers()[0]
    empwr_analyzer.plot()
    plt.show()

sc.toc()
print('Done.')
