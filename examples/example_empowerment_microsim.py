'''
Simple simulation using the new empowement related attributes
'''

import sciris as sc
import fpsim as fp
import matplotlib.pyplot as plt
import numpy as np
# Set options
do_plot = True
pars = fp.pars(location='kenya',
               use_partnership=True,
               use_urban=True,
               use_empowerment=True)
pars['n_agents'] = 500 # Small population size



sc.tic()
age_bins = np.arange(100)[::5] # 5 year bins
sim = fp.Sim(pars=pars, analyzers=[fp.empowerment_recorder(bins=age_bins)])
sim.run()

if do_plot:
    sim.plot()
    empwr_analyzer= sim.get_analyzers()[0]
    empwr_analyzer.plot()
    plt.show()

sc.toc()
print('Done.')
