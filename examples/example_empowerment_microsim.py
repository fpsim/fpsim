'''
Simple simulation using the new empowement related attributes
'''
import numpy as np
import matplotlib.pyplot as plt

import sciris as sc
import fpsim as fp

# Set options
do_plot = True
empwr = fp.Empowerment()
pars = fp.pars(location='kenya')
pars['n_agents'] = 1_000  # Small population size

sc.tic()
sim = fp.Sim(pars=pars, empowerment_module=empwr, analyzers=[fp.empowerment_recorder()])
sim.run()

if do_plot:
    sim.plot()
    empwr_analyzer = sim.get_analyzers()[0]
    empwr_analyzer.plot()
    plt.show()

sc.toc()
print('Done.')
