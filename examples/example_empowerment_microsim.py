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
sim = fp.Sim(pars=pars, analyzers=fp.empowerment_recorder())
sim.run()

if do_plot:
    sim.plot()
    empwr_analyzer= sim.get_analyzers()[0]
    empwr_analyzer.plot(data_args=['edu_objective'])
    import  ipdb; ipdb.set_trace()

sc.toc()
print('Done.')
