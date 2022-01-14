'''
Plot animation of agents in FPsim
'''

import sciris as sc
import fpsim as fp
import fp_analyses as fa


n = 100

pars = fa.senegal_parameters.make_pars()
pars['n'] = n
pars['start_year'] = 1990
sim = fp.Sim(pars)
sim.run()

sim.plot()
