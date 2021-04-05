'''
Illustration of parallel run
'''

import sciris as sc
import fpsim as fp
import senegal_parameters as sp

def run(seed=1, n=100):
    pars = sp.make_pars()
    pars['seed'] = seed
    pars['n'] = n
    sim = fp.Sim(pars)
    sim.run()
    return sim

sims = sc.parallelize(run, range(10))
