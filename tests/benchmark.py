'''
Benchmark the simulation
'''

import sciris as sc
import fpsim as fp
from fp_analyses import senegal_parameters as sp

pars = sp.make_pars()
sim = fp.Sim(pars)
to_profile = 'update'

func_options = {'sim':        sim.run,
                'update':     sim.people.update,
                # 'preg':       person.get_preg_prob,
                # 'get_method': person.get_method,
                }

def run():
    pars = sp.make_pars()
    pars['n'] = 500
    sim = fp.Sim(pars)
    sim.run()
    return


sc.profile(run, func_options[to_profile])
