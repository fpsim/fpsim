'''
Benchmark the simulation
'''

import sciris as sc
import fpsim as fp
from fp_analyses import senegal_parameters as sp

pars = sp.make_pars()
sim = fp.Sim(pars)
person = sc.odict(sim.people)[0]
to_profile = 'sim'

func_options = {'sim':        sim.run,
                'update':     person.update,
                # 'preg':       person.get_preg_prob,
                # 'get_method': person.get_method,
                }

def run():
    pars = sp.make_pars()
    pars['n'] = 100
    sim = fp.Sim(pars)
    sim.run()
    return


sc.profile(run, func_options[to_profile])
