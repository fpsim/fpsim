'''
Benchmark the simulation
'''

import sciris as sc
import fpsim as fp
from fp_analyses import senegal_parameters as sp

pars = sp.make_pars()
sim = fp.Sim(pars)
to_profile = 'other'

func_options = {'run':         sim.run,
                'update':      sim.people.update,
                'people_init': sim.people.__init__,
                'other': [fp.model.arr][0],
                # 'get_method': person.get_method,
                }

def run():
    pars = sp.make_pars()
    pars['n'] = 500
    sim = fp.Sim(pars)
    sim.run()
    return


sc.profile(run, func_options[to_profile])
