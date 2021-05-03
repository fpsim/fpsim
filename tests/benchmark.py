'''
Benchmark the simulation
'''

import sciris as sc
import fpsim as fp
from fp_analyses import senegal_parameters as sp

do_profile = 0
pars = sp.make_pars()
sim = fp.Sim(pars)
to_profile = 'run'

func_options = {
    'run':         sim.run,
    'update':      sim.people.update,
    'people_init': sim.people.__init__,
    'other':       [fp.model.arr][0],
}

def run():
    pars = sp.make_pars()
    pars['n'] = 50000
    sim = fp.Sim(pars)
    sim.run()
    return sim


if __name__ == '__main__':
    if do_profile:
        sc.profile(run, func_options[to_profile])
    else:
        sc.tic()
        sim = run()
        sc.toc()
