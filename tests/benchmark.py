'''
Benchmark the simulation
'''

import sciris as sc
import fpsim as fp
from fp_analyses import senegal_parameters as sp

do_profile = 1
pars = sp.make_pars()
sim = fp.Sim(pars)

ppl = sim.people
to_profile = sc.objdict(
    run =         sim.run,
    update =      ppl.update, # 86%
    people_init = ppl.__init__,
    contra =      ppl.update_contraception, # 46%
    method_pp =   ppl.get_method_postpartum, # 56%, no obvious performance improvements
    get_method =  ppl.get_method, # 46%, could maybe be merged with previous
)['get_method']

def run():
    pars = sp.make_pars()
    pars['n'] = 10000
    sim = fp.Sim(pars)
    sim.run()
    return sim


if __name__ == '__main__':
    if do_profile:
        sc.profile(run, to_profile)
    else:
        sc.tic()
        sim = run()
        sc.toc()
