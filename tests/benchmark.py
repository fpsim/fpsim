'''
Benchmark the simulation
'''

import sciris as sc
import fpsim as fp

do_profile = 1
sim = fp.Sim()
sim.initialize()

ppl = sim.people
to_profile = sc.objdict(
    run         = sim.run,
    update      = ppl.update,           # 70% of sim.run() runtime is spent here
    people_init = ppl.__init__,
    methods     = ppl.update_methods,   # 50% of ppl.update() runtime is spent here
    method_pp   = ppl.update_method_pp, # 53% of ppl.update_methods() runtime is spent here
    method      = ppl.update_method,    # 44% of ppl.update_methods() runtime is spent here
    filter      = ppl.filter            # 58% of ppl.update_method() runtime is spent here
)['run']


def run():
    pars = fp.pars(n_agents=10e1, method_timestep=1, verbose=0)
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
