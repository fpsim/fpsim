'''
Benchmark the simulation
'''

import sciris as sc
import fpsim as fp

do_profile = 1
sim = fp.Sim()
sim.initialize()

ms = fp.MethodSelector(contra_use_file='contra_coef.csv', method_choice_file='method_mix.csv')
ppl = sim.people

to_profile = sc.objdict(
    run         = sim.run,
    update      = ppl.update,           # 70% of sim.run() runtime is spent here
    people_init = ppl.__init__,
    methods     = ppl.update_methods,   # 50% of ppl.update() runtime is spent here
    method_pp   = ppl.update_method_pp, # 53% of ppl.update_methods() runtime is spent here
    method      = ppl.update_method,    # 44% of ppl.update_methods() runtime is spent here
    choose_method = ms.choose_method,    # 44% of ppl.update_methods() runtime is spent here
    filter      = ppl.filter            # 58% of ppl.update_method() runtime is spent here
)['choose_method']


def run():
    ms = fp.MethodSelector(contra_use_file='contra_coef.csv', method_choice_file='method_mix.csv')
    pars = fp.pars(location='test', verbose=1)
    sim = fp.Sim(pars, method_selector=ms)
    sim.run()
    return sim


if __name__ == '__main__':
    if do_profile:
        sc.profile(run, to_profile)
    else:
        sc.tic()
        sim = run()
        sc.toc()
