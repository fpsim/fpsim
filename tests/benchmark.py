'''
Benchmark the simulation
'''

import sciris as sc
import fpsim as fp

do_profile = 1
sim = fp.Sim()
sim.initialize()

coefficients = sc.objdict(intercept=.1, age=2, parity=3)
method_choice = fp.SimpleChoice(coefficients)
ppl = sim.people

to_profile = sc.objdict(
    run         = sim.run,
    update      = ppl.update,           # 70% of sim.run() runtime is spent here
    people_init = ppl.__init__,
    method      = ppl.update_method,    # 44% of ppl.update_methods() runtime is spent here
    choose_method = method_choice.choose_method,    # 44% of ppl.update_methods() runtime is spent here
    filter      = ppl.filter            # 58% of ppl.update_method() runtime is spent here
)['choose_method']


def run():
    coefficients = sc.objdict(intercept=.1, age=2, parity=3)
    method_choice = fp.SimpleChoice(coefficients)
    pars = fp.pars(location='test', verbose=1)
    sim = fp.Sim(pars, contraception_module=method_choice)
    sim.run()
    return sim


if __name__ == '__main__':
    if do_profile:
        sc.profile(run, to_profile)
    else:
        sc.tic()
        sim = run()
        sc.toc()
