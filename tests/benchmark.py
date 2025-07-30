'''
Benchmark the simulation
'''

import sciris as sc
import fpsim as fp

do_profile = 1
location = 'kenya'
pars = dict(location=location, n_agents=500, start_year=2000, end_year=2010, seed=1, verbose=1/12)


def make_sim():
    method_choice = fp.SimpleChoice(location=location)
    sim = fp.Sim(pars, contraception_module=method_choice)
    sim.init()
    return sim, method_choice


def run_sim():
    sim, _ = make_sim()
    sim.run()
    return sim


if __name__ == '__main__':

    sim, method_choice = make_sim()
    ppl = sim.people
    to_profile = sc.objdict(
        run             = sim.run,
        step            = ppl.step,           # 70% of sim.run() runtime is spent here
        people_init     = ppl.__init__,
        update_method   = ppl.update_method,    # 95% of ppl.update_methods() runtime is spent here
        set_dur_method  = method_choice.set_dur_method,    # 98% of ppl.update_methods() runtime is spent here
    )['set_dur_method']

    if do_profile:
        sc.profile(run_sim, to_profile)
    else:
        sc.tic()
        sim = run_sim()
        sc.toc()
