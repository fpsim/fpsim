'''
Benchmark how much longer education-related functions take.
Education adds 7% to the runtime of a simulation.
'''

import sciris as sc
import fpsim as fp
import fpsim.education as fpedu


do_profile = 0
sim = fp.Sim()
sim.initialize()

ppl = sim.people
to_profile = sc.objdict(
    run                 =   sim.run,
    ppl_update    =  ppl.update,
    ppl_init      =  ppl.__init__,
    education =  fpedu.update_education,
    edu_start     =  fpedu.start_education,      # 12% of update_education()
    edu_advance   =  fpedu.advance_education,    # 56% of update_education()
    edu_interrupt =  fpedu.interrupt_education,  # 19% of advance_education()
    edu_dropout   =  fpedu.dropout_education,    # 30% of advance_education() -- called twice
    edu_resume    =  fpedu.resume_education,     # 16% of update_education()
    edu_done      =  fpedu.graduate,             # 14% of update_education()
)


def run_with_education():
    pars = fp.pars(location='kenya',
                   n_agents=10e1,
                   method_timestep=1,
                   verbose=0,
                   use_education=True)
    sim = fp.Sim(pars)
    sim.run()
    return sim


def run_without_education():
    pars = fp.pars(location='kenya',
                   n_agents=10e1,
                   method_timestep=1,
                   verbose=0,
                   use_education=False)
    sim = fp.Sim(pars)
    sim.run()
    return sim


if __name__ == '__main__':
    selection = 'education'
    if do_profile:
        sc.profile(run_with_education, to_profile[selection])
        sc.profile(run_without_education, to_profile[selection])
    else:
        sc.tic()
        sim = run_with_education()
        sc.toc()
