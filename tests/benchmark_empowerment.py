'''
Benchmark how much longer empowerment-related functions take.
Education adds 7% to the runtime of a simulation.
'''

import sciris as sc
import fpsim as fp

do_profile = 1
sim = fp.Sim()
sim.initialize()

ppl = sim.people
to_profile = sc.objdict(
    run =         sim.run,
    ppl_update    =  ppl.update,
    ppl_init      =  ppl.__init__,
    ppl_education =  ppl.update_education,
    edu_start     =  ppl.start_education,      # 10% of update_eduaction()
    edu_advance   =  ppl.advance_education,    # 63% of update_eduaction()
    edu_interrupt =  ppl.interrupt_education,  # 22% of advance_education()
    edu_dropout   =  ppl.dropout_education,    # 40% of advance_education() -- called twice
    edu_resume    =  ppl.resume_education,     # 19% of update_education()
    edu_done      =  ppl.graduate,             #  8% of update_education()
)


def empowerment_pars():
    ''' Additional empowerment parameters'''
    empwrmnt_pars = dict(
        urban_prop      = None,
        empowerment     = None,
        education       = None,
        age_partnership = None,
    )

    return empwrmnt_pars


def run_with_empowerment():
    pars = fp.pars(location='kenya', n_agents=10e1, method_timestep=1, verbose=0)

    sim = fp.Sim(pars)
    sim.run()
    return sim


def run_without_empowerment():
    pars = fp.pars(location='kenya', n_agents=10e1, method_timestep=1, verbose=0)
    # Overwrite empowerment parameters
    pars.update(empowerment_pars())
    sim = fp.Sim(pars)
    sim.run()
    return sim


if __name__ == '__main__':
    selection = 'ppl_education'
    if do_profile:
        sc.profile(run_with_empowerment, to_profile[selection])
        sc.profile(run_without_empowerment, to_profile[selection])
    else:
        sc.tic()
        sim = run_with_empowerment()
        sc.toc()
