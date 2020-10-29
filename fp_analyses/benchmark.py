'''
Benchmark the simulation
'''

import sciris as sc
import lemod_fp as lfp
import senegal_parameters as sp

pars = sp.make_pars()
sim = lfp.Sim(pars)
person = sc.odict(sim.people)[0]
to_profile = 'get_method'

func_options = {'sim':        sim.run,
                'update':     person.update,
                'preg':       person.get_preg_prob,
                'get_method': person.get_method,
                }

sc.profile(run=sim.run, follow=func_options[to_profile])
