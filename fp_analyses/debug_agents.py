import sciris as sc
import fpsim as fp
import fp_analyses as fa
import pandas as pd

n = 100000

pars = fa.senegal_parameters.make_pars()
pars['n'] = n
pars['start_year'] = 1990
sim = fp.Sim(pars)
sim.run()

agents = {}

ppl = sim.people
inds = sc.findinds(ppl.alive * (ppl.age >= 50) * (ppl.pregnant))

agents['uid'] = ppl.uid[inds]
agents['alive?'] = ppl.alive[inds]
agents['age'] = ppl.age[inds]
#agents['male?'] = ppl.sex[inds]
agents['parity'] = ppl.parity[inds]
agents['method'] = ppl.method[inds]
agents['sexually_active'] = ppl.sexually_active[inds]
agents['pregnant?'] = ppl.pregnant[inds]

df_preg_over_50 = pd.DataFrame(data=agents)

df_preg_over_50.to_csv('/Users/Annie/model_postprocess_files/agent_states.csv')