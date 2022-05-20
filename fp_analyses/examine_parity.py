'''A script to pull out agents by parity and method use from the model for postprocessing'''

import pandas as pd
import sciris as sc
import fpsim as fp
import senegal_parameters as sp

do_run = 1
do_save = 1

min_age = 15
max_age = 50

if do_run:
     pars = sp.make_pars()
     sim = fp.Sim(pars=pars)
     sim.run()

     agents = {}

     ppl = sim.people
     inds = sc.findinds(ppl.alive * ~ppl.sex * (ppl.age >= min_age) * (ppl.age < max_age))
     agents['parity'] = ppl.parity[inds]
     agents['method'] = ppl.method[inds]

     df = pd.DataFrame(data=agents)

     if do_save:
          df.to_csv('model_files/parity_methods_model.csv')