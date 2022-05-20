'''
Benchmark the simulation
'''

import sciris as sc
import fpsim as fp
from fp_analyses import senegal_parameters as sp

pars = sp.make_pars()
exp = fp.Experiment(pars)
to_profile = 'extract_birth_spacing'

func_options = {'initialize':  exp.initialize,
                'extract_dhs_data': exp.extract_dhs_data,
                'run_model':   exp.run_model,
                'postprocess': exp.post_process_results,
                'extract_birth_spacing': exp.extract_birth_spacing,
                }

def run():
    pars = sp.make_pars()
    pars['n'] = 100
    exp = fp.Experiment(pars)
    exp.run()
    return


sc.profile(run, func_options[to_profile])
