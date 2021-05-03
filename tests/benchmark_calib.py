'''
Benchmark the simulation
'''

import sciris as sc
import fpsim as fp
from fp_analyses import senegal_parameters as sp

pars = sp.make_pars()
calib = fp.Calibration(pars)
to_profile = 'extract_birth_spacing'

func_options = {'initialize':  calib.initialize,
                'extract_dhs_data': calib.extract_dhs_data,
                'run_model':   calib.run_model,
                'postprocess': calib.post_process_results,
                'extract_birth_spacing': calib.extract_birth_spacing,
                }

def run():
    pars = sp.make_pars()
    pars['n'] = 100
    calib = fp.Calibration(pars)
    calib.run()
    return


sc.profile(run, func_options[to_profile])
