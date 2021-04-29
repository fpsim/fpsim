'''
Save or test the regression. To save, run
'''

import numpy as np
import sciris as sc
import fpsim as fp
import fp_analyses as fa

do_plot = 0
do_save = 0
baseline_filename  = sc.thisdir(__file__, 'baseline.json')
benchmark_filename = sc.thisdir(__file__, 'benchmark.json')

def run(do_save=False):
    pars = fa.senegal_parameters.make_pars()
    pars['n'] = 500
    calib = fp.Calibration(pars=pars)
    calib.run()
    json = sc.jsonify(calib.model_to_calib)
    if do_save:
        sc.savejson(filename, json)
    return calib, json


if __name__ == '__main__':
    calib, json = run()
