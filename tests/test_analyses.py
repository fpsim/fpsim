'''
Ensure that basic analyses at least, like, run. Adapted from covasim/tests/test_examples.py.
'''

import sys
import pylab as pl
import sciris as sc
from pathlib import Path
import importlib.util as iu
import pytest

pl.switch_backend('agg') # To avoid graphs from appearing -- if you want them, run the examples directly
cwd = Path(sc.thisdir(__file__))
analyses_dir = cwd.joinpath('../fp_analyses')
sys.path.append(str(analyses_dir)) # To enable imports to work


def run_script(name):
    '''
    Execute a py script as __main__

    Args:
        name (str): the filename without the .py extension
    '''
    spec = iu.spec_from_file_location("__main__", analyses_dir.joinpath(f"{name}.py"))
    module = iu.module_from_spec(spec)
    spec.loader.exec_module(module)
    return


# @pytest.mark.skip('Too long to run as part of continuous integration')
def test_run_senegal():
    run_script("run_senegal")
    return


def test_run_experiment():
    run_script("run_experiment")
    return


#%% Run as a script
if __name__ == '__main__':

    T = sc.tic()

    test_run_senegal()
    test_run_experiment()

    sc.toc(T)
    print('Done.')
