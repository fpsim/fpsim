'''
Ensure that basic analyses at least, like, run. Adapted from covasim/tests/test_examples.py.
'''

import sys
import pylab as pl
import sciris as sc
from pathlib import Path
import importlib.util as iu
import pytest
import unittest
import os

class TestAnalyses(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        pl.switch_backend('agg') # To avoid graphs from appearing -- if you want them, run the examples directly
        cwd = Path(sc.thisdir(__file__))
        self.analyses_dir = cwd.joinpath('../fp_analyses')
        sys.path.append(str(self.analyses_dir)) # To enable imports to work

        # suppresses unnecessary warning statements to increase runtime
        sys.stdout = open(os.devnull, 'w')

    def run_script(self, name):
        '''
        Execute a py script as __main__

        Args:
            name (str): the filename without the .py extension
        '''
        spec = iu.spec_from_file_location("__main__", self.analyses_dir.joinpath(f"{name}.py"))
        module = iu.module_from_spec(spec)
        spec.loader.exec_module(module)
        return


    def test_run_senegal(self):
        self.run_script("run_senegal")
        return


    def test_run_experiment(self):
        self.run_script("run_experiment")
        return

    #%% Run as a script
if __name__ == '__main__':

    # suppresses unnecessary warning statements to increase runtime
    sys.stdout = open(os.devnull, 'w')

    # run test suite
    unittest.main()
