"""
Simplest possible FPsim test.
"""

import sciris as sc
import fpsim as fp
import fp_analyses as fa
import unittest
import os
import sys
import pytest

class TestSimple(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # suppresses unnecessary warning statements to increase runtime
        #sys.stdout = open(os.devnull, 'w')

        self.do_plot = False
        self.n = 100

    def test_simple(self):
        '''
        Define a default simulation for testing the baseline.
        '''
        pars = fa.senegal_parameters.make_pars()
        pars['n'] = self.n
        sim = fp.Sim(pars)
        sim.run()

        if self.do_plot:
            sim.plot()

        return sim

if __name__ == '__main__':

    # run test suite
    unittest.main()