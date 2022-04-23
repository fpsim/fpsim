"""
Run tests on individual parameters.
"""

import sciris as sc
import fpsim as fp
import fp_analyses as fa
import unittest
import pytest
import os
import sys


class TestParameters(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.do_plot=False
        
        # suppresses unnecessary warning statements to increase runtime
        sys.stdout = open(os.devnull, 'w')

    def get_pars(self):
        return fa.senegal_parameters.make_pars()

    def make(self, pars, n=1000, verbose=0, do_run=True, **kwargs):
        '''
        Define a default simulation for testing the baseline.
        '''

        pars['n'] = n
        pars['verbose'] = verbose
        pars.update(kwargs)
        sim = fp.Sim(pars=pars)

        if do_run:
            sim.run()

        return sim


    def test_null(self):
        sc.heading('Testing no births, no deaths...')

        pars = self.get_pars() # For default pars
        pars['age_mortality']['f'] *= 0
        pars['age_mortality']['m'] *= 0
        pars['age_mortality']['trend'] *= 0
        pars['maternal_mortality']['probs'] *= 0
        pars['infant_mortality']['probs'] *= 0
        pars['exposure_correction'] = 0
        pars['high_parity']         = 4
        pars['high_parity_nonuse_correction']  = 0

        sim = self.make(pars)

        if self.do_plot:
            sim.plot()

        return sim

if __name__ == '__main__':

    # run test suite
    unittest.main()