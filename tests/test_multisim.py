"""
Run tests on the multisim object.
"""

import sciris as sc
import fpsim as fp
import fp_analyses as fa
import sys
import os
import unittest
import pytest

class TestMultisim(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.do_plot = False
    
    def test_multisim(self):
        ''' Try running a multisim '''
        sc.heading('Testing multisim...')

        # Generate sims in a loop
        sims = []
        for i in range(3):
            exposure = 0.5 + 0.5*i # Run a sweep over exposure
            pars = fa.senegal_parameters.make_pars()
            pars['n'] = 500
            pars['verbose'] = 0.1
            pars['exposure_correction'] = exposure
            sim = fp.Sim(pars=pars, label=f'Exposure {exposure}')
            sims.append(sim)

        msim = fp.MultiSim(sims)
        msim.run() # Run sims in parallel
        msim.to_df() # Test to_df

        births = msim.results.births
        self.assertGreater(sum(births.high), sum(births.low), 'Expected the higher bound of births to be higher than the lower bound')

        if self.do_plot:
            msim.plot(plot_sims=True)
            msim.plot(plot_sims=False)

if __name__ == '__main__':

    # suppresses unnecessary warning statements to increase runtime
    sys.stdout = open(os.devnull, 'w')

    # run test suite
    unittest.main()