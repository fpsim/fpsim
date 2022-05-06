"""
Run tests on the calibration object.
"""

import sciris as sc
import fpsim as fp
import fp_analyses as fa
import unittest
import os
import sys

class TestInterventions(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        # suppresses unnecessary warning statements to increase runtime
        sys.stdout = open(os.devnull, 'w')
        pass

    def make_sim(self, n=500, **kwargs):
        '''
        Define a default simulation for testing the baseline.
        '''
        pars = fa.senegal_parameters.make_pars()
        pars['n'] = n
        pars['verbose'] = 0.1
        pars.update(kwargs)
        sim = fp.Sim(pars=pars)

        return sim


    def test_interventions(self):
        ''' Test interventions '''
        sc.heading('Testing interventions...')

        def test_interv(sim):
            if sim.i == 100:
                print(f'Success on day {sim.t}/{sim.y}')

        pars = dict(
            interventions = [test_interv],
        )

        sim = self.make_sim(**pars)
        sim.run()

        return sim


    def test_analyzers(self):
        ''' Test analyzers '''
        sc.heading('Testing analyzers...')

        pars = dict(
            analyzers = [fp.snapshot(timesteps=[100, 200])],
        )

        sim = self.make_sim(**pars)
        sim.run()

        return sim

if __name__ == '__main__':

    # run test suite
    unittest.main()
