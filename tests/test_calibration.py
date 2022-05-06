"""
Run tests on the calibration object.
"""

import sciris as sc
import fpsim as fp
import fp_analyses as fa
import unittest
import pytest
import sys
import os


class TestCalibration(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.do_plot = 0
        self.n_trials = 3
        
        # suppresses unnecessary print statements to increase runtime
        sys.stdout = open(os.devnull, 'w')

    def make_calib(self, n=50):
        '''
        Define a default simulation for testing the baseline.
        '''
        pars = fa.senegal_parameters.make_pars()
        pars['n'] = n
        pars['verbose'] = 0
        calib = fp.Calibration(pars=pars)

        return calib


    def test_calibration(self):
        ''' Compare the current default sim against the saved baseline '''
        sc.heading('Testing calibration...')

        calib_pars = dict(
            exposure_correction = [1.5, 0.7, 1.5],
        )

        # Calculate calibration
        calib = self.make_calib()
        calib.calibrate(calib_pars=calib_pars, n_trials=self.n_trials, n_workers=2)
        before,after = calib.summarize()

        # assert before > after

        if self.do_plot:
            calib.before.plot()
            calib.after.plot()
            calib.before.fit.plot()
            calib.after.fit.plot()

        return calib

if __name__ == '__main__':

    # run test suite
    unittest.main()