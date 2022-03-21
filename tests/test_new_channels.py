from collections import defaultdict
import sys
import argparse
import os
import pandas as pd
import numpy as np
import fpsim as fp
import fp_analyses as fa
import unittest

class TestChannels(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.pars = fa.senegal_parameters.make_pars()
        self.pars['n'] = 1000
        exp = fp.ExperimentVerbose(self.pars)
        exp.run_model()
        self.exp = exp
        self.total_results = exp.total_results
        self.events = exp.events
        self.channels = ["Births", "Conceptions", "Miscarriages", "Sexual_Debut", "Deaths"]

    def test_channels_sanity_check(self):
        """
        Checks that none of the channels from self.channels contain no entries.
        """
        for channel in self.channels:
            if channel != "Deaths":
                max = 0
                for timestep in self.events:
                    if len(self.events[timestep][channel]) > max:
                        max = len(self.events[timestep][channel])

                self.assertGreater(max, 0)
    
    def test_births(self):
        """
        Checks that births (formatted as timestep: [indices]) is consistent with
        the births aggregate value from step results (formatted as timestep: total).
        """
        births = 0
        births_step = 0
        for timestep in self.events:
            births_step += self.events[timestep]["Step_Results"]["births"]
            births += len(self.events[timestep]['Births'])
        self.assertEqual(births, births_step)
    
    def test_conceptions(self):
        """
        Checks that conceptions is approximately births, and that conceptions is greater
        than the number of births.
        """
        births = 0
        conceptions = 0
        for timestep in self.events:
            births = births + len(self.events[timestep]['Births']) 
            conceptions = conceptions + len(self.events[timestep]['Conceptions']) 

        # We wouldn't expect more than a quarter of conceptions to end in miscarriages
        self.assertAlmostEqual(births, conceptions, delta = 0.25 * births)
        self.assertGreater(conceptions, births)

    def test_miscarriages(self):
        """
        Checks that miscarriages < difference between conceptions and births
        """
        births = 0
        conceptions = 0
        miscarriages = 0
        for timestep in self.events:
            births += len(self.events[timestep]['Births']) 
            conceptions += len(self.events[timestep]['Conceptions']) 
            miscarriages += len(self.events[timestep]['Miscarriages']) 

        self.assertGreater(conceptions - births, miscarriages)

    @unittest.skip("Need to verify this works over multiple runs")
    def test_sexual_debut(self):
        """
        Checks that a person is SA at their sexual debut, 
        and doesn't show up twice in the log of sexual debuts
        """
        sexually_active = set()
        for timestep in self.events:
            if timestep != 1960.0:
                for item in self.events[timestep]['Sexual_Debut']:
                    self.assertTrue(item not in sexually_active)
                    self.assertTrue(self.total_results[timestep]['sexually_active'][item], msg=f"Inconsistency at timestep {timestep} index {item}")
                sexually_active.update(self.events[timestep]['Sexual_Debut'])

if __name__ == '__main__':
    unittest.main()
