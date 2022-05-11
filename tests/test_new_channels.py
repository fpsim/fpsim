'''
Test births, conceptions, etc.
'''

import fpsim as fp
import unittest

class TestChannels(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.pars = fp.pars('test', n=500, end_year=2020) # CK: TODO: check why this test fails for small n
        exp = fp.ExperimentVerbose(self.pars)
        exp.run_model()
        self.exp = exp
        self.total_results = exp.total_results
        self.events = exp.events
        self.channels = ["Births", "Conceptions", "Miscarriages", "Sexual_Debut", "Deaths"]
        return

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

                self.assertGreater(max, 0, msg=f"Detected empty channel: {channel}")

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
        self.assertEqual(births, births_step, "Mismatch between step results births and births channel")

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
        self.assertAlmostEqual(births, conceptions, delta = 0.25 * births, msg="Less than 75 percent of conceptions result in births")
        self.assertGreater(conceptions, births, msg="Number of conceptions not greater than recorded live births")

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

        self.assertGreater(conceptions - births, miscarriages, msg="The number of miscarriages is greater than the differences between conceptions and births")

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
                    self.assertTrue(item not in sexually_active, msg=f"Record of individual at timestep {timestep} sexual debuted but not sexually active")
                    self.assertTrue(self.total_results[timestep]['sexually_active'][item], msg=f"Sexual debut but not sexually active at timestep {timestep} index {item}")
                sexually_active.update(self.events[timestep]['Sexual_Debut'])

if __name__ == '__main__':

    # run test suite
    unittest.main()
