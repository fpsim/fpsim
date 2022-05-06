"""
Run tests on the calibration object.
"""

import sciris as sc
import fpsim as fp
import fp_analyses as fa
import unittest
import os
import sys

class TestScenarios(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # suppresses unnecessary warning statements to increase runtime
        sys.stdout = open(os.devnull, 'w')
        pass

    def test_update_methods_eff(self):
        """
        Checks that fp.update_methods() properly updates sim.pars efficacies
        """
        low_eff = dict(dist='uniform', par1=0.80, par2=0.90)
        high_eff = dict(dist='uniform', par1=0.91, par2=0.95)

        pars_high_eff = fa.senegal_parameters.make_pars()
        pars_low_eff = fa.senegal_parameters.make_pars()

        scen_low_eff = sc.objdict(
            eff = {'Other modern':low_eff},
        )

        scen_high_eff = sc.objdict(
            eff = {'Other modern':high_eff}
        )

        low_eff = fp.update_methods(2005, scen_low_eff) 
        high_eff = fp.update_methods(2005, scen_high_eff)

        interventions = [low_eff, high_eff]

        simlist = []
        for index, pars in enumerate([pars_high_eff, pars_low_eff]):
            pars['n'] = 500
            pars.update(interventions=interventions[index])
            simlist.append(fp.Sim(pars=pars))

        msim = fp.MultiSim(sims=simlist)
        msim.run()

        low_eff_post_sim = msim.sims[0].pars['method_efficacy'][9]
        high_eff_post_sim = msim.sims[1].pars['method_efficacy'][9]

        self.assertGreater(high_eff_post_sim, low_eff_post_sim, msg=f"Method efficacy after updating to about .93 is {high_eff_post_sim} \
                                                                        and after updating to about 0.85 is actually {low_eff_post_sim}")

    def test_update_methods_probs(self):
        """
        Checks that fp.update_methods() function properly updates sim.pars for
        both the selected age keys, and the type (methods or postpartum_methods) of
        transition matrix
        """
        pars_no_keys_methods = fa.senegal_parameters.make_pars()
        pars_keys_methods = fa.senegal_parameters.make_pars()
        pars_no_keys_pp = fa.senegal_parameters.make_pars()
        pars_keys_pp = fa.senegal_parameters.make_pars()

        scen_no_keys = sc.objdict(
            probs = [
                dict(
                    source = 'None', # Source method, 'all' for all methods
                    dest   = 'Other modern', # Destination
                    factor = None, # Factor by which to multiply existing probability
                    value  = 0.2 # Alternatively, specify the absolute probability of switching to this method
                ),
            ]
        )
        scen_keys = sc.objdict(
            probs = [
                dict(
                    source = 'None', # Source method, 'all' for all methods
                    dest   = 'Other modern', # Destination
                    factor = None, # Factor by which to multiply existing probability
                    value  = 0.2, # Alternatively, specify the absolute probability of switching to this method
                    keys   = ['<18','18-20'], # Which age keys to modify -- if not specified, all
                ),
                dict(
                    source = 'Other modern', # Source method, 'all' for all methods
                    dest   = 'None', # Destination
                    factor = None, # Factor by which to multiply existing probability
                    value  = 0.8, # Alternatively, specify the absolute probability of switching to this method
                    keys   = ['<18','18-20'], # Which age keys to modify -- if not specified, all
                )
            ]
        )


        uptake_no_keys_methods = fp.update_methods(2005, scen_no_keys, matrix='probs_matrix') # Create intervention
        uptake_keys_methods = fp.update_methods(2005, scen_keys, matrix='probs_matrix') # Create intervention
        uptake_no_keys_pp = fp.update_methods(2005, scen_no_keys, matrix='probs_matrix_1-6') # Create intervention
        uptake_keys_pp = fp.update_methods(2005, scen_keys, matrix='probs_matrix_1-6') # Create intervention

        interventions = [uptake_no_keys_methods, uptake_keys_methods, uptake_no_keys_pp, uptake_keys_pp]

        simlist = []
        for index, pars in enumerate([pars_no_keys_methods, pars_keys_methods, pars_no_keys_pp, pars_keys_pp]):
            pars['n'] = 500
            pars.update(interventions=interventions[index])
            simlist.append(fp.Sim(pars=pars))
        
        msim = fp.MultiSim(sims=simlist)
        msim.run()

        none_index = msim.sims[0].pars['methods']['map']['None']
        other_modern_index = msim.sims[0].pars['methods']['map']['Other modern']


        self.assertNotEqual(msim.sims[0].pars['methods']['probs_matrix']['21-25'][none_index][other_modern_index], msim.sims[1].pars['methods']['probs_matrix']['21-25'][none_index][other_modern_index], "update_methods did not change contraceptive matrix for key 21-25")
        self.assertEqual(msim.sims[0].pars['methods']['probs_matrix']['21-25'][none_index][other_modern_index], 0.2, "update_methods did not change contraceptive matrix 21-25 to spcified 0.2")
        self.assertEqual(msim.sims[1].pars['methods']['probs_matrix']['<18'][none_index][other_modern_index], 0.2, "update_methods did not change contraceptive matrix <25 to spcified 0.2")

        self.assertNotEqual(msim.sims[2].pars['methods_postpartum']['probs_matrix_1-6']['21-25'][none_index][other_modern_index], msim.sims[3].pars['methods_postpartum']['probs_matrix_1-6']['21-25'][none_index][other_modern_index], "update_methods did not change postpartum contraceptive matrix for key 21-25")
        self.assertEqual(msim.sims[2].pars['methods_postpartum']['probs_matrix_1-6']['21-25'][none_index][other_modern_index], 0.2, "update_methods did not change postpartum contraceptive matrix for 21-25 to specified 0.2")
        self.assertEqual(msim.sims[3].pars['methods_postpartum']['probs_matrix_1-6']['<18'][none_index][other_modern_index], 0.2, "update_methods did not change postpartum contraceptive matrix for <18 to specified 0.2")

        none_switching_18 = msim.sims[3].pars['methods_postpartum']['probs_matrix_1-6']['<18'][other_modern_index][none_index]
        self.assertEqual(none_switching_18, 0.8, "After updating method switching postpartum for <18 for None to 0.8, actual value is {none_switching_18}")

        other_modern_25 = msim.sims[3].pars['methods_postpartum']['probs_matrix_1-6']['21-25'][other_modern_index][none_index]
        self.assertNotEqual(other_modern_25, 0.8, f"After updating method postpartum for 21-25, value is still 0.8")

if __name__ == '__main__':

    # run test suite
    unittest.main()