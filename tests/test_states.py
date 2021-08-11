import unittest
import os
import pandas as pd
import numpy as np
import sciris as sc
import fpsim as fp
import fp_analyses as fa

@unittest.skip("Must run this with test_mode on in model.py, should not be run with other regression tests in GHA")
class TestStates(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        if os.path.exists("total_results.json"):
            os.remove("total_results.json")

        self.debug_mode = True
        pars = fa.senegal_parameters.make_pars()
        pars['n'] = 1000

        exp = fp.Experiment(pars)
        exp.run(keep_people = True)

        if not os.path.exists("total_results.json"):
            raise ValueError("You must enable test mode in model.py. Currently disabled by default due to speed difference")
        else:
            with open("total_results.json") as result_file:
                self.result_dict = sc.loadjson(result_file)

        self.year = None # Change to run the cross sectional tests on a specific year for the purpose of debugging specific reported values
    
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def find_indices(self, value_list, value):
        return [i for i, x in enumerate(value_list) if i == value]

    # Checks that no one is SA and pregnant
    # Check that lam coincides with breastfeeding
    # Everyone who is breastfeeding is lactating
    # Check everyone who is gestating is pregnant (gestation > 0)
    @unittest.skip("This reveals issue #113")
    def test_states_cross(self):

        sex_and_preg = 0
        preg_count = 0
        gestation = {}
        feed_lact = 0
        lam_lact = 0
        lact_lam = 0
        lact_total = 0
        lam_total = 0
        gest_not_preg = 0

        postpartum = {}

        postpartum_dur = {}
        
        for year, attribute_dict in self.result_dict.items():
            if self.year is None or year == self.year: # Checking for user input of year
                pregnant = attribute_dict['pregnant']
                breastfeeding = attribute_dict['breastfeed_dur']
                lactating = attribute_dict['lactating']
                lam = attribute_dict['lam']
                gestation_list = attribute_dict['gestation']
                postpartum[year] = attribute_dict['postpartum']
                postpartum_dur[year] = [attribute_dict['postpartum_dur'][index] for index in [i for i, x in enumerate(attribute_dict['postpartum']) if x]]
                # If you are looking at postpartum values and want to change what's considered "reduced birth" here you can alter that. Currently set to < 7
                postpartum_dur[year] = [attribute_dict['postpartum_dur'][index] for index in [i for i, x in enumerate(postpartum_dur[year]) if x < 7]]

                # These are all of the checks being made
                for index, preg_value in enumerate(pregnant):
                    if breastfeeding[index] > 0:
                        if not lactating[index]:
                            feed_lact = feed_lact + 1
                    if lam[index]:
                        lam_total = lam_total + 1
                        if not lactating[index]:
                            lam_lact = lam_lact + 1
                    if lactating[index]:
                        lact_total = lact_total + 1
                        if lam[index]:
                            lact_lam = lact_lam + 1
                    if preg_value:
                        preg_count = preg_count + 1
                        gestation[index] = gestation_list[index]
                    if gestation_list[index] > 0:
                        if not preg_value:
                            gest_not_preg = gest_not_preg + 1
                
        if self.debug_mode:
            sum_num = 0
            total_pop = 0
            average_lt6weeks = 0
            total_total_pop = 0

            for key, value in postpartum_dur.items():
                sum_num = sum_num + np.sum(postpartum_dur[key])
                total_pop = total_pop + len(postpartum_dur[key])

            for key, value in postpartum_dur.items():
                total_total_pop = total_total_pop + len(postpartum[key])
                average_lt6weeks = average_lt6weeks + len(postpartum_dur[key])

            print(f"On average, {average_lt6weeks / total_total_pop} people were affected by post partum birth reduction (as a proportion of the whole)") # Would want to use SA, but can be SA and pregnant
                
            print(f"Total pregnant: {preg_count}")
            print(f"Total breastfeeding and not lactating: {feed_lact}")
            print(f"Total on LAM and not lactating: {lam_lact} out of {lam_total} on lam")
            print(f"Total lactating and are on LAM: {lact_lam} out of {lact_total} lactating")
            print(f"Total that are gestating but not pregnant: {gest_not_preg}")

        descriptions = {0: "were breastfeeding while not lactating", 1: "were on LAM while not lactating", 2: "were gestating while not pregnant"}
        for index, count_check in enumerate([feed_lact, lam_lact, gest_not_preg]):
            self.assertEqual(count_check, 0, msg=f"{count_check} {descriptions[index]}")

    # Checking that no dead people are updating any parameters
    # dictionary with index: array with values of single person over time so we have dict[person][day]
    def test_states_long_dead(self):
        alive_recorder = {}
        gestation_dur = {}
        breastfeed_dur = {}

        for year, attribute_dict in self.result_dict.items():
           # take the array add each attribute[i] into 
            for index, value in enumerate(attribute_dict["alive"]):
                if index not in alive_recorder:
                    alive_recorder[index] = []
                    gestation_dur[index] = []
                    breastfeed_dur[index] = []
                alive_recorder[index].append(value)
                gestation_dur[index].append(attribute_dict["gestation"][index])
                breastfeed_dur[index].append(attribute_dict["breastfeed_dur"][index])

        print("Checking alive against updated parameter")
        i = 1 # comparing to previous value so must be one
        while (i < len(gestation_dur)):
            prec_gestation = 100 # value of gestation on the previous day
            prec_breastfeed = 100

            for index, compared_value in enumerate(gestation_dur[i]):
                if prec_gestation < compared_value:
                    # print(f"{i} is the index of the person (key in dict)")
                    # print(f"{index} is the index of the compared value")
                    # print(f"{prec_gestation} is the previous gestation and {compared_value} is current.")
                    self.assertEqual(alive_recorder[i][index-1], True, msg="At [{i}, {index}] a person's pregnancy is progressing while they are dead")
                if prec_breastfeed < breastfeed_dur[i][index]:
                    self.assertEqual(alive_recorder[i][index-1], True, msg="At [{i}, {index}] a person is breastfeeding while they are dead")

            i = i + 1

    # Nobody should have their pregnancy reset after month 4 (unless they give birth at month 9)
    @unittest.skip("This reveals issue 117")
    def test_states_long_gestation_reset(self):
        gestation_dur = {}
        # Getting everyone's total gestation history
        for year, attribute_dict in self.result_dict.items():
            for index, value in enumerate(attribute_dict["gestation"]): 
                if index not in gestation_dur:
                    gestation_dur[index] = []
                if attribute_dict["alive"][index]:
                    gestation_dur[index].append(value)
            if year == "2019.0":
                # count pregnant people at end 
                print(f"At the end of the sim there are: {sum(attribute_dict['pregnant'])} pregnant people" )
               
        
        for person, gest_history in gestation_dur.items():
            last = -5
            for index, current_dur in enumerate(gest_history):
                if last in range(4, 9):
                    self.assertEqual(current_dur, last + 1, msg=f"Person at index {person} has gestation history: {gest_history} which is faulty at index {index}")
                if last == 9:
                    self.assertEqual(current_dur, 0, msg=f"Person at index {person} has gestation history: {gest_history} which is faulty at index {index}")
                last = current_dur

    # Checking that lactation, gestation, postpartum do not preclude pregnancy
    def test_states_long_pre_pregnancy(self):
        was_pregnant = {}
        for year, attribute_dict in self.result_dict.items():
            for index, value in enumerate(attribute_dict["pregnant"]):
                if index not in was_pregnant or value:
                    was_pregnant[index] = value
                # If it's false, check if it's
                gestation = attribute_dict["gestation"][index]
                lactating = attribute_dict["lactating"][index]
                postpartum = attribute_dict["postpartum"][index]
                if (gestation > 0 or lactating or postpartum > 0):
                    self.assertTrue(was_pregnant[index], msg=f"In year {year} there was a person whose gestation is {gestation} lactation is {lactating} and postpartum is {postpartum} and their was_pregnant status is {was_pregnant[index]}")                
