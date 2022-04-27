import json
from collections import defaultdict
import unittest
import os
import pandas as pd
import numpy as np
import sciris as sc
import fpsim as fp
import fp_analyses as fa

class TestStates(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        """
        Sets up test class by storing useful variables ahead of running unittests
        Configurable:
            self.debug_mode::bool:
                Prints useful sim run information from crossectional tests
            self.year::float:
                Limits crossectional tests to only year specified                
        """
        if os.path.exists("total_results.json"):
            os.remove("total_results.json")

        self.debug_mode = True
        pars = fa.senegal_parameters.make_pars()
        pars['n'] = 1000


        self.exp = fp.ExperimentVerbose(pars)
        self.exp.run_model(mother_ids=True)

        self.people = self.exp.people
        self.result_dict = self.exp.total_results
        self.year = None # change to make cross sectional tests apply to specific year
    
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def save_person_states(self, person_id, filename):
        """
        Saves a json of the states of the given individual, typically for the sake
        of debugging a failed assertion.

        Inputs:
            person_id::int
                Index of the person logged
            filename::str
                Name of the file where the output is logged

        Outputs:
            json file in the debug/ directory
        """
        person_dict = defaultdict(list)
        for timestep, attribute_dict in self.result_dict.items():
            for state in attribute_dict:
                if state != "dobs":
                    person_dict[state].append(int(attribute_dict[state][person_id]))

        if not os.path.exists("debug"):
            os.mkdir("debug")
        with open(filename, 'w') as output_file:
            json.dump(person_dict, output_file)

        print(f"Saved debug file at {filename}")

    def find_indices(self, value_list, value):
        """
        Returns indices of value given in list
        Input:
            value_list::list:
                List of values from which to find the indices value
            value::any:
                Value being queried from list, can be anything able to be equated with '=='
        """
        return [i for i, x in enumerate(value_list) if x == value]

    @unittest.skip("This reveals issue #113")
    def test_states_cross(self):
        """
        Checks that:
            no one is SA and pregnant
            lam coincides with breastfeeding
            everyone who is breastfeeding is lactating
            everyone who is gestating is pregnant (gestation > 0)
        """
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

            for year, value in postpartum_dur.items():
                sum_num = sum_num + np.sum(value)
                total_pop = total_pop + len(value)

            for year, value in postpartum_dur.items():
                total_total_pop = total_total_pop + len(value)
                average_lt6weeks = average_lt6weeks + len(value)

            print(f"On average, {average_lt6weeks / total_total_pop} people were affected by post partum birth reduction (as a proportion of the whole)") # Would want to use SA, but can be SA and pregnant
                
            print(f"Total pregnant: {preg_count}")
            print(f"Total breastfeeding and not lactating: {feed_lact}")
            print(f"Total on LAM and not lactating: {lam_lact} out of {lam_total} on lam")
            print(f"Total lactating and are on LAM: {lact_lam} out of {lact_total} lactating")
            print(f"Total that are gestating but not pregnant: {gest_not_preg}")

        descriptions = {0: "were breastfeeding while not lactating", 1: "were on LAM while not lactating", 2: "were gestating while not pregnant"}
        for index, count_check in enumerate([feed_lact, lam_lact, gest_not_preg]):
            if count_check != 0:
                self.save_person_states(index, "debug/cross_sectional_error.json")
                self.assertEqual(count_check, 0, msg=f"{count_check} {descriptions[index]}")

    def test_states_long_dead(self):
        """
        Checking that no dead people are updating any parameters, specifically gestation and breastfeeding
        """
        alive_recorder = {}
        gestation_dur = {}
        breastfeed_dur = {}

        for year, attribute_dict in self.result_dict.items():
            for index, value in enumerate(attribute_dict["alive"]):
                if index not in alive_recorder:
                    alive_recorder[index] = []
                    gestation_dur[index] = []
                    breastfeed_dur[index] = []
                alive_recorder[index].append(value)
                gestation_dur[index].append(attribute_dict["gestation"][index])
                breastfeed_dur[index].append(attribute_dict["breastfeed_dur"][index])

        prec_gestation = 100
        prec_breastfeed = 100
        for person in gestation_dur:
            for index, compared_value in enumerate(gestation_dur[person]):
                if prec_gestation < compared_value and not alive_recorder[person][index-1]:
                    self.save_person_states(index, "debug/dead_pregnancy_error.json")
                    self.assertTrue(alive_recorder[person][index-1], msg="At [{i}, {index}] a person's pregnancy is progressing while they are dead")
                if prec_breastfeed < breastfeed_dur[person][index] and not alive_recorder[person][index-1]:
                    self.save_person_states(index, "debug/dead_breastfeed_error.json")
                    self.assertTrue(alive_recorder[person][index-1], msg="At [{i}, {index}] a person is breastfeeding while they are dead")

    @unittest.skip("This reveals issue 117")
    def test_states_long_gestation_reset(self):
        """
        Checks that:
            nobody has their pregnancy reset after month 4 (unless they give birth at month 9)
            pregnancies increase after 4 months until completion
        """
        gestation_dur = {}
        for year, attribute_dict in self.result_dict.items():
            for index, value in enumerate(attribute_dict["gestation"]): 
                if index not in gestation_dur:
                    gestation_dur[index] = []
                if attribute_dict["alive"][index]:
                    gestation_dur[index].append(value)

        for person, gest_history in gestation_dur.items():
            last = -5
            for index, current_dur in enumerate(gest_history):   
                if last in range(4, 9) and current_dur != last + 1:
                    self.save_person_states(person, "debug/pregnancy_error.json")
                    self.assertEqual(current_dur, last + 1, msg=f"Person at index {person} has gestation history: {gest_history} which is faulty at index {index}")
                    break
                    
                if last == 9 and current_dur not in [0, 1]:
                    self.save_person_states(person, "debug/end_pregnancy_error.json")
                    self.assertTrue(current_dur in [0, 1], msg=f"Person at index {person} has gestation history: {gest_history} which is faulty at index {index}")
                    break
                last = current_dur

    def test_states_long_pre_pregnancy(self):
        """
        Checks that lactation, gestation, postpartum do not preclude pregnancy
        """
        was_pregnant = {}
        for year, attribute_dict in self.result_dict.items():
            for index, value in enumerate(attribute_dict["pregnant"]):
                if index not in was_pregnant or value:
                    was_pregnant[index] = value

                gestation = attribute_dict["gestation"][index]
                lactating = attribute_dict["lactating"][index]
                postpartum = attribute_dict["postpartum"][index]
                if (gestation > 0 or lactating or postpartum > 0) and not was_pregnant[index]:
                    self.save_person_states(index, "debug/preclude_pregnancy_error.json")
                    self.assertTrue(was_pregnant[index], msg=f"In year {year} there was a person whose gestation is {gestation} lactation is {lactating} and postpartum is {postpartum} and their was_pregnant status is {was_pregnant[index]}")   

    @unittest.skip("Mothers needs to be configured on for this to work")
    def test_mothers_indices(self):
        mothers = self.people.mothers
        children = self.people.children

        flattened_children = [item for sublist in children for item in sublist]
        self.assertGreater(len(mothers), len(flattened_children))

        for mother_index, child_list in enumerate(children):
            for child in child_list:
                self.assertEqual(mothers[child], mother_index)

    def test_first_birth_age(self):
        last_year_first_age = []
        for year, attribute_dict in self.result_dict.items():
            # dobs and age at first birth should be consistent (specifically checks that mothers represented in first_birth_age is subset of those represented in dobs)
            for index, age_first_birth in enumerate(attribute_dict['first_birth_age']):
                if age_first_birth is not None:
                    self.assertGreater(len(attribute_dict['dobs'][index]), 0, "Some people are shown as having given birth in first_birth_age but not in dobs")

            # once assigned, age at first birth should stay the same
            for index, last_year_age_first_birth in enumerate(last_year_first_age):
                if last_year_age_first_birth is not None:
                    self.assertEqual(last_year_age_first_birth, attribute_dict['first_birth_age'][index])

            last_year_first_age = attribute_dict['first_birth_age']

    def test_age_boundaries(self):
        """
        Checks that people under 13 or over 40 can't get pregnant
        """
        for year, attribute_dict in self.result_dict.items():
            for index, pregnant_bool in enumerate(attribute_dict["pregnant"]):
                age = attribute_dict['age'][index]
                if age < 11 or age > 99:
                    self.assertFalse(pregnant_bool, msg=f"Individual {index} can't be pregnant she's {age}")

    @unittest.skip("Reveals issue #305")
    def test_ages(self):
        for year, attribute_dict in self.result_dict.items():
            if year in sorted(self.result_dict.keys())[-10:]:
                for individual, age in enumerate(attribute_dict['age']):
                    age_year = int(age)
                    month = (age - age_year)
                    self.assertAlmostEqual(month * 12, round(month * 12), delta=0.01, msg=f"Individual at index: {individual} in year {year} has an age of {age} with a month ({month}) that is not a multiple of 1/12. month * 12 = {month * 12}")
            
                    
if __name__ == "__main__":
    unittest.main()