import numpy as np
import sciris as sc
import fpsim as fp
import fp_analyses as fa
import unittest
import json
import os
from copy import deepcopy
import pylab as pl
import sp_nomcpr

@unittest.skip("This test suite reveals issue #137")
class TestContraceptives(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.base_pars = sp_nomcpr.make_pars()
        self.base_pars['n'] = 500

        exp = fp.Experiment(self.base_pars)
        exp.run(keep_people = True)
        exp.to_json(filename="baseline_results.json")
    
    def setUp(self):
        self.debug_on = True
        self.pars = deepcopy(self.base_pars)

    def tearDown(self):
        if self.debug_on:
            self.add_reference_values()
        print("If this test had a large pause after finishing sim, turn off test_mode in model.py")

    # This function toggles a particular contraceptive method to zero
    # Within the given age groups list (such as ["<18"])
    def toggle_to_zero(self, method, age_classification=None):
        method_index = self.pars['methods']['map'][method]
        prob_matrix = self.pars['methods']["probs_matrix"]

        def toggle_index_off(method_array, method_index):
            added_value = method_array[method_index]
            new_list = deepcopy(method_array)
            new_list[0] += added_value
            new_list[method_index] = 0
            return new_list
        
        new_probs_matrix = {}
        # Switch the index element of each array in the dictionary (format  is age_key: array)
        for age_key, value in prob_matrix.items():
            if age_classification is None:
                new_2d_array = [0] * 10
                prob_arrays = self.pars['methods']["probs_matrix"][age_key]
                for index, prob_array in enumerate(prob_arrays): # adding prob of transitioning to method to transitioning to none, setting prob of transition to method to 0 for all methods
                    if index != method_index:
                        new_2d_array[index] = toggle_index_off(prob_array, method_index)
                    else:
                        new_method_list = [0] * 10
                        new_method_list [0] = 1.0
                        new_2d_array[index] = new_method_list
            new_probs_matrix[age_key] = pl.array(new_2d_array)
        self.pars['methods']["probs_matrix"] = new_probs_matrix     

    # This compares the given aggregate statistic between the baseline (without any contraceptive changes)
    # to the statistics given when the relevant contraceptive method is toggled off
    # Error margin relates to the proportional difference between baseline and test simulation statistics (delta of baseline * error_margin)
    # Target statistics is the list of statistic/s you want to compare
    def compare_to_baseline(self, error_margin=1.0, target_statistics=["pop_growth_rate_mean"]):

        with open('baseline_results.json') as baseline_file:
            baseline_dict = json.load(baseline_file) 

        with open(self.experiment_json) as target_file:
            target_dict = json.load(target_file)   
        
        for key, value in baseline_dict["model"].items():
            if key in target_statistics:
                self.assertAlmostEqual(baseline_dict["model"][key], target_dict["model"][key], 
                                        delta = error_margin * baseline_dict["model"][key], 
                                        msg=f"The value {key} in baseline is {baseline_dict['model'][key]} but in model it is {target_dict['model'][key]}, an unusually high difference")

        self.assertGreater(target_dict['model']['pop_size_mean'], baseline_dict['model']['pop_size_mean'])
        
    # This simply adds the baseline dictionary to the result 
    def add_reference_values(self):
        with open(self.experiment_json) as target_file:
            target_file = json.load(target_file) 

        with open('baseline_results.json') as baseline_file:
            baseline_dict = json.load(baseline_file) 

        target_file['baseline_model'] = baseline_dict['model']

        with open(self.experiment_json, "w") as output_file:
            json.dump(target_file, output_file)


    def toggle_and_run(self, test_name):
        self.toggle_to_zero(test_name)
        self.exp = fp.Experiment(self.pars)
        self.exp.run(keep_people = True)
        self.experiment_json = (f"{test_name}Test.json")
        self.exp.to_json(self.experiment_json)
        self.compare_to_baseline()

        if not self.debug_on:
            os.remove(self.experiment_json)
              
    def test_pills(self):
        self.toggle_and_run("Pill")

    def test_iuds(self):
        self.toggle_and_run("IUDs")

    def test_injectables(self):
        self.toggle_and_run("Injectables")

    def test_condoms(self):
        self.toggle_and_run("Condoms")

    @unittest.skip("This should fail since BTL not used")
    def test_BTL(self):
        self.toggle_and_run("BTL")

    def test_rythm(self):
        self.toggle_and_run("Rhythm")

    def test_withdrawl(self):
        self.toggle_and_run("Withdrawal")

    def test_implants(self):
        self.toggle_and_run("Implants")

    def test_other(self):
        self.toggle_and_run("Other")

    @unittest.skip("Type of transition setting isn't cumulative, this doesn't work for current implementation")
    def test_all(self):
        methods = ["None", "Pill", "IUDs", "Injectables", "Condoms", "BTL", "Rhythm", "Withdrawal", "Implants", "Other"]
        for method in methods:
            self.toggle_to_zero(method)

        self.exp = fp.Experiment(self.pars)
        self.exp.run(keep_people = True)
        self.experiment_json = "AllTest.json"
        self.exp.to_json(self.experiment_json)

        self.compare_to_baseline(error_margin=0.4)

        if not self.debug_on:
            os.remove(self.experiment_json)

@unittest.skip("Long, but all passes")
class TestContraceptiveEfficacy(unittest.TestCase):
    # Toggle off every method except for method specified
    @classmethod
    def setUpClass(self):
        self.pars = fa.senegal_parameters.make_pars()

    def all_but(self, method="BTL"):
        base_pars = deepcopy(self.pars)
        method_index = self.pars['methods']['map'][method]

        prob_dict = {}
        new_list = [0] * 10
        new_list[method_index] = 1.0
        list_list = [0] * 10
        for i in range(10):
            list_list[i] = new_list

        for age_group in ['<18', '18-20', '21-25', '>25']:
            prob_dict[age_group] = pl.array(list_list)

        self.pars['methods']['probs_matrix'] = prob_dict

        efficacy = sc.odict({
                "None":        0.0,
                "Pill":        0.0,
                "IUDs":        0.0,
                "Injectables": 0.0,
                "Condoms":     0.0,
                "BTL":         0.0,
                "Rhythm":       0.0,
                "Withdrawal":   0.0,
                "Implants":     0.0,
                "Other":       0.0,
                })

        last_result = 100000 # Sentinel
        for index, efficacy_value in enumerate([0, .3, .6, 0.9]):
            efficacy[method] = efficacy_value
            self.pars["method_efficacy"] = efficacy
            exp = fp.Experiment(self.pars)
            exp.run(keep_people = True)
            exp.to_json(filename=f"{method}_efficacy.json")

            with open(f"{method}_efficacy.json") as efficacy_file:
                efficacy_dict = json.load(efficacy_file) 
            
            self.assertGreater(last_result, efficacy_dict['model']["pop_growth_rate_mean"])
            if index == 3:
                print(self.pars["method_efficacy"])
                print(self.pars['methods']["probs_matrix"])
                self.assertLessEqual(efficacy_dict['model']["pop_growth_rate_mean"], 0)

        self.pars = base_pars

    def test_efficacy_none(self):
        self.all_but("None")

    def test_efficacy_pill(self):
        self.all_but("Pill")

    def test_efficacy_IUD(self):
        self.all_but("Pill")

    def test_efficacy_injectable(self):
        self.all_but("Injectables")

    def test_efficacy_condoms(self):
        self.all_but("Condoms")

    def test_efficacy_condoms(self):
        self.all_but("Rhythm")

    def test_efficacy_condoms(self):
        self.all_but("Withdrawal")

    def test_efficacy_condoms(self):
        self.all_but("Implants")

    def test_efficacy_condoms(self):
        self.all_but("Other")

    def test_efficacy_BTL(self):
        self.all_but("BTL")
