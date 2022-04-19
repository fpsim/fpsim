import fpsim as fp
import fp_analyses as fa
import unittest
import json
from copy import deepcopy
import pylab as pl
import numpy as np
import pytest

@unittest.skip("Need to optimize with multisim before it can be in GHA")
@pytest.mark.long
class TestContraceptiveEfficacy(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.pars = fa.senegal_parameters.make_pars()
        self.pars['n'] = 500
        self.contraceptives =  [
            "None",
            "Pill",
            "IUDs",
            "Injectable",
            "Condoms",
            "BTL",
            "Rhythm",
            "Withdrawal",
            "Implants",
            "Other"
        ]

    def all_but(self, method="BTL"):
        """
        Toggles off every method except for method specified
        Inputs:
            method::str
                method name, for example 'BTL'
        """
        base_pars = deepcopy(self.pars)
        method_index = self.pars['methods']['map'][method]

        prob_dict = {}
        list_list = [0] * 10
        for i in range(10):
            new_list = [0] * 10
            new_list[method_index] = 1.0
            list_list[i] = new_list

        for age_group in ['<18', '18-20', '21-25', '>25']:
            prob_dict[age_group] = pl.array(list_list)

        self.pars['methods']['probs_matrix'] = prob_dict

        efficacy = [0] * 10
        sims = []
        last_result = 100000 # Sentinel
        for efficacy_value in [0, .3, 0.6, 1.0]:
            efficacy[method_index] = efficacy_value
            self.pars["method_efficacy"] = np.array(efficacy)
            sim = fp.SimVerbose(self.pars)
            sims.append(sim)
        
        multi = fp.MultiSim(sims=sims)
        multi.run()
        for sim in multi.sims:
            self.assertGreater(last_result, sim['results']['pop_size'][-1])
            last_result = sim['results']['pop_size'][-1]

        self.pars = base_pars

    def test_efficacy_none(self):
        self.all_but("None")

    def test_efficacy_pill(self):
        self.all_but("Pill")

    def test_efficacy_IUD(self):
        self.all_but("IUDs")

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
