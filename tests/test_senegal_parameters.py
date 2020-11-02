import json
import numpy as np
import os
import unittest
from tempfile import TemporaryDirectory

import fp_analyses.senegal_parameters as sp


class TestSenegalParametersConfiguration(unittest.TestCase):
    with open(sp.DEFAULTS_FILE, 'r') as f:
        DEFAULT_INPUT_PARAMETERS = json.load(f)

    DEFAULT_PARAMETERS = sp.make_pars()

    def setUp(self):
        pass

    def tearDown(self):
        pass

    @classmethod
    def is_equal(cls, item1, item2):
        if type(item1) != type(item2):
            eq = False
        elif isinstance(item1, list):
            if len(item1) != len(item2):
                eq = False
            else:
                eq = all([cls.is_equal(item1[index], item2[index]) for index in range(len(item1))])
        elif isinstance(item1, dict):
            if item1.keys() != item2.keys():
                eq = False
            else:
                eq = all([cls.is_equal(item1[key], item2[key]) for key in item1.keys()])
        elif isinstance(item1, np.ndarray):
            eq = np.equal(item1, item2).all()
        else:
            eq = item1 == item2
        return eq

    @staticmethod
    def load_parameters(configuration):
        with TemporaryDirectory() as dirname:
            configuration_file = os.path.join(dirname, 'config.json')
            with open(configuration_file, 'w') as f:
                json.dump(obj=configuration, fp=f)
            parameters = sp.make_pars(configuration_file=configuration_file)
        return parameters

    def test_use_default_parameters(self):
        self.maxDiff = None
        configuration = {}
        parameters = self.load_parameters(configuration=configuration)
        for parameter in self.DEFAULT_PARAMETERS.keys():
            self.assertTrue(self.is_equal(self.DEFAULT_PARAMETERS[parameter], parameters[parameter]))

    def test_override_one_parameter_with_value(self):
        test_name = 'ThisIsATestOfTheEmergencyBroadcastSystem'
        configuration = {'name': test_name}
        parameters = self.load_parameters(configuration=configuration)
        for parameter in self.DEFAULT_PARAMETERS.keys():
            if parameter == 'name':
                self.assertEqual(test_name, parameters[parameter])
            else:
                self.assertTrue(self.is_equal(self.DEFAULT_PARAMETERS[parameter], parameters[parameter]))

    def test_override_one_parameter_with_null(self):
        configuration = {'name': None}
        parameters = self.load_parameters(configuration=configuration)
        for parameter in self.DEFAULT_PARAMETERS.keys():
            self.assertTrue(self.is_equal(self.DEFAULT_PARAMETERS[parameter], parameters[parameter]))

    def test_override_more_complicated_parameters_with_values(self):
        low = 0.22
        high = 0.33

        configuration = {'fertility_variation_low': low, 'fertility_variation_high': high}
        parameters = self.load_parameters(configuration=configuration)
        self.assertEqual([low, high], parameters['fertility_variation'])
        configuration = {'fertility_variation_low': low, 'fertility_variation_high': None}
        parameters = self.load_parameters(configuration=configuration)
        self.assertEqual([low, self.DEFAULT_INPUT_PARAMETERS['fertility_variation_high']], parameters['fertility_variation'])

        configuration = {'preg_dur_low': low, 'preg_dur_high': high}
        parameters = self.load_parameters(configuration=configuration)
        self.assertEqual([low, high], parameters['preg_dur'])
        configuration = {'preg_dur_low': low, 'preg_dur_high': None}
        parameters = self.load_parameters(configuration=configuration)
        self.assertEqual([low, self.DEFAULT_INPUT_PARAMETERS['preg_dur_high']], parameters['preg_dur'])

        configuration = {'breastfeeding_dur_low': low, 'breastfeeding_dur_high': high}
        parameters = self.load_parameters(configuration=configuration)
        self.assertEqual([low, high], parameters['breastfeeding_dur'])
        configuration = {'breastfeeding_dur_low': low, 'breastfeeding_dur_high': None}
        parameters = self.load_parameters(configuration=configuration)
        self.assertEqual([low, self.DEFAULT_INPUT_PARAMETERS['breastfeeding_dur_high']], parameters['breastfeeding_dur'])

    def test_override_all_parameters_with_null(self):
        configuration = {k: None for k in self.DEFAULT_INPUT_PARAMETERS.keys()}
        parameters = self.load_parameters(configuration=configuration)
        for parameter in self.DEFAULT_PARAMETERS.keys():
            self.assertTrue(self.is_equal(self.DEFAULT_PARAMETERS[parameter], parameters[parameter]))


if __name__ == '__main__':
    unittest.main()
