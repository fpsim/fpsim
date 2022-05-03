import coverage
import unittest
loader = unittest.TestLoader()
cov = coverage.Coverage(source=[
    "fpsim.base"
    , "fpsim.calibration"
    , "fpsim.defaults"
    , "fpsim.experiment"
    , "fpsim.interventions"
    , "fpsim.sim"
    , "fpsim.utils"
])
cov.start()

# First, load and run the unittest tests
from test_analyses import TestAnalyses
from test_baselines import TestBaselines
from test_calibration import TestCalibration
from test_contraceptives import TestContraceptiveEfficacy
from test_fertility import TestFertility
from test_interventions import TestInterventions
from test_multisim import TestMultisim
from test_new_channels import TestChannels
from test_parameters import TestParameters
from test_simple import TestSimple
from test_states import TestStates

test_classes_to_run = [TestAnalyses,
                       TestBaselines,
                       TestCalibration,
                       TestContraceptiveEfficacy,
                       TestFertility,
                       TestInterventions,
                       TestChannels,
                       TestMultisim,
                       TestParameters,
                       TestSimple,
                       TestStates]

suites_list = []
for tc in test_classes_to_run:
    suite = loader.loadTestsFromTestCase(tc)
    suites_list.append(suite)
    pass

big_suite = unittest.TestSuite(suites_list)
runner = unittest.TextTestRunner()
results = runner.run(big_suite)

cov.stop()
cov.save()
cov.html_report()