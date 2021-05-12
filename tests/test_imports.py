
import unittest

class FPSimImportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.expected_items = None
        self.found_items = None

        # region fpsim items
        self.fpsim_model_items = [
            "People",
            "Sim",
            "single_run",
            "multi_run"
        ]
        self.fpsim_analyses_items = [
            "demo",
            "Multisim"
        ]
        self.fpsim_calibration_items = [
            "Experiment",
            "Fit",
            "Calibration",
            "compute_gof",
            "datapath",
            "diff_summaries"
        ]
        self.fpsim_base_items = [
            "ParsObj",
            "BasePeople",
            "BaseSim"
        ]
        self.fpsim_version_items =[
            "__version__",
            "__versiondate__"
        ]
        # endregion

        pass

    def verify_expected_items_present(self, namespace):
        self.found_items = dir(namespace)
        for item in self.expected_items:
            self.assertIn(
                item,
                self.found_items
            )

    def tearDown(self) -> None:
        pass

    # def test_requirements(self):
    #     import emod_api
    #     import emodpy_generic
    #     import emodpy
    #     # Testing that we can import all requirements
    #     checks = [dir(package) for package in [emod_api, emodpy_generic, emodpy]]
    #     for package in checks:
    #         self.assertIn('__package__', package)

    # def test_module_interventions_complex_import(self):
    #     from emodpy_generic.interventions import complex_import
    #     self.expected_items = [
    #         "ComplexImportationEvent"
    #     ]
    #
    #     self.verify_expected_items_present(namespace=complex_import)
    #     pass

    def test_fpsim(self):
        import fpsim as fp
        self.expected_items = self.fpsim_analyses_items +\
                              self.fpsim_calibration_items +\
                              self.fpsim_model_items +\
                              self.fpsim_version_items

        self.verify_expected_items_present(
            namespace=fp
        )

    def test_fpsim_analyses(self):
        from fpsim import analyses as fp_analyses

        self.expected_items = self.fpsim_analyses_items
        self.verify_expected_items_present(
            namespace=fp_analyses
        )

    def test_fpsim_base(self):
        from fpsim import base as fp_base

        self.expected_items = self.fpsim_base_items
        self.verify_expected_items_present(
            namespace=fp_base
        )

    def test_fpsim_calibration(self):
        from fpsim import calibration as fp_calibration

        self.expected_items = self.fpsim_calibration_items
        self.verify_expected_items_present(
            namespace=fp_calibration
        )

    def test_fpsim_model(self):
        from fpsim import model as fp_model

        self.expected_items = self.fpsim_model_items
        self.verify_expected_items_present(
            namespace=fp_model
        )

    def test_fpsim_version(self):
        from fpsim import version as fp_version

        self.expected_items = self.fpsim_version_items
        self.verify_expected_items_present(
            namespace=fp_version
        )

    def test_fp_analyses(self):
        import fp_analyses as fp_analyses_dir

        self.expected_items = ["senegal_parameters"]
        self.verify_expected_items_present(
            namespace=fp_analyses_dir
        )


if __name__ == "__main__":
    unittest.main()
