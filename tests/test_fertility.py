import unittest

import pytest
import fpsim as fp

import fp_analyses.test_parameters as tp

import json
import sciris as sc
import os
from pathlib import Path

class TestFPSimFertility():
    def setup_method(self):
        self.is_debugging = False
        self.output_files = []
        self.testname = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
        self.log_lines = []
        self.default_seeds = [1] # Add more when debugging
        pass

    def teardown_method(self):
        if self.is_debugging:
            with open(f'DEBUG_log_{self.testname}.json', 'w') as outfile:
                log = {'lines': self.log_lines}
                json.dump(log, outfile, indent=4, sort_keys=True)
        else:
            self.cleanup_debug_files()

    def get_results_by_filename(self, result_filename):
        ret_dict = None
        with open(result_filename) as infile:
            ret_dict = json.load(infile)['results']
        return ret_dict

    def cleanup_debug_files(self):
        for f in self.output_files:
            if Path(f).exists():
                os.remove(f)

    def sweep_seed(self, seeds=None, pars=tp.make_pars(),
                   num_humans=1000, end_year=1963,
                   var_str=None):
        '''Sweep the specified parameters over random seed and save to disk'''
        if not var_str:
            var_str = "seeds"
        pars['n'] = num_humans
        pars['end_year'] = end_year
        if seeds is None:
            seeds = [0]

        for seed in seeds:
            pars['seed'] = seed
            output_filename = f'DEBUG_{var_str}_sim_{seed}.json'
            experiment = fp.Experiment(pars=pars)
            experiment.run()
            sim_stuff = {'pars': experiment.sim.pars,
                         'results': experiment.sim.results}
            sc.savejson(output_filename, sim_stuff)
            self.output_files.append(output_filename)

    def sweep_lam(self, lam_rates, parameters, seeds=None):
        if seeds is None:
            seeds = self.default_seeds
        previous_birth_sums = [0] * len(seeds)
        all_birth_sums = {}
        current_birth_sums = []
        for rate in lam_rates:
            parameters['lactational_amenorrhea'] = tp.default_lactational_amenorrhea(rate)
            self.sweep_seed(
                seeds=seeds, pars=parameters, end_year=1966, num_humans=2000
            )
            for result_file in self.output_files[-(len(seeds)):]:
                result_dict = self.get_results_by_filename(result_file)
                current_birth_sums.append(sum(result_dict['births']))
            self.log_lines.append(f"At rate {rate} saw {sum(current_birth_sums)} births {current_birth_sums}")
            assert sum(previous_birth_sums) < sum(current_birth_sums)
            all_birth_sums[rate] = current_birth_sums
            previous_birth_sums = current_birth_sums
            current_birth_sums = []
        self.log_lines.append(all_birth_sums)

    def sweep_fecundity_nullip(self, nullip_rates, parameters, seeds=None):
        if seeds is None:
            seeds = self.default_seeds
        nullip_rates = sorted(nullip_rates)
        previous_birth_sums = [0] * len(seeds)
        all_birth_sums = {}
        current_birth_sums = []
        for rate in nullip_rates:
            parameters['fecundity_ratio_nullip'] = tp.default_fecundity_ratio_nullip(rate)
            self.sweep_seed(
                seeds=seeds, pars=parameters, end_year=1966, var_str="fecundity_nullip_ratio"
            )
            for result_file in self.output_files[-(len(seeds)):]:
                result_dict = self.get_results_by_filename(result_file)
                current_birth_sums.append(sum(result_dict['births']))
            self.log_lines.append(f"At rate {rate} saw {sum(current_birth_sums)} births {current_birth_sums}")
            assert sum(previous_birth_sums) < sum(current_birth_sums)
            all_birth_sums[rate] = current_birth_sums
            previous_birth_sums = current_birth_sums
            current_birth_sums = []
        self.log_lines.append(all_birth_sums)

    def sweep_sexual_activity(self, sex_rates, parameters, seeds=None):
        if seeds is None:
            seeds = self.default_seeds
        if 0 not in sex_rates:
            sex_rates = [0] + sex_rates
        sex_rates = sorted(sex_rates)
        previous_birth_sums = None
        all_birth_sums = {}
        current_birth_sums = []
        for rate in sex_rates:
            parameters['sexual_activity'] = tp.default_sexual_activity(rate)
            self.sweep_seed(
                seeds=seeds, pars=parameters
            )
            for result_file in self.output_files[-(len(seeds)):]:
                result_dict = self.get_results_by_filename(result_file)
                current_birth_sums.append(sum(result_dict['births']))
            self.log_lines.append(f"At rate {rate} saw {sum(current_birth_sums)} births {current_birth_sums}")
            if previous_birth_sums:
                assert sum(previous_birth_sums) < sum(current_birth_sums)
            else: # Rate should be zero
                assert sum(current_birth_sums) == 0
            all_birth_sums[rate] = current_birth_sums
            previous_birth_sums = current_birth_sums
            current_birth_sums = []
        self.log_lines.append(all_birth_sums)

    def sweep_exposure_correction(self, exposure_corrections, parameters, seeds=None):
        if seeds is None:
            seeds = self.default_seeds
        previous_birth_sums = [0, 0, 0, 0, 0]
        all_birth_sums = {}
        current_birth_sums = []
        for ec in exposure_corrections:
            parameters['exposure_correction'] = ec
            self.sweep_seed(
                seeds=seeds, pars=parameters, var_str=f'ec_{ec}'
            )
            for result_file in self.output_files[-(len(seeds)):]:
                result_dict = self.get_results_by_filename(result_file)
                current_birth_sums.append(sum(result_dict['births']))
            # self.log_lines.append(sc.jsonify(parameters))
            self.log_lines.append(f"At rate {ec} saw {sum(current_birth_sums)} births {current_birth_sums}")
            assert sum(previous_birth_sums) < sum(current_birth_sums)
            all_birth_sums[ec] = current_birth_sums
            previous_birth_sums = current_birth_sums
            current_birth_sums = []
        self.log_lines.append(all_birth_sums)

    def sweep_abortion_probability(self, abortion_probs, parameters, seeds=None):
        if seeds is None:
            seeds = self.default_seeds
        if 1 not in abortion_probs: # Ensure that we get the zero baseline
            abortion_probs = [1] + abortion_probs
        previous_birth_sums = None
        all_birth_sums = {}
        current_birth_sums = []
        for ap in abortion_probs:
            parameters['abortion_prob'] = ap
            self.sweep_seed(
                seeds=seeds, pars=parameters
            )
            for result_file in self.output_files[-(len(seeds)):]:
                result_dict = self.get_results_by_filename(result_file)
                current_birth_sums.append(sum(result_dict['births']))
            self.log_lines.append(f"At abortion prob {ap=} saw {sum(current_birth_sums)=} {current_birth_sums=}")
            if previous_birth_sums:
                assert sum(previous_birth_sums) < sum(current_birth_sums)
            else: # rate must be 0 here
                assert sum(current_birth_sums) == 0
            all_birth_sums[ap] = current_birth_sums
            previous_birth_sums = current_birth_sums
            current_birth_sums = []
        self.log_lines.append(all_birth_sums)

    def sweep_spacing_preference(self, spacing_prefs, parameters, seeds=None):
        if seeds is None:
            seeds = self.default_seeds
        if 0 not in spacing_prefs:
            spacing_prefs = [0] + spacing_prefs
        previous_birth_sums = None
        all_birth_sums = {}
        current_birth_sums = []
        for sp in spacing_prefs:
            parameters['pref_spacing'] = tp.default_birth_spacing_preference(sp)
            self.sweep_seed(
                seeds=seeds, pars=parameters
            )
            for result_file in self.output_files[-(len(seeds)):]:
                result_dict = self.get_results_by_filename(result_file)
                current_birth_sums.append(sum(result_dict['births']))
            self.log_lines.append(f"At spacing pref {sp=} saw {sum(current_birth_sums)=} {current_birth_sums=}")
            if previous_birth_sums:
                assert sum(previous_birth_sums) < sum(current_birth_sums)
            all_birth_sums[sp] = current_birth_sums
            previous_birth_sums = current_birth_sums
            current_birth_sums = []
        self.log_lines.append(all_birth_sums)

    def sweep_ec_age(self, ec_ages, parameters, seeds=None):
        if seeds is None:
            seeds = self.default_seeds
        if 0 not in ec_ages:
            ec_ages = [0] + ec_ages
        all_birth_sums = {}
        current_birth_sums = []
        # start populating all birth sums
        for ec in ec_ages:
            parameters['exposure_correction_age'] = \
                tp.default_exposure_correction_age(ec)
            self.sweep_seed(
                seeds=seeds, pars=parameters, var_str=f'ec_{ec}'
            )
            for result_file in self.output_files[-(len(seeds)):]:
                result_dict = self.get_results_by_filename(result_file)
                current_birth_sums.append(sum(result_dict['births']))
            self.log_lines.append(f"At rate {ec} saw {sum(current_birth_sums)} births {current_birth_sums}")
            all_birth_sums[ec] = current_birth_sums
            current_birth_sums = []
        prev_key = None
        for ec in ec_ages:
            # check first one for zeros
            if ec == 0:
                zeros = [0] * len(seeds)
                assert all_birth_sums[ec] == zeros, f"Expected zeros with {ec=}. Got {all_birth_sums[ec]=}"
            else:
                # check rest for increasing
                assert sum(all_birth_sums[prev_key]) < sum(all_birth_sums[ec]), \
                    f"Expected less births with exposure correction {prev_key}:({all_birth_sums[prev_key]}, " \
                    f"than {ec}:({all_birth_sums[ec]}) "
            prev_key = ec
        self.log_lines.append(all_birth_sums)

    def sweep_maternal_mortality(self, maternal_mortality_multipliers, parameters, seeds=None):
        if seeds is None:
            seeds = self.default_seeds
        if 0 not in maternal_mortality_multipliers:
            maternal_mortality_multipliers = [0] + maternal_mortality_multipliers
        previous_maternal_deaths = 0
        all_maternal_death_sums = {}
        for mmm in maternal_mortality_multipliers:
            current_maternal_death_sums = []
            parameters['maternal_mortality'] = tp.default_maternal_mortality(mmm)
            self.sweep_seed(
                seeds=seeds, pars=parameters
            )
            for result_file in self.output_files[-(len(seeds)):]:
                result_dict = self.get_results_by_filename(result_file)
                current_maternal_death_sums.append(sum(result_dict['maternal_deaths']))
            self.log_lines.append(f"At maternal mortality multiplier {mmm=}, saw {sum(current_maternal_death_sums)=} {current_maternal_death_sums=}")
            if previous_maternal_deaths:
                assert sum(previous_maternal_deaths) <= sum (current_maternal_death_sums)
            all_maternal_death_sums[mmm] = current_maternal_death_sums
            previous_maternal_deaths = current_maternal_death_sums
        self.log_lines.append(all_maternal_death_sums)
        pass

    def sweep_infant_mortality(self, infant_mortality_rates, parameters, seeds=None):
        if seeds is None:
            seeds = self.default_seeds
        if 0 not in infant_mortality_rates:
            infant_mortality_rates = [0] + infant_mortality_rates
        previous_infant_deaths = 0
        all_infant_death_sums = {}
        for imr in infant_mortality_rates:
            current_infant_death_sums = []
            parameters['infant_mortality'] = tp.default_infant_mortality(test_rate=imr)
            self.sweep_seed(
                seeds=seeds, pars=parameters
            )
            for result_file in self.output_files[-(len(seeds)):]:
                result_dict = self.get_results_by_filename(result_file)
                current_infant_death_sums.append(sum(result_dict['infant_deaths']))
            self.log_lines.append(f"At infant mortality rate {imr=}, saw {sum(current_infant_death_sums)=} {current_infant_death_sums=}")
            if previous_infant_deaths:
                assert sum(previous_infant_deaths) <= sum (current_infant_death_sums)
            all_infant_death_sums[imr] = current_infant_death_sums
            previous_infant_deaths = current_infant_death_sums
        self.log_lines.append(all_infant_death_sums)
        pass

    def test_sweep_sexual_activity(self):
        """model at 0.1, 0.2, 0.4, 0.8 sexual_activity should
        have non-overlapping birth counts"""
        self.is_debugging = False
        pars = tp.make_pars()
        sex_rates = [30.0, 60.0, 120.0]
        seeds = self.default_seeds
        self.sweep_sexual_activity(
            sex_rates=sex_rates,
            parameters=pars,
            seeds=seeds)

    @unittest.skip("Test is unstable at this time")
    def test_sweep_lam(self):
        """model at lam rates [1.0, 0.4, 0.1] should
        have increasing birth counts"""
        self.is_debugging = False
        pars = tp.make_pars()

        # Crank up birth rate
        pars['sexual_activity'] = tp.default_sexual_activity(320.0)
        pars['abortion_prob'] = 0.0
        pars['miscarriage_rates'] = tp.default_miscarriage_rates(0)

        # Make LAM very important
        pars['breastfeeding_dur_low'] = 48
        pars['breastfeeding_dur_high'] = 48
        pars['LAM_efficacy'] = 1.0
        lam_rates = [1.0, 0.4, 0.1]
        seeds = self.default_seeds
        self.sweep_lam(
            lam_rates=lam_rates,
            parameters=pars,
            seeds=seeds)

    def test_sweep_nullip_ratio_array(self):
        """model at 0.1, 0.2, 0.4, 0.8 sexual_activity should
        have non-overlapping birth counts"""
        self.is_debugging = False
        pars = tp.make_pars()
        pars['sexual_activity'] = tp.default_sexual_activity(320.0)
        pars['lactational_amenorrhea'] = tp.default_lactational_amenorrhea(0.0)
        nullip_ratios = [0.1, 0.5, 1.0, 2.0, 4.0]
        seeds = self.default_seeds
        self.sweep_fecundity_nullip(
            nullip_rates=nullip_ratios,
            parameters=pars,
            seeds=seeds)

    def test_sweep_exposure_correction(self):
        """model at 0.1, 0.2, 0.4, 0.8 sexual_activity should
        have non-overlapping birth counts"""
        pars = tp.make_pars()
        exposure_corrections = [0.1, 0.5, 1.0, 2.0, 4.0]
        seeds = self.default_seeds
        self.sweep_exposure_correction(
            exposure_corrections=exposure_corrections,
            parameters=pars,
            seeds=seeds)

    def test_sweep_ec_age(self):
        """model at 0.1, 0.2, 0.4, 0.8 sexual_activity should
        have non-overlapping birth counts"""
        pars = tp.make_pars()
        exposure_corrections = [0.0, 0.1, 0.5, 1.0, 2.0]
        seeds = self.default_seeds
        self.sweep_ec_age(
            ec_ages=exposure_corrections,
            parameters=pars,
            seeds=seeds)

    def test_sweep_abortion(self):
        """model at abortion rates [1.0 to 0.0] should
        start at 0.0 and have increasing birth counts"""
        pars = tp.make_pars()
        pars['sexual_activity'] = tp.default_sexual_activity(320.0)
        pars['lactational_amenorrhea'] = tp.default_lactational_amenorrhea(0.0)
        abortion_probs = [1.0, 0.5, 0.25, 0.1, 0.0]
        seeds = self.default_seeds
        self.sweep_abortion_probability(
            abortion_probs=abortion_probs,
            parameters=pars,
            seeds=seeds)

    def test_sweep_spacing_preference(self):
        """
        Spacing preference is a multiplier on sexual activity in the 12 months postpartum.
        Test is like the sexual activity test other than greatly cranking up the birth rates
        so that we have enough data to consdier the scaling.
        """
        self.is_debugging = False
        pars = tp.make_pars()
        pars['sexual_activity'] = tp.default_sexual_activity(320.0)
        uniform_spacing_preferences = [0.3, 1.0, 2.0]
        seeds = self.default_seeds
        seeds = [1, 2, 3, 4, 5]
        self.sweep_spacing_preference(
            spacing_prefs=uniform_spacing_preferences,
            parameters=pars,
            seeds=seeds
        )

    def test_sweep_maternal_mortality(self):
        self.is_debugging = False
        pars = tp.make_pars()
        pars['sexual_activity'] = tp.default_sexual_activity(320.0) # many conceptions
        multipliers = [10, 50, 100, 200] # default is 2/1000
        self.sweep_maternal_mortality(maternal_mortality_multipliers=multipliers,
                                      parameters=pars)

    def test_sweep_infant_mortality(self):
        self.is_debugging = False
        pars = tp.make_pars()
        pars['sexual_activity'] = tp.default_sexual_activity(320.0) # many conceptions
        infant_mortality_rates = [10, 50, 100, 200]
        self.sweep_infant_mortality(infant_mortality_rates=infant_mortality_rates,
                                    parameters=pars)

    @pytest.mark.skip("NYI")
    def test_sweep_sexual_activity_postpartum(self):
        # Possible:
        # # Set default sexual activity rate to 50 or something with a larger overall population
        # # Sweep postpartum across much larger numbers (100, 200, 400)
        # Better:
        # # Set all women of age to "pregnant" in year 0 with a large base population
        # # Sweep postpartum rates of 0 to 320 in years 0 and 1
        raise NotImplementedError()
        pass


if __name__ == "__main__":
    pytest.main(['-vs', __file__])
