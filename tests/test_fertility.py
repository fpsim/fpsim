import pytest
import fpsim as fp

import fp_analyses.default_parameters as dp

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

    def sweep_seed(self, seeds=None, pars=dp.make_pars(),
                   num_humans=1000, end_year=1963,
                   var_str=None):
        '''Sweep the specified parameters over random seed and save to disk'''
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
            parameters['lactational_amenorrhea'] = dp.default_lactational_amenorrhea(rate)
            self.sweep_seed(
                seeds=seeds, pars=parameters, end_year=1966
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
            parameters['fecundity_ratio_nullip'] = dp.default_fecundity_ratio_nullip(rate)
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
            parameters['sexual_activity'] = dp.default_sexual_activity(rate)
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
                dp.default_exposure_correction_age(ec)
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

    def test_sweep_sexual_activity(self):
        """array model at 0.1, 0.2, 0.4, 0.8 sexual_activity should
        have non-overlapping birth counts"""
        pars = dp.make_pars()
        sex_rates = [40.0, 80.0, 160.0, 320.0]
        seeds = self.default_seeds
        self.sweep_sexual_activity(
            sex_rates=sex_rates,
            parameters=pars,
            seeds=seeds)

    def test_sweep_lam(self):
        """array model at lam rates [1.0, 0.5, 0.25, 0.0] should
        have increasing birth counts"""
        self.is_debugging = False
        pars = dp.make_pars()
        pars['sexual_activity'] = dp.default_sexual_activity(320.0)
        lam_rates = [1.0, 0.5, 0.25, 0.0]
        seeds = self.default_seeds
        self.sweep_lam(
            lam_rates=lam_rates,
            parameters=pars,
            seeds=seeds)

    def test_sweep_nullip_ratio_array(self):
        """array model at 0.1, 0.2, 0.4, 0.8 sexual_activity should
        have non-overlapping birth counts"""
        self.is_debugging = False
        pars = dp.make_pars()
        pars['sexual_activity'] = dp.default_sexual_activity(320.0)
        pars['lactational_amenorrhea'] = dp.default_lactational_amenorrhea(0.0)
        nullip_ratios = [0.1, 0.5, 1.0, 2.0, 4.0]
        seeds = self.default_seeds
        self.sweep_fecundity_nullip(
            nullip_rates=nullip_ratios,
            parameters=pars,
            seeds=seeds)

    def test_sweep_exposure_correction(self):
        """array model at 0.1, 0.2, 0.4, 0.8 sexual_activity should
        have non-overlapping birth counts"""
        pars = dp.make_pars()
        exposure_corrections = [0.1, 0.5, 1.0, 2.0, 4.0]
        seeds = self.default_seeds
        self.sweep_exposure_correction(
            exposure_corrections=exposure_corrections,
            parameters=pars,
            seeds=seeds)

    def test_sweep_ec_age(self):
        """array model at 0.1, 0.2, 0.4, 0.8 sexual_activity should
        have non-overlapping birth counts"""
        pars = dp.make_pars()
        exposure_corrections = [0.0, 0.1, 0.5, 1.0, 2.0]
        seeds = self.default_seeds
        self.sweep_ec_age(
            ec_ages=exposure_corrections,
            parameters=pars,
            seeds=seeds)

    def test_sweep_abortion(self):
        """array model at abortion rates [1.0 to 0.0] should
        start at 0.0 and have increasing birth counts"""
        pars = dp.make_pars()
        pars['sexual_activity'] = dp.default_sexual_activity(320.0)
        pars['lactational_amenorrhea'] = dp.default_lactational_amenorrhea(0.0)
        abortion_probs = [1.0, 0.5, 0.25, 0.1, 0.0]
        seeds = self.default_seeds
        self.sweep_abortion_probability(
            abortion_probs=abortion_probs,
            parameters=pars,
            seeds=seeds)

    @pytest.mark.skip("NYI")
    def test_sweep_age_pyramid(self):
        raise NotImplementedError()
        pass

    @pytest.mark.skip("NYI")
    def test_sweep_age_mortality(self):
        raise NotImplementedError()
        pass

    @pytest.mark.skip("NYI")
    def test_sweep_age_fecundity(self):
        raise NotImplementedError()
        pass

    @pytest.mark.skip("NYI")
    def test_sweep_maternal_mortality(self):
        raise NotImplementedError()
        pass

    @pytest.mark.skip("NYI")
    def test_sweep_infant_mortality(self):
        raise NotImplementedError()
        pass

    @pytest.mark.skip("NYI")
    def test_sweep_sexual_activity_postpartum(self):
        raise NotImplementedError()
        pass


if __name__ == "__main__":
    pytest.main(['-vs', __file__])
