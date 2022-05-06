import pytest
import fpsim as fp

import json
import sciris as sc
import os
from pathlib import Path
import unittest
import sys

@pytest.mark.skip("Need to refactor test parameters to be within the test suite")
class TestFertility(unittest.TestCase):
    def setUp(self):
        self.is_debugging = False
        self.output_files = []
        self.testname = self._testMethodName
        self.log_lines = []
        self.default_seeds = [1] # Add more when debugging

        # suppresses unnecessary warning statements to increase runtime
        sys.stdout = open(os.devnull, 'w')
        pass

    def tearDown(self):
        if self.is_debugging:
            with open(f'DEBUG_log_{self.testname}.json', 'w') as outfile:
                log = {'lines': self.log_lines}
                json.dump(log, outfile, indent=4, sort_keys=True)
        else:
            self.cleanup_debug_files()

    @staticmethod
    def get_results_by_filename(result_filename):
        return sc.loadjson(result_filename)['results']

    def cleanup_debug_files(self):
        for f in self.output_files:
            if Path(f).exists():
                os.remove(f)

    def sweep_seed(self, seeds=None, pars=fp.pars(), # CK: this is a mutable object so shouldn't be defined here
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

    def verify_increasing_channel(self, parameter_name, parameter_test_values,
                                  parameter_method, parameters, seeds,
                                  has_zero_baseline=False, channel_name='births'):
        '''Verifies that as a specified parameter increases, the total number of births increases

        Args:
            parameter_name: name of the parameter in the model (key in parameters dictionary)
            parameter_test_values: array of values used in creating parameter.
              should be sorted so that channel monitored will increase.
            parameter_method: method that creates parameter values
            parameters: a parameters dictionary for fpsim
            seeds: optional. Seeds to sweep against for deeper investigation
            has_zero_baseline: (False) set to True if the first result should be zeroes
            channel_name: ('births') channel monitored to observe increase
        '''

        if seeds is None:
            seeds = self.default_seeds
        all_channel_sums = {}
        current_channel_sums = []
        # start populating all birth sums
        for p_val in parameter_test_values:
            if parameter_method:
                parameters[parameter_name] = \
                    parameter_method(p_val)
            else:
                parameters[parameter_name] = p_val
            self.sweep_seed(
                seeds=seeds, pars=parameters, var_str=f'{parameter_name}_{p_val}'
            )
            # This loop requires explanation
            # self.output_files is a flat list of all sims created
            # each p_val has been run as many times as there are seeds (3 seeds, run 3x)
            # If you have three seeds, this next line says self.output_files[-3:]
            # Which means "consider the last 3 (number of seeds) items in the list".
            for result_file in self.output_files[-(len(seeds)):]:
                result_dict = self.get_results_by_filename(result_file)
                current_channel_sums.append(sum(result_dict[channel_name]))
            self.log_lines.append(f"At rate {p_val} saw {sum(current_channel_sums)} {channel_name} {current_channel_sums}")
            all_channel_sums[p_val] = current_channel_sums
            current_channel_sums = []

        prev_key = None
        checked = []
        for p_val in parameter_test_values:
            # check first one for zeros
            if has_zero_baseline and prev_key == None:
                checked.append(p_val)
                zeros = [0] * len(seeds)
                assert all_channel_sums[p_val] == zeros, f"Expected zeros with {p_val=}. Got {all_channel_sums[p_val]=}"
            elif prev_key is not None:
                checked.append(p_val)
                # check rest for increasing
                assert sum(all_channel_sums[prev_key]) < sum(all_channel_sums[p_val]), \
                    f"Expected less {channel_name} with {parameter_name} {prev_key}:({all_channel_sums[prev_key]}, " \
                    f"than {p_val}:({all_channel_sums[p_val]}) "
            prev_key = p_val
        self.log_lines.append(all_channel_sums)
        self.log_lines.append(f"Checked all these: {checked}")

    def sweep_fecundity_nullip(self, nullip_rates, parameters,
                               channel_name='births', seeds=None):
        '''validates that increasing fecundity_ratio_nullip (multiplier on
        sexual activity postpartum) leads to increasing numbers of births

        Args:
            nullip_rates: array of multipliers on postpartum sexual activity. If not
              included, 0 will be prepended (no sexual activity until end of postpartum)
            channel_name: (births) channel to be observed
        '''
        if 0 not in nullip_rates:
            nullip_rates = [0] + nullip_rates
        nullip_rates = sorted(nullip_rates)
        self.verify_increasing_channel(
            parameter_name='fecundity_ratio_nullip',
            parameter_method=sp.default_fecundity_ratio_nullip,
            parameter_test_values=nullip_rates,
            parameters=parameters,
            seeds=seeds,
            channel_name=channel_name,
            has_zero_baseline=True
        )

    def sweep_sexual_activity(self, sex_rates, parameters,
                              channel_name, seeds=None):
        '''validates that increasing rates of sexual activity lead to increased births

        Args:
            sex_rates: array of sex rates (which are the same for all age groups)
              will start with 0 or be prepended with 0
            channel_name: (births) channel to be observed
            '''
        if 0 not in sex_rates:
            sex_rates = [0] + sex_rates
        sex_rates = sorted(sex_rates)
        self.verify_increasing_channel(
            parameter_name='sexual_activity',
            parameter_test_values=sex_rates,
            parameter_method=sp.default_sexual_activity,
            parameters=parameters,
            channel_name=channel_name,
            seeds=seeds,
            has_zero_baseline=True
        )

    def sweep_exposure_correction(self, exposure_corrections, parameters,
                                  channel_name, seeds=None):
        '''validates that increasing the exposure correction parameter increases births

        Args:
            exposure_corrections: array of multipliers on birth probability
            channel_name: (births) channel to be observed
        '''
        if 0 not in exposure_corrections:
            exposure_corrections = [0] + exposure_corrections
        exposure_corrections = sorted(exposure_corrections)
        self.verify_increasing_channel(
            parameter_name='exposure_correction',
            parameter_method=None,
            parameters=parameters,
            parameter_test_values=exposure_corrections,
            seeds=seeds,
            channel_name=channel_name,
            has_zero_baseline=True
        )

    def sweep_abortion_probability(self, abortion_probs, parameters,
                                   channel_name='births', seeds=None):
        '''validates that decreasing the probability of abortion increases total births

        Args:
            abortion_probs: array of decreasing probabilities of abortion. will start with
              1 or be prepended with 1
            channel_name: (births) channel to be observed
        '''
        if 1 not in abortion_probs: # Ensure that we get the zero baseline
            abortion_probs = [1] + abortion_probs
        abortion_probs = sorted(abortion_probs, reverse=True)
        self.verify_increasing_channel(
            parameter_name='abortion_prob',
            parameter_method=None,
            parameter_test_values=abortion_probs,
            parameters=parameters,
            seeds=seeds,
            channel_name=channel_name,
            has_zero_baseline=True
        )

    def sweep_ec_age(self, ec_ages, parameters,
                     channel_name, seeds=None):
        '''validates that increasing exposure correction by age increases total births

        see also the sweep_exposure_correction method. this method sets all age groups to
        the same multiplier

        Args:
            ec_ages: array of exposure correction multipliers to be
              applied to each age group
            channel_name: (births) channel to be observed
        '''
        if 0 not in ec_ages:
            ec_ages = [0] + ec_ages
        ec_ages = sorted(ec_ages)
        self.verify_increasing_channel(
            parameter_name='exposure_correction_age',
            parameter_test_values=ec_ages,
            parameter_method=sp.default_exposure_correction_age,
            parameters=parameters,
            channel_name=channel_name,
            seeds=seeds,
            has_zero_baseline=True
        )

    def sweep_maternal_mortality(self, maternal_mortality_multipliers, parameters,
                                 channel_name='maternal_deaths', seeds=None):
        '''verifies that as you increase the rate of maternal mortality, you get increasing maternal death

        Args:
            maternal_mortality_multipliers: array of increasing multipliers on maternal mortality
              will be prepended with 0 if not included
            channel_name: (maternal_deaths) channel to be observed
            '''
        if 0 not in maternal_mortality_multipliers:
            maternal_mortality_multipliers = [0] + maternal_mortality_multipliers
        maternal_mortality_multipliers = sorted(maternal_mortality_multipliers)
        self.verify_increasing_channel(
            parameter_name='maternal_mortality',
            parameter_test_values=maternal_mortality_multipliers,
            parameter_method=sp.default_maternal_mortality,
            parameters=parameters,
            seeds=seeds,
            has_zero_baseline=True,
            channel_name=channel_name
        )

    def sweep_infant_mortality(self, infant_mortality_rates, parameters,
                               channel_name='infant_deaths', seeds=None):
        '''verifies that as you increase the infant mortality rate, you get increasing infant deaths

        Args:
            infant_mortality_rates: array of increasing infant mortality rates
              will be prepended with 0 if not included
            channel_name: (infant_deaths) channel to be observed
        '''
        if 0 not in infant_mortality_rates:
            infant_mortality_rates = [0] + infant_mortality_rates
        infant_mortality_rates = sorted(infant_mortality_rates)
        self.verify_increasing_channel(
            parameter_name='infant_mortality',
            parameter_test_values=infant_mortality_rates,
            parameters=parameters,
            parameter_method=sp.default_infant_mortality,
            seeds=seeds,
            has_zero_baseline=True,
            channel_name=channel_name
        )

    def test_sweep_sexual_activity(self):
        """as you increase sexual activity, births should increase"""
        self.is_debugging = False
        pars = sp.make_pars()
        sex_rates = [30.0, 60.0, 120.0]
        seeds = self.default_seeds
        self.sweep_sexual_activity(
            sex_rates=sex_rates,
            parameters=pars,
            channel_name='births',
            seeds=seeds)

    def test_sweep_nullip_ratio_array(self):
        """as you increase nulliparous fecundity, you should increase births"""
        self.is_debugging = False
        pars = sp.make_pars()
        pars['sexual_activity'] = sp.default_sexual_activity(320.0)
        pars['lactational_amenorrhea'] = sp.default_lactational_amenorrhea(0.0)
        nullip_ratios = [0.1, 1.0, 4.0]
        if self.is_debugging:
            nullip_ratios = [0.1, 0.5, 1.0, 2.0, 4.0]
        seeds = self.default_seeds
        self.sweep_fecundity_nullip(
            nullip_rates=nullip_ratios,
            parameters=pars,
            channel_name='births',
            seeds=seeds)

    def test_sweep_exposure_correction(self):
        """as you increase exposure correction, births should increase"""
        self.is_debugging = True
        pars = sp.make_pars()
        exposure_corrections = [0.1, 1.0, 4.0]
        if self.is_debugging:
            exposure_corrections = [0.1, 0.5, 1.0, 2.0, 4.0]
        seeds = self.default_seeds
        self.sweep_exposure_correction(
            exposure_corrections=exposure_corrections,
            parameters=pars,
            channel_name='births',
            seeds=seeds)

    def test_sweep_ec_age(self):
        """as you increase exposure correction by age, births should increase"""
        self.is_debugging = False
        pars = sp.make_pars()
        exposure_corrections = [0.0, 1.0, 2.0]
        if self.is_debugging:
            exposure_corrections = [0.0, 0.1, 0.5, 1.0, 2.0]
        seeds = self.default_seeds
        self.sweep_ec_age(
            ec_ages=exposure_corrections,
            parameters=pars,
            channel_name='births',
            seeds=seeds)

    def test_sweep_abortion(self):
        """as you decrease abortion probability, number of births should increase"""
        self.is_debugging = False
        pars = sp.make_pars()
        pars['sexual_activity'] = sp.default_sexual_activity(320.0)
        pars['lactational_amenorrhea'] = sp.default_lactational_amenorrhea(0.0)
        abortion_probs = [1.0, 0.25, 0.0]
        if self.is_debugging:
            abortion_probs = [1.0, 0.5, 0.25, 0.1, 0.0]
        seeds = self.default_seeds
        self.sweep_abortion_probability(
            abortion_probs=abortion_probs,
            parameters=pars,
            channel_name='births',
            seeds=seeds)

    def test_sweep_maternal_mortality(self):
        """as you increase maternal mortality, maternal deaths should increase"""
        self.is_debugging = False
        pars = sp.make_pars()
        pars['sexual_activity'] = sp.default_sexual_activity(320.0) # many conceptions
        multipliers = [10, 100] # default is 2/1000. Increased for observability
        if self.is_debugging:
            multipliers = [10, 50, 100, 200] # default is 2/1000
        self.sweep_maternal_mortality(maternal_mortality_multipliers=multipliers,
                                      channel_name='maternal_deaths', parameters=pars)

    def test_sweep_infant_mortality(self):
        """as you increase infant mortality, infant deaths should increase"""
        self.is_debugging = False
        pars = sp.make_pars()
        pars['sexual_activity'] = sp.default_sexual_activity(320.0) # many conceptions
        infant_mortality_rates = [0, 50, 200]
        if self.is_debugging:
            infant_mortality_rates = [0, 10, 50, 100, 200]
        self.sweep_infant_mortality(infant_mortality_rates=infant_mortality_rates,
                                    channel_name='infant_deaths', parameters=pars)


if __name__ == '__main__':

    # run test suite
    unittest.main()
