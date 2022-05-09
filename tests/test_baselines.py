"""
Test that the current version of FPsim exactly matches
the baseline results. To update baseline, run ./update_baseline.
"""

import numpy as np
import datetime
import fpsim as fp
import sciris as sc
import fp_analyses as fa
import unittest
import os
import sys

class TestBaselines(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.do_plot = 0
        self.do_save = 0
        self.baseline_filename  = sc.thisdir(__file__, 'baseline.json')
        self.benchmark_filename = sc.thisdir(__file__, 'benchmark.json')

        # suppresses unnecessary print statements to increase runtime
        sys.stdout = open(os.devnull, 'w')

    def make_exp(self, n=10000, do_run=False, do_plot=False):
        '''
        Define a default simulation for testing the baseline.
        '''
        pars = fa.senegal_parameters.make_pars()
        pars['n'] = n
        exp = fp.Experiment(pars=pars)

        if do_run or do_plot:
            exp.run()

        if do_plot:
            exp.plot()

        return exp

    def save_baseline(self):
        '''
        Refresh the baseline results. This function is not called during standard testing,
        but instead is called by the update_baseline script.
        '''
        print('Updating baseline values...')
        exp = self.make_exp(do_run=True)
        exp.to_json(filename=self.baseline_filename)
        print('Done.')
        return

    @unittest.skip("Benchmark is changing rapidly with lots of code churn")
    def test_baseline(self):
        ''' Compare the current default sim against the saved baseline '''
        sc.heading('Testing baseline...')

        # Load existing baseline
        old = sc.loadjson(self.baseline_filename)

        # Calculate new baseline
        exp = self.make_exp(do_run=True)
        new = exp.summarize()

        # Compute the comparison
        fp.diff_summaries(old, new, die=True)

        pass


    def test_benchmark(self, repeats=1):
        ''' Compare benchmark performance '''

        sc.heading('Running benchmark...')
        previous = sc.loadjson(self.benchmark_filename)

        t_inits = []
        t_runs  = []
        t_posts  = []

        def normalize_performance():
            ''' Normalize performance across CPUs -- simple Numpy calculation '''
            t_bls = []
            bl_repeats = 3
            n_outer = 10
            n_inner = 1e6
            for r in range(bl_repeats):
                t0 = sc.tic()
                for i in range(n_outer):
                    a = np.random.random(int(n_inner))
                    b = np.random.random(int(n_inner))
                    a*b
                t_bl = sc.toc(t0, output=True)
                t_bls.append(t_bl)
            t_bl = min(t_bls)
            reference = 0.112 # Benchmarked on an Intel i9-8950HK CPU @ 2.90GHz
            ratio = reference/t_bl
            return ratio
        # Test CPU performance before the run
        r1 = normalize_performance()

        # Do the actual benchmarking
        for r in range(repeats):

            # Create the sim
            exp = self.make_exp()

            # Time initialization
            t0 = sc.tic()
            exp.initialize()
            t_init = sc.toc(t0, output=True)

            # Time running
            t0 = sc.tic()
            exp.run_model()
            t_run = sc.toc(t0, output=True)

            # Time postprocessing
            t0 = sc.tic()
            exp.post_process_results()
            t_post = sc.toc(t0, output=True)

            # Store results
            t_inits.append(t_init)
            t_runs.append(t_run)
            t_posts.append(t_post)

        # Test CPU performance after the run
        r2 = normalize_performance()
        ratio = (r1+r2)/2
        t_init = min(t_inits)*ratio
        t_run  = min(t_runs)*ratio
        t_post = min(t_posts)*ratio

        # Construct json
        n_decimals = 3
        json = {'time': {
                    'initialize':  round(t_init, n_decimals),
                    'run':         round(t_run,  n_decimals),
                    'postprocess': round(t_post, n_decimals),
                    'total':       round(t_init+t_run+t_post, n_decimals)
                    },
                'parameters': {
                    'n':          exp.pars['n'],
                    'start_year': exp.pars['start_year'],
                    'end_year':   exp.pars['end_year'],
                    'timestep':   exp.pars['timestep'],
                    },
                'cpu_performance': ratio,
                }

        def printjson(json):
            ''' Print more nicely '''
            print(sc.jsonify(json, tostring=True, indent=2))

        print('Previous benchmark:')
        printjson(previous)

        print('\nNew benchmark:')
        printjson(json)

        if self.do_save:
            sc.savejson(filename=self.benchmark_filename, obj=json, indent=2)

        print('Done.')

        pass

if __name__ == '__main__':

    # run test suite
    unittest.main()

