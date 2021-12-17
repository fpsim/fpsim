"""
Simplest possible FPsim test.
"""

import sciris as sc
import fpsim as fp
import fp_analyses as fa


def test_simple(n=100, do_plot=False):
    '''
    Define a default simulation for testing the baseline.
    '''
    pars = fa.senegal_parameters.make_pars()
    pars['n'] = n
    sim = fp.Sim(pars)
    sim.run()

    if do_plot:
        sim.plot()

    return sim


if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    T = sc.tic()

    sim = test_simple(do_plot=True) # Run this first so benchmarking is available even if results are different

    print('\n'*2)
    sc.toc(T)
    print('Done.')