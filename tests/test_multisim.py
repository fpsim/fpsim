"""
Run tests on the calibration object.
"""

import sciris as sc
import fpsim as fp
import fp_analyses as fa

do_plot = 0


def test_multisim(do_plot=False):
    ''' Try running a multisim '''
    sc.heading('Testing multisim...')

    sims = []
    for i in range(3):
        pars = fa.senegal_parameters.make_pars()
        pars['n'] = 500
        pars['verbose'] = 0.1
        pars['exposure_correction'] = 0.5 + 0.5*i
        sim = fp.Sim(pars=pars)
        sims.append(sim)

    msim = fp.MultiSim(sims)
    msim.run()

    births = msim.results.births
    assert sum(births.low) < sum(births.high), 'Expecting the higher bound of births to be higher than the lower bound'

    return msim


if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    T = sc.tic()

    msim = test_multisim(do_plot=do_plot)

    print('\n'*2)
    sc.toc(T)
    print('Done.')