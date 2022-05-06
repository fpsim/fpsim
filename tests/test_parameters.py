"""
Run tests on individual parameters.
"""

import sciris as sc
import fpsim as fp
import fp_analyses as fa

def get_pars():
    return fa.senegal_parameters.make_pars()


def make(pars, n=100, verbose=0, do_run=True, **kwargs):
    '''
    Define a default simulation for testing the baseline.
    '''

    pars['n'] = n
    pars['verbose'] = verbose
    pars.update(kwargs)
    sim = fp.Sim(pars=pars)

    if do_run:
        sim.run()

    return sim


def test_null(do_plot=False):
    sc.heading('Testing no births, no deaths...')

    pars = get_pars() # For default pars
    pars['age_mortality']['f'] *= 0
    pars['age_mortality']['m'] *= 0
    pars['age_mortality']['trend'] *= 0
    pars['maternal_mortality']['probs'] *= 0
    pars['infant_mortality']['probs'] *= 0
    pars['exposure_correction'] = 0
    pars['high_parity']         = 4
    pars['high_parity_nonuse_correction']  = 0

    sim = make(pars)

    if do_plot:
        sim.plot()

    return sim


if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    T = sc.tic()

    do_plot = True
    null = test_null(do_plot=do_plot)

    print('\n'*2)
    sc.toc(T)
    print('Done.')
