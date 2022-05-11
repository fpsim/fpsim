"""
Run tests on individual parameters.
"""

import sciris as sc
import fpsim as fp

do_plot = True
sc.options(backend='agg') # Turn off interactive plots

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


def test_null(do_plot=do_plot):
    sc.heading('Testing no births, no deaths...')

    pars = fp.pars() # For default pars
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


def test_method_timestep():
    sc.heading('Test sim speed')

    p = dict(n=100, verbose=0, start_year=2000, end_year=2010)
    pars1 = fp.pars(method_timestep=1, **p)
    pars2 = fp.pars(method_timestep=6, **p)
    sim1 = fp.Sim(pars1)
    sim2 = fp.Sim(pars2)

    T = sc.timer()

    sim1.run()
    t1 = T.toctic(output=True)

    sim2.run()
    t2 = T.toc(output=True)

    assert t2 < t1, 'Expecting runtime to be less with a larger method timestep'

    return [t1, t2]



if __name__ == '__main__':

    sc.options(backend=None) # Turn on interactive plots
    with sc.timer():
        null = test_null(do_plot=do_plot)
        timings = test_method_timestep()
