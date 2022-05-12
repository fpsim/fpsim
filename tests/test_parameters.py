"""
Run tests on individual parameters.
"""

import sciris as sc
import fpsim as fp

do_plot = True
sc.options(backend='agg') # Turn off interactive plots


def test_null(do_plot=do_plot):
    sc.heading('Testing no births, no deaths...')

    pars = fp.pars('test') # For default pars
    pars['age_mortality']['f'] *= 0
    pars['age_mortality']['m'] *= 0
    pars['age_mortality']['trend'] *= 0
    pars['maternal_mortality']['probs'] *= 0
    pars['infant_mortality']['probs'] *= 0
    pars['exposure_correction'] = 0
    pars['high_parity']         = 4
    pars['high_parity_nonuse_correction']  = 0

    sim = fp.Sim(pars)
    sim.run()

    if do_plot:
        sim.plot()

    return sim


def test_method_timestep():
    sc.heading('Test sim speed')

    pars1 = fp.pars(location='test', method_timestep=1)
    pars2 = fp.pars(location='test', method_timestep=6)
    sim1 = fp.Sim(pars1)
    sim2 = fp.Sim(pars2)

    T = sc.timer()

    sim1.run()
    t1 = T.tt(output=True)

    sim2.run()
    t2 = T.tt(output=True)

    assert t2 < t1, 'Expecting runtime to be less with a larger method timestep'

    return [t1, t2]



if __name__ == '__main__':

    sc.options(backend=None) # Turn on interactive plots
    with sc.timer():
        null = test_null(do_plot=do_plot)
        timings = test_method_timestep()
