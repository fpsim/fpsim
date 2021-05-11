"""
Run tests on the calibration object.
"""

import sciris as sc
import fpsim as fp
import fp_analyses as fa

do_plot = 0

def make_sim(n=500, **kwargs):
    '''
    Define a default simulation for testing the baseline.
    '''
    pars = fa.senegal_parameters.make_pars()
    pars['n'] = n
    pars['verbose'] = 0.1
    pars.update(kwargs)
    sim = fp.Sim(pars=pars)

    return sim


def test_interventions():
    ''' Test interventions '''
    sc.heading('Testing interventions...')

    def test_interv(sim):
        if sim.i == 100:
            print(f'Success on day {sim.t}/{sim.y}')

    pars = dict(
        interventions = [test_interv],
    )

    sim = make_sim(**pars)
    sim.run()

    return sim


def test_analyzers():
    ''' Test analyzers '''
    sc.heading('Testing analyzers...')

    pars = dict(
        analyzers = [fp.snapshot(timesteps=[100, 200])],
    )

    sim = make_sim(**pars)
    sim.run()

    return sim


if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    T = sc.tic()

    sim1 = test_interventions()
    sim2 = test_analyzers()

    print('\n'*2)
    sc.toc(T)
    print('Done.')