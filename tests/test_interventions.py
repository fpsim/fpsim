"""
Run tests on the calibration object.
"""

import sciris as sc
import fpsim as fp
import fp_analyses as fa


def make_sim(n=100, **kwargs):
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

    sim = make_sim(interventions=test_interv)
    sim.run()

    return sim


def test_analyzers():
    ''' Test analyzers '''
    sc.heading('Testing analyzers...')

    sim = make_sim(analyzers=fp.snapshot(timesteps=[100, 200]))
    sim.run()

    return sim


if __name__ == '__main__':

    isim = test_interventions()
    asim = test_analyzers()
