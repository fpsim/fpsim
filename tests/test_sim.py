"""
Test running sims
"""

import fpsim as fp
import sciris as sc
import numpy as np

par_kwargs = dict(n_agents=100, start_year=2000, end_year=2010, seed=1, verbose=1)


def test_simple():
    sc.heading('Test simplest possible FPsim run')
    sim = fp.Sim()
    sim.run()
    sim.plot()
    return sim


def test_random_methods():
    sc.heading('Choose method annually based on age and parity')
    coefficients = sc.objdict(intercept=.1, age=2, parity=3)
    method_choice = fp.SimpleChoice(coefficients)
    pars = fp.pars(location='kenya', **par_kwargs)
    s = fp.Sim(pars, contraception_module=method_choice)
    s.run()
    return s


def test_methods():
    sc.heading('Test time on method')

    # Define new modules
    ms = fp.EmpoweredChoice(contra_use_file='contra_coef.csv', method_choice_file='method_mix.csv')
    emp = fp.Empowerment(empowerment_file='empower_coef.csv')
    edu = fp.Education()

    # Define pars
    pars = fp.pars(location='kenya', **par_kwargs)

    # Make and run sim
    s = fp.Sim(pars, contraception_module=ms, empowerment_module=emp, education_module=edu)
    s.run()

    return s


if __name__ == '__main__':

    s0 = test_simple()
    s1 = test_random_methods()
    s2 = test_methods()
