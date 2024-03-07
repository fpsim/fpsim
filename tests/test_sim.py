"""
Test running sims
"""

import fpsim as fp
import sciris as sc
import numpy as np

par_kwargs = dict(n_agents=100, start_year=2000, end_year=2010, seed=1, verbose=1)


def test_simple():
    sc.heading('Test simplest possible FPsim run')
    sim = fp.Sim(location='test')
    sim.run()
    sim.plot()
    return sim


def test_simple_choice():
    sc.heading('Choose method annually based on age and parity')
    coefficients = sc.objdict(intercept=.1, age=2, parity=3)
    method_choice = fp.SimpleChoice(coefficients)
    pars = fp.pars(location='kenya', **par_kwargs)
    s = fp.Sim(pars, contraception_module=method_choice)
    s.run()
    return s


def test_empowered_choice():
    sc.heading('Test sim with empowerment')

    # Define new modules
    ms = fp.EmpoweredChoice(location='kenya')
    emp = fp.Empowerment(location='kenya')
    edu = fp.Education(location='kenya')

    # Define pars
    pars = fp.pars(location='kenya', **par_kwargs)

    # Make and run sim
    s = fp.Sim(pars, contraception_module=ms, empowerment_module=emp, education_module=edu)
    s.run()

    return s


if __name__ == '__main__':

    s0 = test_simple()
    # s1 = test_simple_choice()
    # s2 = test_empowered_choice()
