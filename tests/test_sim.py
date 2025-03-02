"""
Test running sims
"""

import fpsim as fp
import sciris as sc
import pylab as pl
import numpy as np

# par_kwargs = dict(n_agents=1000, start_year=1960, end_year=2020, seed=1, verbose=1)
par_kwargs = dict(n_agents=500, start_year=2000, end_year=2010, seed=1, verbose=-1)
serial = 1  # Whether to run in serial (for debugging)


def test_simple():
    sc.heading('Test simplest possible FPsim run')
    sim = fp.Sim()
    sim.run()
    return sim


def test_simple_choice():
    sc.heading('Method choice is based on age & previous method')

    # Make & run sim
    sims = sc.autolist()
    for location in ['kenya', 'ethiopia', 'senegal']:
        pars = fp.pars(location=location, **par_kwargs)
        method_choice = fp.SimpleChoice(location=location, methods=sc.dcp(fp.Methods))
        sim = fp.Sim(pars, contraception_module=method_choice, analyzers=fp.cpr_by_age())
        sims += sim

    m = fp.parallel(sims, serial=serial, compute_stats=False)
    print(f'✓ (successfully ran SimpleChoice)')

    return m.sims


def test_mid_choice():
    sc.heading('Test sims with default contraceptive choice module')

    sims = sc.autolist()
    for location in ['kenya', 'ethiopia', 'senegal']:
        ms = fp.StandardChoice(location=location, methods=sc.dcp(fp.Methods))
        edu = fp.Education(location=location)
        pars = fp.pars(location=location, **par_kwargs)
        s = fp.Sim(pars, contraception_module=ms, education_module=edu)
        sims += s

    m = fp.parallel(sims, serial=serial, compute_stats=False)
    print(f'✓ (successfully ran StandardChoice)')

    return m.sims


if __name__ == '__main__':

    s0 = test_simple()
    sims1 = test_simple_choice()
    sims2 = test_mid_choice()
    print('Done.')
