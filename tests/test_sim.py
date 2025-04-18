"""
Test running sims
"""

import fpsim as fp
import sciris as sc


# par_kwargs = dict(n_agents=1000, start_year=1960, end_year=2020, seed=1, verbose=1)
par_kwargs = dict(n_agents=500, start=2000, stop=2010, unit='year', dt=1/12, rand_seed=1, verbose=-1)
parallel = 0  # Whether to run in serial (for debugging)

def test_simple(location='kenya'):
    sc.heading('Test simplest possible FPsim run')
    sim = fp.Sim(location=location)
    sim.run()
    return sim


def test_simple_choice():
    sc.heading('Method choice is based on age & previous method')

    # Make & run sim
    sims = sc.autolist()
    for location in ['kenya', 'ethiopia', 'senegal']:
        method_choice = fp.SimpleChoice(location=location)
        sim = fp.Sim(pars=par_kwargs, location=location, contraception_module=method_choice, analyzers=fp.cpr_by_age())
        sims += sim

    m = fp.parallel(sims, parallel=parallel, compute_stats=False)
    print(f'✓ (successfully ran SimpleChoice)')

    return m.sims


def test_mid_choice():
    sc.heading('Test sims with default contraceptive choice module')

    sims = sc.autolist()
    for location in ['kenya', 'ethiopia', 'senegal']:
        ms = fp.StandardChoice(location=location)
        edu = fp.Education(location=location)
        s = fp.Sim(pars=par_kwargs, location=location, contraception_module=ms, education_module=edu)
        sims += s

    m = fp.parallel(sims, parallel=parallel, compute_stats=False)
    print(f'✓ (successfully ran StandardChoice)')

    return m.sims


if __name__ == '__main__':

    s0 = test_simple('ethiopia')
    sims1 = test_simple_choice()
    sims2 = test_mid_choice()
    print('Done.')
