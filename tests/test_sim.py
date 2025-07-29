"""
Test running sims
"""

import fpsim as fp
import sciris as sc


# par_kwargs = dict(n_agents=1000, start_year=1960, end_year=2020, seed=1, verbose=1)
par_kwargs = dict(n_agents=500, start=2000, stop=2010, unit='year', dt=1/12, rand_seed=1, verbose=1/12)
parallel = 0  # Whether to run in serial (for debugging)


def test_simple(location='kenya'):
    sc.heading('Test simplest possible FPsim run')
    sim = fp.Sim(location=location)
    sim.run()
    return sim


def test_random_choice():
    sc.heading('Random method choice')

    rc = fp.RandomChoice()
    sim = fp.Sim(pars=par_kwargs, location='kenya', contraception_module=rc)
    sim.run()
    print(f'✓ (successfully ran RandomChoice)')

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


def test_sim_creation():
    sc.heading('Test creating a sim in different ways')

    contra_pars = dict(prob_use_year=2000)
    edu_pars = dict(init_dropout=0.2)
    fp_pars = dict(postpartum_dur=24)

    # Test 1: par passed in separate dicts
    sim1 = fp.Sim(pars=par_kwargs, fp_pars=fp_pars, contra_pars=contra_pars, edu_pars=edu_pars, location='kenya')

    sim1.init()

    # assert sim1.diseases.ng.pars.eff_condom == 0.6, "Disease parameter not set correctly"
    # assert len(sim1.diseases) == 5, "Incorrect number of diseases initialized"
    # assert len(sim1.connectors) > 0, "No connectors initialized"
    #
    # demographics = [sti.Pregnancy(), ss.Deaths()]  # Replace the default ss.Pregnancy module with the sti one
    # networks = sti.StructuredSexual()
    # diseases = [sti.Gonorrhea(), 'hiv']
    #
    # sim2 = sti.Sim(
    #     pars=pars,
    #     networks=networks,
    #     demographics=demographics,
    #     diseases=diseases,
    #     connectors=True,
    # )
    #
    # sim2.init()
    #
    # assert isinstance(sim2.networks.structuredsexual, sti.StructuredSexual), "Network not initialized correctly"
    # assert len(sim2.diseases) == 2, "Incorrect number of diseases initialized"
    # assert len(sim2.connectors) > 0, "No connectors initialized"
    # assert len(sim2.demographics) == 2, "Incorrect number of demographics initialized"
    #
    # # Test 3: flat pars dict
    # pars = dict(
    #     start=2010,  # Sim par
    #     beta_m2f=0.05,  # STI parameter applied to all STIs
    #     prop_f0=0.45,
    #     location='zimbabwe',
    #     datafolder='./test_data/',
    #     diseases=['ng', 'ct', 'tv'],
    #     ng=dict(eff_condom=0.6),  # Gonorrhea-specific parameter
    # )
    #
    # sim3 = sti.Sim(**pars)
    # sim3.init()
    #
    # assert sim3.diseases.ng.pars.beta_m2f == pars['beta_m2f'], "Disease parameter not set correctly"
    # assert sim3.diseases.ct.pars.beta_m2f == pars['beta_m2f'], "Disease parameter not set correctly"
    # assert sim3.diseases.ng.pars.eff_condom == pars['ng']['eff_condom'], "Disease parameter not set correctly"
    # assert sim3.networks.structuredsexual.pars.prop_f0 == pars['prop_f0'], "Network parameter not set correctly"
    # assert len(sim3.networks) == 2, "Default networks not added"
    # assert len(sim3.diseases) == 3, "Incorrect number of diseases initialized"

    return


if __name__ == '__main__':

    # s0 = test_simple('ethiopia')
    # s1 = test_random_choice()
    # sims1 = test_simple_choice()
    # sims2 = test_mid_choice()
    test_sim_creation()
    print('Done.')
