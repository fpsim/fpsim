"""
Test running sims
"""

import fpsim as fp
import sciris as sc
import starsim as ss


par_kwargs = dict(test=True)
parallel = 1  # Whether to run in serial (for debugging)


def test_simple(location='kenya'):
    sc.heading('Test simplest possible FPsim run')
    sim = fp.Sim(test=True)
    sim.run()
    sim.to_df(resample='year', use_years=True)
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
    for location in ['kenya', 'ethiopia', 'senegal', 'amhara']:
        method_choice = fp.SimpleChoice(location=location)
        sim = fp.Sim(pars=par_kwargs, location=location, contraception_module=method_choice, analyzers=fp.cpr_by_age())
        sims += sim

    m = ss.parallel(sims, parallel=parallel)
    print(f'✓ (successfully ran SimpleChoice)')

    return m.sims


def test_mid_choice():
    sc.heading('Test sims with default contraceptive choice module')

    sims = sc.autolist()
    for location in ['kenya', 'ethiopia', 'senegal', 'amhara']:
        ms = fp.StandardChoice(location=location)
        edu = fp.Education(location=location)
        s = fp.Sim(pars=par_kwargs, location=location, contraception_module=ms, education_module=edu)
        sims += s

    m = ss.parallel(sims, parallel=parallel)
    print(f'✓ (successfully ran StandardChoice)')

    return m.sims


def test_sim_creation():
    sc.heading('Test creating a sim in different ways')

    # Test 1: par passed in separate dicts
    contra_pars = dict(prob_use_year=2000)
    edu_pars = dict(init_dropout=0.2)
    fp_pars = dict(primary_infertility=0.1)
    sim1 = fp.Sim(sim_pars=par_kwargs, fp_pars=fp_pars, contra_pars=contra_pars, edu_pars=edu_pars, location='kenya')
    sim1.init()

    assert sim1.pars.n_agents == 500, "Sim par failed"
    assert sim1.connectors.contraception.pars.prob_use_year == contra_pars['prob_use_year'], "Contraception par failed"
    assert sim1.connectors.edu.pars.init_dropout.pars.p == edu_pars['init_dropout'], "Education par failed"
    assert sim1.pars.fp.primary_infertility == fp_pars['primary_infertility'], "FP par failed"

    # Test 2: separate modules
    contra_mod = fp.SimpleChoice(location='kenya', prob_use_trend_par=0.3)
    edu_mod = fp.Education(location='kenya', init_dropout=0.1)

    sim2 = fp.Sim(pars=par_kwargs, primary_infertility=0.1, contraception_module=contra_mod, education_module=edu_mod, location='kenya')
    sim2.init()

    assert sim2.connectors.contraception.pars.prob_use_trend_par == 0.3, "Contraception par failed"
    assert sim2.connectors.edu.pars.init_dropout.pars.p == 0.1, "Education par failed"
    assert sim2.pars.fp.primary_infertility == 0.1, "FP par failed"

    # Test 3: flat pars dict
    pars = dict(
        start=2010,  # Sim par
        primary_infertility=0.18,  # FP par
        prob_use_intercept=0.5,  # Contraception par
        init_dropout=0.15,  # Education par
        location='kenya',
    )

    sim3 = fp.Sim(**pars)
    sim3.init()

    assert sim3.connectors.contraception.pars.prob_use_intercept == 0.5, "Contraception par failed"
    assert sim3.connectors.edu.pars.init_dropout.pars.p == 0.15, "Education par failed"
    assert sim3.pars.fp.primary_infertility == 0.18, "FP par failed"

    # Test 4: mixed types: some flat pars, some in dicts, and some modules
    fp_pars = dict(short_int=20)
    contra_pars = dict(prob_use_year=2010)
    education_module = fp.Education(location='kenya', age_start=7)
    sim4 = fp.Sim(pars, fp_pars=fp_pars, contra_pars=contra_pars, education_module=education_module)
    sim4.init()

    assert sim4.connectors.contraception.pars.prob_use_intercept == 0.5, "Contraception par failed"
    assert sim4.connectors.contraception.pars.prob_use_year == 2010, "Contraception par failed"
    assert sim4.connectors.edu.pars.age_start == 7, "Education par failed"
    assert sim4.pars.fp.short_int == 20, "FP par failed"
    assert sim4.pars.fp.primary_infertility == 0.18, "FP par failed"

    print('✓ (successfully created sims with different methods)')

    return


def test_senegal():
    sc.heading('Test Senegal sim')

    # Make the sim
    exp = fp.Experiment(pars=dict(location='senegal'))
    exp.run()
    exp.summarize()

    print(f'✓ (successfully ran Senegal sim)')

    return exp


if __name__ == '__main__':

    sim = test_simple('senegal')
    s1 = test_random_choice()
    sims1 = test_simple_choice()
    sims2 = test_mid_choice()
    test_sim_creation()
    # exp = test_senegal()  # NOT WORKING - need to re-enable loading of calibration data

    print('Done.')
