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
    sim = fp.Sim(location=location, test=True)
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
    fp_pars = dict(postpartum_dur=24)
    sim1 = fp.Sim(sim_pars=par_kwargs, fp_pars=fp_pars, contra_pars=contra_pars, edu_pars=edu_pars, location='kenya')
    sim1.init()

    assert sim1.pars.n_agents == 500, "Sim par failed"
    assert sim1.connectors.contraception.pars.prob_use_year == contra_pars['prob_use_year'], "Contraception par failed"
    assert sim1.connectors.edu.pars.init_dropout.pars.p == edu_pars['init_dropout'], "Education par failed"
    assert sim1.pars.fp.postpartum_dur == fp_pars['postpartum_dur'], "FP par failed"

    # Test 2: separate modules
    contra_mod = fp.SimpleChoice(location='kenya', prob_use_trend_par=0.3)
    edu_mod = fp.Education(location='kenya', init_dropout=0.1)

    sim2 = fp.Sim(pars=par_kwargs, postpartum_dur=21, contraception_module=contra_mod, education_module=edu_mod, location='kenya')
    sim2.init()

    assert sim2.connectors.contraception.pars.prob_use_trend_par == 0.3, "Contraception par failed"
    assert sim2.connectors.edu.pars.init_dropout.pars.p == 0.1, "Education par failed"
    assert sim2.pars.fp.postpartum_dur == 21, "FP par failed"

    # Test 3: flat pars dict
    pars = dict(
        start=2010,  # Sim par
        postpartum_dur=18,  # FP par
        prob_use_intercept=0.5,  # Contraception par
        init_dropout=0.15,  # Education par
        location='kenya',
    )

    sim3 = fp.Sim(**pars)
    sim3.init()

    assert sim3.connectors.contraception.pars.prob_use_intercept == 0.5, "Contraception par failed"
    assert sim3.connectors.edu.pars.init_dropout.pars.p == 0.15, "Education par failed"
    assert sim3.pars.fp.postpartum_dur == 18, "FP par failed"

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

    # sim = test_simple('senegal')
    # s1 = test_random_choice()
    # sims1 = test_simple_choice()
    # sims2 = test_mid_choice()
    # test_sim_creation()
    # exp = test_senegal()

    print('Done.')

    import fpsim as fp
    import sciris as sc
    import starsim as ss

    pars = dict(
        n_agents   = 1_000,
        location   = 'kenya',
        start_year = 2000,
        end_year   = 2020,
        exposure_factor = 1.0  # Overall scale factor on probability of becoming pregnant
    )
    method_choice = fp.RandomChoice()
    s1 = fp.Sim(pars=pars, contraception_module=method_choice, label="Baseline")

    Methods = fp.make_methods()
    for method in Methods.values(): print(f"{method.idx}: {method.label}")
    change_efficacy_intervention = fp.update_methods(eff={"Injectables": 0.99}, year=2010)  # new efficacy starts in 2010

    s2 = fp.Sim(pars=pars, contraception_module=method_choice,
                     interventions=change_efficacy_intervention,
                     label="More effective Injectables")

    # The baseline duration for Injectables is a lognormal with parameter par1=2, and par2=3
    change_duration_intervention = fp.update_methods(dur_use={'Injectables': dict(dist='lognormal', par1=3, par2=0.2)}, year=2010)

    # Define a simulaiton for this intervention called s3
    s3 = fp.Sim(pars=pars, contraception_module=method_choice,
                     interventions=change_duration_intervention,
                     label="Longer time on Injectables")

    # The values in method_mix should add up to 1, but if they don't, the intervention update_methods() will autamotailly normalize them to add up to 1.
    change_mix = fp.update_methods(method_mix=[0.25, 0.05, 0.05, 0.0, 0.05, 0.3, 0.1, 0.1, 0.0], year=2010.0)

    # Define a simulation for this intervention called s4
    s4 = fp.Sim(pars=pars, contraception_module=method_choice,
                interventions=change_mix,
                label='Different mix')
    simlist = sc.autolist([s1, s2, s3, s4])
    msim = ss.MultiSim(sims=simlist)
    msim.run(parallel=False)

    msim.plot(key='cpr')