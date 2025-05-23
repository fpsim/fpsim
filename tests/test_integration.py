"""
Test different components of the simulation behave as expected
"""

# Imports
import sciris as sc
import fpsim as fp
from fpsim import utils as fpu
from fpsim import people as fpppl
from fpsim import methods as fpm
from fpsim import interventions as fpi

import numpy as np
import pylab as pl
import starsim as ss

serial = 1  # For testing purposes


def custom_init(sim, force=False, init_contra=True, age=None, sex=None, empowerment_module=None, education_module=None, person_defaults=None):
    if force or not sim.initialized:
        sim.ti = 0  # The current time index
        fpu.set_seed(sim['seed'])
        sim.init_results()
        sim.people=fpppl.People(pars=sim.pars, age=age, sex=sex, empowerment_module=empowerment_module, education_module=education_module, person_defaults=person_defaults )  # This step also initializes the empowerment and education modules if provided
        if init_contra: sim.init_contraception()  # Initialize contraceptive methods
        sim.initialized = True
        sim.pars['verbose'] = -1
    return sim


def test_pregnant_women():
    sc.heading('Test pregnancy and birth outcomes... ')

    # start out 24f, no contra, no pregnancy, active, fertile, has_intent
    methods = ss.ndict([
        fpm.Method(name='none', efficacy=0, modern=False, dur_use=fpm.ln(2, 3), label='None'),
        fpm.Method(name='test', efficacy=0.0, modern=True,  dur_use=fpm.ln(2, 3), label='Test'),
    ])
    contra_mod = fpm.RandomChoice(methods=methods)

    # force all to have debuted
    debut_age = {}
    debut_age['ages']=np.arange(10, 45, dtype=float)
    debut_age['probs']=np.zeros(35, dtype=float)
    debut_age['probs'][10:20] = 1.0

    # force all women to have the same fecundity and be sexually active
    # Note: not all agents will be active at t==0 but will be after t==1
    sexual_activity = np.zeros(51, dtype=float)
    sexual_activity[20:30] = 1.0

    custom_pars = {
        'start_year': 2000,
        'end_year': 2001,
        'n_agents': 1000,
        'primary_infertility': 0,
    }

    sim = fp.Sim(pars=custom_pars, contraception_module=contra_mod, sexual_activity=sexual_activity, debut_age=debut_age)
    sim = custom_init(sim, age=24, sex=0, person_defaults={'fertility_intent':True})

    # Override fecundity to maximize pregnancies and minimize variation during test
    sim.people.personal_fecundity[:] = 1

    sim.run()

    n_agents = sim.pars['n_agents']

    # all women are 24 and can become pregnant, so we can look at percentages to estimate the
    # expected number of pregnancies, stillborns, living infants, and miscarriages/abortions
    # pregnancy rate: 11.2% per month or approx 80% per year
    print(f'Checking pregnancy and birth outcomes from {n_agents} women... ')
    assert 0.85 * n_agents > sim.results.pregnancies[0:12].sum() > 0.75 * n_agents, "Expected number of pregnancies not met"
    print(f'✓ ({sim.results.pregnancies[0:12].sum()} pregnancies, as expected)')

    # stillbirth rate: 2.5% of all pregnancies, but only count completed pregnancies
    assert 0.06 * sim.results.pregnancies[0:3].sum() > sim.results.stillbirths[9:12].sum() > 0.01 * sim.results.pregnancies[0:3].sum() , "Expected number of stillbirths not met"
    print(f'✓ ({sim.results.stillbirths[9:12].sum()} stillbirths, as expected)')

    # miscarriage rate: 10% of all pregnancies. Miscarriages are calculated at end of 1st trimester, so pregnancies from months 0-9
    # have miscarriages from months 3-12.
    pregnancies = sim.results.pregnancies[0:9].sum()
    miscarriages = sim.results.miscarriages[3:12].sum()
    assert 0.15 * pregnancies > miscarriages > 0.05 * pregnancies, "Expected number of miscarriages not met"
    print(f'✓ ({miscarriages} miscarriages, as expected)')

    # abortion rate: 8% of all pregnancies
    # abortions occur at same timestep as conception, so all pregnancies were checked for abortions
    pregnancies = sim.results.pregnancies.sum()
    abortions = sim.results.abortions.sum()
    assert 0.13 * pregnancies > abortions > 0.03 * pregnancies, "Expected number of abortions not met"
    print(f'✓ ({abortions} abortions, as expected)')

    # live birth rate: 79.5% of all pregnancies
    # no premature births so all births are full term
    pregnancies = sim.results.pregnancies[0:3].sum()
    live_births = sim.results.births[9:12].sum()
    assert 0.85 * pregnancies > live_births > 0.75 * pregnancies, "Expected number of live births not met"
    print(f'✓ ({live_births} live births from {pregnancies} pregnancies, as expected)')

    return sim


def test_contraception():
    sc.heading('Test contraception prevents pregnancy... ')

    # start out 24f, on contra, no starting pregnancy, active, fertile, high fecundity.
    # test 1: Assume contraception 100% effective, 100% uptake -> no pregnancies.
    # test 2: contraception use restarts after postpartum period

    methods = ss.ndict([
        fpm.Method(name='none', efficacy=0, modern=False, dur_use=fpm.ln(2, 3), label='None'),
        fpm.Method(name='test',     efficacy=1.0, modern=True,  dur_use=fpm.ln(2, 3), label='Test'),
    ])
    contra_mod = fpm.RandomChoice(methods=methods, pars={'p_use': 1.0})

    # force all to have debuted
    debut_age = {}
    debut_age['ages']=np.arange(10, 45, dtype=float)
    debut_age['probs']=np.zeros(35, dtype=float)
    debut_age['probs'][10:20] = 1.0

    sexual_activity = np.zeros(51, dtype=float)
    sexual_activity[20:30] = 1.0

    # after 12 months, stop using any contraception, so we should see pregnancies after the switch
    p_use_change = fpi.update_methods(year=2001, p_use=0.0, )

    # after 24 months, we should be seeing pregnancies again so we can reenable contraception use rates and check postpartum uptake
    p_use_change2 = fpi.update_methods(year=2002, p_use=0.5, )

    custom_pars = {
        'start_year': 2000,
        'end_year': 2003,
        'n_agents': 1000,
        'primary_infertility': 0,
    }

    sim = fp.Sim(pars=custom_pars, contraception_module=contra_mod, sexual_activity=sexual_activity, debut_age=debut_age, interventions=[p_use_change, p_use_change2])
    sim = custom_init(sim, age=24, sex=0, person_defaults={'fertility_intent': False})

    # Override fecundity to maximize pregnancies and minimize variation during test
    sim.people.personal_fecundity[:] = 1

    sim.run()

    print(f'Checking pregnancy and birth outcomes from {custom_pars["n_agents"]} women... ')
    assert sim.results.pregnancies[0:12].sum() == 0, "Expected no pregnancies"
    assert sim.results.pregnancies[12:].sum() > 0, "Expected pregnancies after contraception switch"
    print(f'✓ (no pregnancies with 100% effective contraception)')

    pp1 = sim.people.filter(sim.people.postpartum_dur==1)
    assert pp1['on_contra'].sum() == 0, "Expected no contraception use immediately postpartum"
    print(f'✓ (no contraception use postpartum)')
    pp2plus = sim.people.filter(sim.people.postpartum_dur == 2 )
    assert 0.55 > pp2plus['on_contra'].sum()/len(pp2plus) > 0.45, "Expected contraception use rate to be approximately = p_use 1 month after postpartum period"
    assert (sim.people.on_contra==True).sum() < sim.pars['n_agents'], "Expected some agents to be off of birth control at any given time"
    print(f'✓ (contraception use rate {(pp2plus["on_contra"].sum()/len(pp2plus)):.2f}, as expected)')

    return sim


def test_simplechoice_contraception_dependencies():
    # There are a number of different factors that affect contraception use, including previous use, age, etc.
    # All agents are women, initialized at age 15, all active, all infertile.
    # sim1: Age impacts
    #    no parity, all have used contra before
    #    Init to all 15yo female agents, run for 40 yr to see how age impacts use
    # sim2: Previous use impacts
    #    used contra previously, all ages.
    # sim3: Previous use impacts
    #    none have used contra. all ages.
    sc.heading('Testing contraception dependencies... ')

    custom_pars = {
        'start_year': 2000,
        'end_year': 2005,
        'n_agents': 1000,
        'primary_infertility': 1, # make sure no pregnancies!
    }

    cm_pars = dict(
        prob_use_trend_par=0.0, # no change in use trend, so changes should be driven by other factors
        force_choose=False,
    )
    method = fpm.SimpleChoice(pars=cm_pars, location="kenya")
    analyzers = fp.cpr_by_age()

    # force all to have debuted and sexually active
    debut_age = {
        'ages': np.arange(10, 49, dtype=float),
        'probs': np.ones(35, dtype=float)
    }

    # Note: not all agents will be active at t==0 but will be after t==1
    sexual_activity = np.ones(51, dtype=float)

    # Set up all sims to run in parallel
    # custom init forces age, sex and other person defaults
    # p_use by age: <18: ~.18, 18-20: ~.49, 20-25: ~.54, 25-35: ~.52, 35-50: ~.32
    pars1 = sc.dcp(custom_pars)
    pars1['end_year'] = 2040  # run for 40 years to see how age impacts use
    sim1 = fp.Sim(location="kenya", pars=pars1, contraception_module=sc.dcp(method), sexual_activity=sexual_activity, debut_age=debut_age, analyzers=sc.dcp(analyzers))
    sim1 = custom_init(sim1, age=15, sex=0, person_defaults={'ever_used_contra':True})

    sim2 = fp.Sim(location="kenya", pars=custom_pars, contraception_module=sc.dcp(method), sexual_activity=sexual_activity, debut_age=debut_age, analyzers=sc.dcp(analyzers))
    sim2 = custom_init(sim2, sex=0, person_defaults={'ever_used_contra': True})

    sim3 = fp.Sim(location="kenya", pars=custom_pars, contraception_module=sc.dcp(method), sexual_activity=sexual_activity, debut_age=debut_age, analyzers=sc.dcp(analyzers))
    sim3 = custom_init(sim3, sex=0, person_defaults={'ever_used_contra': False})

    # Run all sims
    msim = fp.parallel(sim1, sim2, sim3, serial=serial, compute_stats=False)
    sim1, sim2, sim3 = msim.sims

    # CHECK SIM 1
    # If CPR responded instantly to changes in p_use, then CPR by age should increase from u18->u20, increase from
    # u20->u25, decrease slightly from u25->u35, then decrease more from u35+. However, the rate of change in CPR is
    # more gradual than the change in p_use because the duration of use varies and can be many years long. The direction
    # of change is determined by the cpr when p_use changes and the new value of p_use. Generally speaking, trend should
    # be positive until > 35.

    # Note: u18 tends to have artifacts from the test's initial conditions that haven't stabilized yet, so it's an unreliable indicator of test success
    cpr_by_age = sim1['analyzers'].results
    u_18_nonzero = cpr_by_age['<18'][np.nonzero(cpr_by_age['<18'])]
    m_u18, b = np.polyfit(np.arange(len(u_18_nonzero)), u_18_nonzero, 1)

    u_20_nonzero = cpr_by_age['18-20'][np.nonzero(cpr_by_age['18-20'])]
    m_u20, b = np.polyfit(np.arange(len(u_20_nonzero)), u_20_nonzero, 1)

    u_25_nonzero = cpr_by_age['20-25'][np.nonzero(cpr_by_age['20-25'])]
    m_u25, b = np.polyfit(np.arange(len(u_25_nonzero)), u_25_nonzero, 1)

    u_35_nonzero = cpr_by_age['25-35'][np.nonzero(cpr_by_age['25-35'])]
    m_u35, b = np.polyfit(np.arange(len(u_35_nonzero)), u_35_nonzero, 1)

    o_35_nonzero = cpr_by_age['>35'][np.nonzero(cpr_by_age['>35'])]
    m_o35, b = np.polyfit(np.arange(len(o_35_nonzero)), o_35_nonzero, 1)

    print(f"Printing CPR trends ... ")
    # NB, we expect CPR to increase from u18->u20, u20->u25, u25->u35, and decrease from u35+."
    # The assert statements below have been commented out while we figure out a robust way to test this.
    # assert m_u20 > 0, "Expected CPR to increase over time in the 18->20 cohort"
    # assert m_u25 > 0, "Expected CPR to increase from u25->u35"
    # assert m_u35 > 0, "Expected CPR to increase from u35->u35+"
    # assert m_o35 < 0, "Expected CPR to decrease from u35->u35+"
    print(f'CPR trends: 18-20: {m_u20:.4f}, 20-25: {m_u25:.2f}, 25-35: {m_u35:.2f}, 35+{m_o35:.2f})')

    # Contraception use is more likely if an agent has used contraception before. Sim2 assumes every agent has used
    # contra before, so its initial CPR will be higher than sim3 and all subsequent checks will be more likely to use
    # contraception too.
    # because trend is set to 0 and ages are distributed across the contra age ranges, we expect no trend in CPR over time
    m, b = np.polyfit(sim2.tvec, sim2.results.cpr, 1)
    assert 0.05 > m > -0.05, "Expected no trend in CPR over time, sim 2"

    m, b = np.polyfit(sim3.tvec, sim3.results.cpr, 1)
    assert 0.05 > m > -0.05, "Expected no trend in CPR over time, sim 3"

    # CPR should be higher in sim2 than sim3 at all time steps
    print(f"Checking CPR is higher if all agents are previous users ... ")
    diffs = np.nonzero((sim2.results.cpr - sim3.results.cpr) <0)[-1]
    diffyears = sim2.results['t'][diffs]
    assert np.all(sim2.results.cpr > sim3.results.cpr), f"Expected CPR to be higher in sim2 than sim3 but they differ: {diffs}, {diffyears}"
    print(f'✓ (CPR is increased by prior contra use)')

    plot_results(sim1)
    plot_results(sim2)
    plot_results(sim3)

    return sim1, sim2, sim3


def test_method_selection_dependencies():
    sc.heading('Testing method selection dependencies... ')

    # set up a bunch of identical women, same age, same history
    # manually assign a previous method and short method dur
    # inspect new methods -> should be distributed according to the method mix
    custom_pars = {
        'start_year': 2000,
        'end_year': 2001,
        'n_agents': 1000,
        'primary_infertility': 1,  # make sure no pregnancies!
    }

    cm_pars = dict(
        prob_use_trend_par=0.0,  # no change in use trend, so changes should be driven by other factors
        force_choose=False,
    )
    method = fpm.SimpleChoice(pars=cm_pars, location="kenya")

    cpr = fp.cpr_by_age()
    snapshots = fp.snapshot([1,2])

    # force all to have debuted and sexually active
    debut_age = {
        'ages': np.arange(10, 45, dtype=float),
        'probs': np.ones(35, dtype=float)
    }

    # Note: not all agents will be active at t==0 but will be after t==1
    sexual_activity = np.ones(51, dtype=float)

    sim1 = fp.Sim(location="kenya", pars=custom_pars, contraception_module=sc.dcp(method), sexual_activity=sexual_activity,
                  debut_age=debut_age, analyzers=[cpr, snapshots])

    # custom init forces age, sex and other person defaults
    # p_use by age: <18: ~.18, 18-20: ~.49, 20-25: ~.54, 25-35: ~.52, 35-50: ~.32
    sim1 = custom_init(sim1, age=15, sex=0, person_defaults={'ever_used_contra':True})
    sim1.run()

    contra_ending_t2 = sim1['analyzers'][1].snapshots['1']['ti_contra'] == 2
    ending_contra_type = sim1['analyzers'][1].snapshots['1']['method'][contra_ending_t2]
    new_contra_type = sim1['analyzers'][1].snapshots['2']['method'][contra_ending_t2]

    # Compare methods. All should have changed.
    print(f"Checking agents change methods at ti_contra ... ")
    assert np.all((ending_contra_type != new_contra_type) | ((ending_contra_type == 0) & (new_contra_type == 0))), "expected all agents at ti_contra to change contra method or remain on None"
    print(f'✓ (all agents change methods)')

    return sim1


def test_education_preg():
    sc.heading('Testing that lower fertility rate leads to more education...')

    def make_sim(pregnant=False):
        pars = dict(start_year=2000, end_year=2010, n_agents=1000, verbose=0.1)
        sim = fp.Sim(pars=pars)
        sim.initialize()
        sim.people.age[:] = 15
        sim.people.sex[:] = 0
        if pregnant:
            sim.people.pregnant[:] = True
            sim.people.method[:] = 0
            sim.people.on_contra[:] = False
            sim.people.ti_contra[:] = 12
        return sim

    sim_base = make_sim()
    sim_preg = make_sim(pregnant=True)
    m = fp.parallel([sim_base, sim_preg], serial=serial, compute_stats=False)
    sim_base, sim_preg = m.sims[:]  # Replace with run versions

    # Check that education has increased
    base_edu = sim_base.results.edu_attainment[-1]
    preg_edu = sim_preg.results.edu_attainment[-1]
    base_births = sum(sim_base.results.births)
    preg_births = sum(sim_preg.results.births)
    assert base_births < preg_births, f'With more pregnancy there should be more births, but {preg_births}<{base_births}'
    assert preg_edu < base_edu, f'With more pregnancy there should be lower education levels, but {preg_edu}>{base_edu}'
    print(f"✓ (Higher teen pregnancy ({preg_births:.0f} vs {base_births:.0f}) -> less education ({preg_edu:.2f} < {base_edu:.2f}))")

    return sim_base, sim_preg


def plot_results(sim):
    # Plots
    fig, axes = pl.subplots(2, 2, figsize=(10, 7))
    axes = axes.ravel()
    age_bins = [18, 20, 25, 35, 50]
    colors = sc.vectocolor(age_bins)
    cind = 0

    # mCPR
    ax = axes[0]
    ax.plot(sim.results.t, sim.results.cpr)
    ax.set_ylim([0, 1])
    ax.set_ylabel('CPR')
    ax.set_title('CPR')

    # mCPR by age
    ax = axes[1]
    for alabel, ares in sim['analyzers'].results.items():
        ax.plot(sim.results.t, ares, label=alabel, color=colors[cind])
        cind += 1
    ax.legend(loc='best', frameon=False)
    ax.set_ylim([0, 1])
    ax.set_ylabel('CPR')
    ax.set_title('CPR')

    # Plot method mix
    ax = axes[2]
    oc = sim.people.filter(sim.people.on_contra)
    method_props = [len(oc.filter(oc.method == i)) / len(oc) for i in range(1, 10)]
    method_labels = [m.name for m in sim.contraception_module.methods.values() if m.label != 'None']
    ax.bar(method_labels, method_props)
    ax.set_ylabel('Proportion among all users')
    ax.set_title('Contraceptive mix')

    sc.figlayout()
    pl.show()


if __name__ == '__main__':

    sc.options(interactive=False)
    s1 = test_pregnant_women()
    s2 = test_contraception()
    s3, s4, s5 = test_simplechoice_contraception_dependencies()
    s6 = test_method_selection_dependencies()
    s7, s8 = test_education_preg()
    print("All tests passed!")


