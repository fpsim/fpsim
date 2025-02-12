import sciris as sc
import fpsim as fp
from fpsim import utils as fpu
from fpsim import people as fpppl
from fpsim import methods as fpm

import numpy as np
import pytest
import starsim as ss

from fpsim.locations.senegal.senegal import sexual_activity


def make_sim_24f(**kwargs):
    custom_pars = {
        'start_year': 2000,
        'end_year': 2001,
    }
    sim = fp.Sim(location='test', pars=custom_pars, **kwargs)

    return sim

def custom_init(sim, force=False, age=None, sex=None, empowerment_module=None, education_module=None, person_defaults=None):
    if force or not sim.initialized:
        sim.ti = 0  # The current time index
        fpu.set_seed(sim['seed'])
        sim.init_results()
        sim.people=fpppl.People(pars=sim.pars, age=age, sex=sex, empowerment_module=empowerment_module, education_module=education_module, person_defaults=person_defaults )  # This step also initializes the empowerment and education modules if provided
        sim.init_contraception()  # Initialize contraceptive methods
        sim.initialized = True
    return sim

def test_pregnant_woman():
    # start out 18f, no contra, no pregnancy, active, fertile, has_intent
    methods = ss.ndict([
        fpm.Method(name='none', efficacy=0, modern=False, dur_use=fpm.ln(2, 3), label='None'),
        fpm.Method(name='test',     efficacy=0.0, modern=True,  dur_use=fpm.ln(2, 3), label='Test'),
    ])
    contra_mod = fpm.RandomChoice(methods=methods)

    # force all to have debuted
    debut_age = {}
    debut_age['ages']=np.arange(10, 45, dtype=float)
    debut_age['probs']=np.zeros(35, dtype=float)
    debut_age['probs'][10:20] = 1.0

    sexual_activity = np.zeros(51, dtype=float)
    sexual_activity[20:30] = 1.0

    sim = make_sim_24f(n_agents=1000, primary_infertility=0, contraception_module=contra_mod, sexual_activity=sexual_activity, debut_age=debut_age)
    sim = custom_init(sim, age=24, sex=0, person_defaults={'fertility_intent':True})

    # force all women to have the same fecundity and be sexually active
    # Note: not all agents will be active at t==0 but will be after t==1
    sim.people.personal_fecundity[:] = 1

    #sim.people.fated_debut[:]=0

    sim.run()

    n_agents = sim.pars['n_agents']

    # all women are 24 and can become pregnant, so we can look at percentages to estimate the
    # expected number of pregnancies, stillborns, living infants, and miscarriages/abortions
    # pregnancy rate: 11.2% per month or approx 80% per year
    assert 0.85 * n_agents > sim.results.pregnancies[0:12].sum() > 0.75 * n_agents, "Expected number of pregnancies not met"

    # stillbirth rate: 2.5% of all pregnancies, but only count completed pregnancies
    assert 0.06 * sim.results.pregnancies[0:3].sum() > sim.results.stillbirths[9:12].sum() > 0.01 * sim.results.pregnancies[0:3].sum() , "Expected number of stillbirths not met"

    # miscarriage rate: 10% of all pregnancies. Miscarriages are calculated at end of 1st trimester, so pregnancies from months 0-9
    # have miscarriages from months 3-12.
    pregnancies = sim.results.pregnancies[0:9].sum()
    miscarriages = sim.results.miscarriages[3:12].sum()
    assert 0.15 * pregnancies > miscarriages > 0.05 * pregnancies, "Expected number of miscarriages not met"

    # abortion rate: 8% of all pregnancies
    # abortions occur at same timestep as conception, so all pregnancies were checked for abortions
    pregnancies = sim.results.pregnancies.sum()
    abortions = sim.results.abortions.sum()
    assert 0.13 * pregnancies > abortions > 0.03 * pregnancies, "Expected number of abortions not met"

    # live birth rate: 79.5% of all pregnancies
    # no premature births so all births are full term
    pregnancies = sim.results.pregnancies[0:3].sum()
    live_births = sim.results.births[9:12].sum()
    assert 0.85 * pregnancies > live_births > 0.75 * pregnancies, "Expected number of live births not met"




