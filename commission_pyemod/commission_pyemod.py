import os
import numpy as np
import pyemod as em

# TODO: This should be emod_api
import json
import GenerateCampaignRCM as gencam

exp_name = 'Family Planning Workflow Development'
inputs = 'inputs'
config_file = os.path.join(inputs, 'config.json') # TODO: Remove boilerplate
demographics_file = os.path.join(inputs, 'demographics.json')

# Commonly modified calibration variables and configuration
BASE_POPULATION_SCALE_FACTOR = 0.0033  # For quick test simulations, this is set to a very low value
N_SAMPLES = 3  # the number of distinct parameter sets to run per iteration
N_REPLICATES = 1  # replicates, 1 is highly recommended.

samples = np.linspace(0, 1, N_SAMPLES)

base_sim= em.Simulation(config=config_file, demographics=demographics_file)

# TODO: This should mostly be emod_api
def make_campaign(pill_efficacy):
    campaign = em.make_campaign()
    con_list = gencam.CreateContraceptives()
    pill_contraceptive = next(x[1] for x in con_list if x[0] == gencam.USE_PILL)
    pill_contraceptive.Waning_Config.Initial_Effect = pill_efficacy
    rc_list = gencam.CreateRandomChoiceMatrixList()
    campaign_pars = gencam.GenerateCampaignFP(con_list, rc_list)
    campaign.pars = json.loads(campaign_pars.to_json())
    return campaign

sims = []
for value in samples:
    sim = base_sim.copy()
    sim.campaign = make_campaign(pill_efficacy=value)
    sims.append(sim)

exp = em.Experiment(sims=sims)
results = exp.run()
exp.plot()
