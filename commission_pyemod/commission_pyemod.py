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
overlay_file = os.path.join(inputs, 'IP_Overlay.json') # TODO: find a better way of doing this

# Commonly modified calibration variables and configuration
BASE_POPULATION_SCALE_FACTOR = 0.0033  # For quick test simulations, this is set to a very low value
n_samples = 3  # the number of distinct parameter sets to run per iteration
n_replicates = 1  # replicates, 1 is highly recommended.
samples = np.linspace(0, 1, n_samples)

base_sim = em.Simulation(config=config_file, demographics=demographics_file)
base_sim.config['parameters']['Base_Population_Scale_Factor'] = BASE_POPULATION_SCALE_FACTOR
base_sim.demographics.update(pars=overlay_file) # TODO: make this simpler, or avoid it altogether

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
for replicate in range(n_replicates):
    for value in samples:
        sim = base_sim.copy()
        sim.campaign = make_campaign(pill_efficacy=value)
        sims.append(sim)

exp = em.Experiment(sims=sims)
results = exp.run(how='serial')
exp.plot()
