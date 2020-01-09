import os
import numpy as np
import pyemod as em
from emod_api.campaign import GenerateCampaignRCM as gencam

exp_name = 'Family Planning Workflow Development from PyEMOD'
inputs = 'inputs'
config_file = os.path.join(inputs, 'config.json') # TODO: Remove boilerplate
demographics_file = os.path.join(inputs, 'demographics.json')
overlay_file = os.path.join(inputs, 'IP_Overlay.json') # TODO: find a better way of doing this

# Commonly modified calibration variables and configuration
BASE_POPULATION_SCALE_FACTOR = 0.0033  # For quick test simulations, this is set to a very low value
n_samples = 3  # the number of distinct parameter sets to run per iteration
n_replicates = 1  # replicates, 1 is highly recommended.
samples = [0.575] #np.linspace(0, 1, n_samples)

base_sim = em.Simulation(config=config_file, demographics=demographics_file)

burn_in_years = 50
static_params = {
    "Base_Year": 2011 - burn_in_years,
    'Base_Population_Scale_Factor': BASE_POPULATION_SCALE_FACTOR,
    'Simulation_Duration': (5+burn_in_years)*365,
    'Report_Event_Recorder': 0,
}

base_sim.config['parameters'].update(static_params)
base_sim.demographics.update(pars=overlay_file) # TODO: make this simpler, or avoid it altogether

# TODO: This should mostly be emod_api
def make_campaign(pill_efficacy):
    campaign = em.make_campaign()
    con_list = gencam.CreateContraceptives()
    pill_contraceptive = next(x[1] for x in con_list if x[0] == gencam.USE_PILL)
    pill_contraceptive.Waning_Config.Initial_Effect = pill_efficacy
    rc_list = gencam.CreateRandomChoiceMatrixList()
    campaign_pars = gencam.GenerateCampaignFP(con_list, rc_list)
    campaign.pars = campaign_pars
    return campaign

sims = []
for replicate in range(n_replicates):
    for value in samples:
        sim = base_sim.copy()
        sim.campaign = make_campaign(pill_efficacy=value)
        sims.append(sim)

exp = em.Experiment(sims=sims)
emod_path = os.path.abspath(os.path.join('..', 'idmtools', 'bin', 'Eradication_FP-Ongoing-ReportFPByAgeAndParity_8a43a9fb7b6db784aa00a3f4a7d0972cc4ae493a.exe'))
em.configure(emod_path = emod_path)
results = exp.run(how='COMPS', emod_path = emod_path)
exp.plot()
