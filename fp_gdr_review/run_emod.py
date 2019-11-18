import pyemod as em
from emod_api.campaign import GenerateCampaignRCM as gencam

exp_name = 'Family planning GDR review test runs'
config_file       = 'inputs/config.json'
demographics_file = 'inputs/demographics.json'
overlay_file      = 'inputs/IP_Overlay.json'

# Commonly modified calibration variables and configuration
BASE_POPULATION_SCALE_FACTOR = 0.0033  # For quick test simulations, this is set to a very low value
n_replicates = 1  # replicates, 1 is highly recommended.

base_sim = em.Simulation(config=config_file, demographics=demographics_file)
base_sim.config['parameters']['Base_Population_Scale_Factor'] = BASE_POPULATION_SCALE_FACTOR
base_sim.demographics.update(pars=overlay_file) # TODO: make this simpler, or avoid it altogether

def make_campaign():
    campaign = em.make_campaign()
    con_list = gencam.CreateContraceptives()
    rc_list = gencam.CreateRandomChoiceMatrixList()
    campaign.pars = gencam.GenerateCampaignFP(con_list, rc_list)
    return campaign

sims = []
for replicate in range(n_replicates):
    sim = base_sim.copy()
    sim.campaign = make_campaign()
    sims.append(sim)

exp = em.Experiment(sims=sims)
results = exp.run(how='serial')
exp.plot()
