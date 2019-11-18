import pyemod as em
from emod_api.campaign import GenerateCampaignRCM as gencam

exp_name = 'Family planning GDR review test runs'
config_file       = 'inputs/config.json'
demographics_file = 'inputs/demographics.json'
overlay_file      = 'inputs/IP_Overlay.json'

# Commonly modified calibration variables and configuration
n_replicates = 1  # replicates, 1 is highly recommended.

base_sim = em.Simulation(config=config_file, demographics=demographics_file)
base_sim.demographics.update(pars=overlay_file) # TODO: make this simpler, or avoid it altogether
base_sim.demographics['Defaults']['IndividualAttributes']['FertilityDistribution']['ResultScaleFactor'] = 1e-5

def make_campaign(use_contraception=True):
    campaign = em.make_campaign()
    if use_contraception:
        con_list = gencam.CreateContraceptives()
        rc_list = gencam.CreateRandomChoiceMatrixList()
        campaign.pars = gencam.GenerateCampaignFP(con_list, rc_list)
    return campaign

sims = []
count = 0
for use_contraception in [True, True]:
    for replicate in range(n_replicates):
        count += 1
        sim = base_sim.copy()
        sim.config['parameters']['Run_Number'] = count
        sim.campaign = make_campaign(use_contraception)
        sims.append(sim)

exp = em.Experiment(sims=sims)
results = exp.run(how='parallel')
exp.plot()
