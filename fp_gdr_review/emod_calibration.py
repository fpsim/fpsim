import sciris as sc
import pyemod as em
from emod_api.campaign import GenerateCampaignRCM as gencam
import senegal_data as sd

sc.tic()

exp_name          = 'Family planning GDR review test runs'
config_file       = 'inputs/config.json'
demographics_file = 'inputs/demographics.json'
overlay_file      = 'inputs/IP_Overlay.json'

# Commonly modified calibration variables and configuration
n_replicates = 1  # replicates, 1 is highly recommended.

base_sim = em.Simulation(config=config_file, demographics=demographics_file)
base_sim.demographics.update(pars=overlay_file) # TODO: make this simpler, or avoid it altogether

def make_campaign(use_contraception=True):
    campaign = em.make_campaign()
    if use_contraception:
        con_list = gencam.CreateContraceptives()
        rc_list = gencam.CreateRandomChoiceMatrixList()
        campaign.pars = gencam.GenerateCampaignFP(con_list, rc_list)
    return campaign

#sims = []
#for use_contraception in [False, True]:
#    for replicate in range(n_replicates):
#        sim = base_sim.copy()
#        sim.campaign = make_campaign(use_contraception)
#        sims.append(sim)
#
#exp = em.Experiment(sims=sims)
#results = exp.run(how='parallel')
#exp.plot()


sc.heading('Loading data...')
data = sc.loadobj('senegal_data.obj')

sc.toc()
print('Done.')
