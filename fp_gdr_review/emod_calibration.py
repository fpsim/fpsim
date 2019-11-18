import pylab as pl
import sciris as sc
import pyemod as em
from emod_api.campaign import GenerateCampaignRCM as gencam

sc.tic()

exp_name          = 'Family planning GDR review test runs'
config_file       = 'inputs/config.json'
demographics_file = 'inputs/demographics.json'
overlay_file      = 'inputs/IP_Overlay.json'

algorithm = 'asd'

sc.heading('Loading data...')
data = sc.loadobj('senegal_data.obj')

sc.heading('Setting up simulations...')


# Commonly modified calibration variables and configuration
n_replicates = 6  # replicates, 1 is highly recommended.

base_sim = em.Simulation(config=config_file, demographics=demographics_file)
base_sim.demographics.update(pars=overlay_file) # TODO: make this simpler, or avoid it altogether

#def make_campaign(use_contraception=True):
#    campaign = em.make_campaign()
#    if use_contraception:
#        con_list = gencam.CreateContraceptives()
#        rc_list = gencam.CreateRandomChoiceMatrixList()
#        campaign.pars = gencam.GenerateCampaignFP(con_list, rc_list)
#    return campaign
#
#sims = []
#for use_contraception in [False, True]:
#    for replicate in range(n_replicates):
#        sim = base_sim.copy()
#        sim.campaign = make_campaign(use_contraception)
#        sims.append(sim)

parameters = [{'name': 'fertility_scale_factor',
              'best': 2.0e-6,
              'min': 0.5e-6,
              'max': 10.0e-6}]

def run_fn(x, sim):
    print('hi!')
    print(sim)
    sim.demographics['Defaults']['IndividualAttributes']['FertilityDistribution']['ResultScaleFactor'] = x[0]
    results = sim.run(verbose=True)
    error = pl.rand() # objective_fn(sim)
    return error    

sim = base_sim.copy()
calib = em.Calibration(sim=sim, parameters=parameters, run_fn=run_fn, algorithm=algorithm)
results = calib.run(args={'sim':calib.sim})

sc.toc()
print('Done.')
