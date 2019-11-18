import pylab as pl
import sciris as sc
import pyemod as em
#from emod_api.campaign import GenerateCampaignRCM as gencam

sc.tic()

exp_name          = 'Family planning GDR review test runs'
config_file       = 'inputs/config.json'
demographics_file = 'inputs/demographics.json'
overlay_file      = 'inputs/IP_Overlay.json'

algorithm = 'asd'
do_calibrate = False
do_plot = True
save_fig = True

sc.heading('Loading data...')
data = sc.loadobj('senegal_data.obj')

sc.heading('Setting up simulations...')


# Commonly modified calibration variables and configuration
n_replicates = 24  # replicates, 1 is highly recommended.

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

def run_fn(x, sim, full_output=False):
    base_sim = sim.copy()
    base_sim.demographics['Defaults']['IndividualAttributes']['FertilityDistribution']['ResultScaleFactor'] = x[0]
    
    sims = []
    count = 0
    for replicate in range(n_replicates):
        count += 1
        S = base_sim.copy()
        S.config['parameters']['Run_Number'] = count
#            S.campaign = make_campaign(use_contraception)
        sims.append(S)
    
    exp = em.Experiment(sims=sims)
    results = exp.run(how='parallel', verbose=False)
    
    error = error_fn(results.results, full_output=full_output) # WARNING, refactor
    return error

def error_fn(results, full_output=False):
    # WARNING, find a better solution!
    nresults = len(results)
    npts = len(results[0].data['Channels']['Statistical Population']['Data'])
    popsize = pl.zeros((nresults, npts))
    for r,result in enumerate(results):
        popsize[r,:] = result.data['Channels']['Statistical Population']['Data']
    
    simpopvec = popsize.mean(axis=0)
    datapopvec = data.unpop.popsize
    
    newnpts   = 101
    nsimpts   = len(simpopvec)
    ndatapts  = len(datapopvec)
    origsimx  = pl.linspace(0, 1, nsimpts)
    origdatax = pl.linspace(0, 1, ndatapts)
    newx      = pl.linspace(0, 1, newnpts)
    sim_y     = sc.smoothinterp(newx=newx, origx=origsimx, origy=simpopvec)
    data_y    = sc.smoothinterp(newx=newx, origx=origdatax, origy=datapopvec)
    
    mismatch = ((data_y - sim_y)**2).sum()
    if full_output:
        sim_full = pl.zeros((nresults, newnpts))
        for r,result in enumerate(results):
            sim_full[r,:] = sc.smoothinterp(newx=newx, origx=origsimx, origy=popsize[r,:])
        return sc.objdict({'x':newx, 'data_y':data_y, 'sim_y':sim_y, 'sim_full':sim_full})
    else:
        return mismatch
    
    
if do_calibrate:
    sim = base_sim.copy()
    calib = em.Calibration(sim=sim, parameters=parameters, run_fn=run_fn, algorithm=algorithm)
    results = calib.run(args={'sim':calib.sim})

if do_plot:
    sc.heading('Plotting...')
    
    origval = 2e-6 # From running the calibration and viewing the results
    calibval = 3.07929688e-06
    
    sim = base_sim.copy()
    orig = run_fn([origval], sim, full_output=True)
    calib = run_fn([calibval], sim, full_output=True)
    xax = orig.x*40 + 1980
    
    yscale = 5.583e-3 # From senegal_data.py
    
    fig = pl.figure(figsize=(20,8))
    orig_col  = [0.6, 0.6, 0.6]
    calib_col = [1.0, 0.6, 0.0]
    data_col  = [0.0, 0.0, 0.5]
    for r in range(n_replicates):
        pl.plot(xax, yscale*orig.sim_full[r,:], c=orig_col, lw=2, alpha=0.2)
        pl.plot(xax, yscale*calib.sim_full[r,:], c=calib_col, lw=2, alpha=0.2)
    pl.scatter(xax, yscale*calib.data_y, c=data_col, s=100, alpha=0.8, zorder=100)
    pl.plot(xax, yscale*calib.sim_y, c=calib_col, lw=4, alpha=1.0, zorder=150)
    pl.plot(xax, yscale*orig.sim_y, c=orig_col, lw=4, alpha=1.0, zorder=200)
    
    sc.setylim()
    pl.xlim([1979,2021])
    sc.boxoff()
    pl.title('Calibration of Senegal demographics', fontweight='bold')
    pl.xlabel('Year', fontweight='bold')
    pl.ylabel('Population (millions)', fontweight='bold')
    if save_fig:
        fig.savefig('figs/senegal_popsize_calibration.png', bbox_inches='tight', dpi=150)

sc.toc()
print('Done.')
