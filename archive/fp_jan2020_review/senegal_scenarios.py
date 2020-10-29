'''
Run Senegal scenarios
'''

import pylab as pl
import sciris as sc
import lemod_fp as lfp
import senegal_parameters as sp

sc.tic()
pl.rcParams['font.size'] = 16

# Set what to run
run_implants = 1
run_health   = 0


if run_implants:
    pars = sp.make_pars()
    pars['end_year'] = 2030
    sim1 = lfp.Sim(pars=pars)
    sim2 = lfp.Sim(pars=pars)
    
    def urhi(sim):
        print('Added URHI')
        sim.pars['methods']['matrix'][0,0] *= 0.80
        # switching = sim.pars['switching']
        # print(switching)
        # for i,method1 in enumerate(switching.keys()):
        #     switching[method1][0] *= 0.0
        #     switching[method1][:] = switching[method1][:]/switching[method1][:].sum()
        # sim.pars['switching'] = switching
        # print(switching)
        # for person in sim.people.values():
        #     person.pars = sim.pars
    
    # def anti_urhi(sim):
    #     print('Added anti-URHI')
    #     sim.pars['methods']['matrix'][0,0] *= 1.5
        
    # for year in [2011]:
    #     sim1.add_intervention(intervention=anti_urhi, year=year)
    
    # for year in [2011, 2015]:
    #     sim1.add_intervention(intervention=urhi, year=year)
    
    for year in [2011, 2015]:
        sim2.add_intervention(intervention=urhi, year=year)
    
    lfp.set_seed(1)
    sim1.run()
    lfp.set_seed(1)
    sim2.run()
    
    fig = pl.figure(figsize=(26,14))
    
    years = sim1.results['t']
    
    pl.subplot(1,2,1)
    pl.plot(years, sim1.results['mcpr']*100, label='Baseline', lw=2)
    pl.plot(years, sim2.results['mcpr']*100, label='Intervention', lw=2)
    pl.xlabel('Year')
    pl.ylabel('Percentage')
    pl.title('MCPR', fontweight='bold')
    sc.setylim()
    pl.xlim([1960, 2030])
    pl.legend()
    
    pl.subplot(1,2,2)
    factor = 16e6/sim1.pars['n']/10/10
    s1d = pl.cumsum(sim1.results['child_deaths'])*factor
    s2d = pl.cumsum(sim2.results['child_deaths'])*factor
    pl.plot(years, s1d, label='Baseline', lw=2)
    pl.plot(years, s2d, label='Intervention', lw=2)
    pl.xlabel('Year')
    pl.ylabel('Count')
    string = sc.sigfig(s1d[-1]-s2d[-1], sigfigs=3, sep=True)
    pl.title(f'Child mortality; deaths averted={string}', fontweight='bold')
    sc.setylim()
    pl.xlim([1960, 2030])
    pl.legend()
    
    pl.savefig('figs/senegal_scenarios.png')
    

sc.toc()
print('Done.')
    
    
    
    