import pylab as pl
import sciris as sc
from . import model as mo

__all__ = ['demo', 'Multisim']


def demo(doplot=True, dosave=False):
    '''
    Demonstration usage of VoISim:
        vs.demo()
    '''
    sim = mo.Sim()
    T = sc.tic()
    sim.run()
    sc.toc(T)
    if doplot:
        sim.plot(dosave=dosave)
    return sim


def run_sim(sim, verbose=None):
    ''' Helper function to make parallelization easier -- not to be called directly '''
    sim.run(verbose=verbose)
    sim.calc_costs()
    return sim


class Multisim(sc.prettyobj):
    '''
    Handle multiple simulations -- for example, running with and without the microarray
    tracker, or for different regions.
    '''

    def __init__(self, nsims=None, parsets=None):
        if nsims is None and parsets is not None:
            nsims = len(parsets)
        self.nsims = nsims
        self.sims = sc.odict()
        self.results = sc.odict()
        self.update_parsets(parsets)
        return
    
    def update_parsets(self, parsets=None):
        assert self.nsims == len(parsets), f'Number of sims must match length of parsets ({self.nsims} vs. {len(parsets)})'
        for i in range(self.nsims):
            if parsets is not None:
                pars = parsets[i]
                label = parsets.keys()[i]
            else:
                pars = None
                label = f'sim_{i}'
            sim = mo.Sim(pars=pars)
            self.sims[label] = sim
        return


    def run(self, **kwargs):
        ''' Run all simulations -- keywords are passed to sc.parallelize() '''
        sims = sc.parallelize(run_sim, self.sims, **kwargs)
        for s,sim in enumerate(sims):
            self.sims[s] = sim # Replace with version with results
        for key,sim in self.sims.items():
            self.results[key] = self.sims[key].results
        return self.results


    def plot(self, dosave=None, figargs=None, plotargs=None, axisargs=None, as_years=True, useSI=True):
        ''' Copied from model.py '''
        if figargs  is None: figargs  = {'figsize':(18,18)}
        if plotargs is None: plotargs = {'lw':2, 'alpha':0.7, 'marker':'o'}
        if axisargs is None: axisargs = {'left':0.1, 'bottom':0.05, 'right':0.9, 'top':0.97, 'wspace':0.2, 'hspace':0.8}

        # Copied from Sim.plot()
        if as_years:
            factor = 1.0/12
            timelabel = 'Year'
        else:
            factor = 1.0
            timelabel = 'Month'

        fig = pl.figure(**figargs)
        pl.subplots_adjust(**axisargs)

        n_results = len(self.results)

        results_labels = sc.odict({
                          '0dose':         'Number of unvaccinated children',
                          'new_vaccines_3+dose': 'Number of wasted vaccine doses',
                          'new_cases':           'Number of new measles infections',
                          'vaccines_cost': 'Cost of vaccines (US$)',
                          'trackers_cost': 'Cost of trackers (US$, delivery + reading)',
                          'total_cost':    'Total cost (US$, vaccines + trackers)',
                          })

        plotcount = 0
        n_labels = len(results_labels)
        for res_key in results_labels.keys():

            # Unvaccinated children
            plotcount += 1
            pl.subplot(n_labels,2,plotcount)
            for key,res in self.results.items():
                pl.plot(factor*res['t'], res[res_key], label=self.sims[key].pars['name'], **plotargs)
            mo.fixaxis(useSI=useSI)
            pl.xlabel(timelabel)
            pl.title(results_labels[res_key])
            print('legend1')
            pl.legend()

            # Summary statistics
            plotcount += 1
            pl.subplot(n_labels,2,plotcount)
            x = pl.arange(n_results)+0.5 # WARNING, center properly!
            y = pl.zeros(n_results)
            for r,res in self.results.enumvals():
                y[r] = res[res_key].sum()
                pl.bar(x[r], y[r], label=None)
            mo.fixaxis(useSI=useSI)
            pl.gca().set_xticks(x)
            pl.gca().set_xticklabels([self.sims[key].pars['name'] for key in self.results.keys()])
            pl.title(results_labels[res_key])

        return fig