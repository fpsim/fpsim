'''
Run efficacy scenarios for the GR
'''

import numpy as np
import pandas as pd
import sciris as sc
import fpsim as fp
import fp_analyses as fa

default_pars = fa.senegal_parameters.make_pars()
method_names = default_pars['methods']['names']
age_keys = list(default_pars['methods']['probs_matrix'].keys())

def key2ind(sim, key):
    ind = key
    if ind in [None, 'all']:
        ind = slice(None) # This is equivalent to ":" in matrix[:,:]
    elif isinstance(ind, str):
        ind = sim.pars['methods']['map'][key]
    return ind


class update_methods(fp.Intervention):

    def __init__(self, year, scen):
        super().__init__()
        self.year = year
        self.scen = scen

    def apply(self, sim, verbose=True):

        if sim.y >= self.year and not(hasattr(sim, 'modified')):
            sim.modified = True

            # Implement efficacy
            if 'eff' in self.scen:
                for k,v in self.scen.eff.items():
                    ind = key2ind(sim, k)
                    orig = sim.pars['method_efficacy'][ind]
                    sim.pars['method_efficacy'][ind] = v
                    if verbose:
                        print(f'At time {sim.y:0.1f}, efficacy for method {k} was changed from {orig:0.3f} to {v:0.3f}')

            # Implement method mix shift
            if 'probs' in self.scen:
                for entry in self.scen.probs:
                    source = key2ind(sim, entry['source'])
                    dest   = key2ind(sim, entry['dest'])
                    factor = entry.pop('factor', None)
                    value  = entry.pop('value', None)
                    keys   = entry.pop('keys', None)
                    if keys is None:
                        keys = age_keys

                    for k in keys:
                        matrix = sim.pars['methods']['probs_matrix'][k]
                        orig = matrix[source, dest]
                        if factor is not None:
                            matrix[source, dest] *= factor
                        elif value is not None:
                            matrix[source, dest] = value
                        if verbose:
                            print(f'At time {sim.y:0.1f}, matrix for age group {k} was changed from:\n{orig}\nto\n{matrix[source, dest]}')


        return

def make_sim(label='<no label>', **kwargs):
    pars = fa.senegal_parameters.make_pars()
    pars.update(kwargs)
    sim = fp.Sim(pars=pars, label=label)
    return sim


def make_sims(repeats=5, **kwargs):
    sims = sc.autolist()
    for i in range(repeats):
        kwargs.setdefault('seed', 0)
        kwargs['seed'] += i
        sims += make_sim(**kwargs)
    return sims


def run_sims(sims):
    msim = fp.MultiSim(sims)
    msim.run()
    return msim


def analyze_sims(msim, start_year=2010, end_year=2020):

    def count_births(sim):
        year = sim.results['t']
        births = sim.results['births']
        inds = sc.findinds((year >= start_year), year < end_year)
        output = births[inds].sum()
        return output

    # Split the sims up by scenario
    results = sc.objdict()
    results.sims = sc.objdict(defaultdict=sc.autolist)
    for sim in msim.sims:
        results.sims[sim.label] += sim

    # Count the births across the scenarios
    raw = sc.objdict(scenario=sc.autolist(), births=sc.autolist())
    results.raw = sc.objdict(defaultdict=sc.autolist)
    for key,sims in results.sims.items():
        for sim in sims:
            n_births = count_births(sim)
            results.raw[key] += n_births
            raw.scenario += key
            raw.births += n_births

    # Calculate basic stats
    results.stats = sc.objdict()
    for statkey in ['mean', 'median', 'std', 'min', 'max']:
        results.stats[statkey] = sc.objdict()
        for k,vals in results.raw.items():
            results.stats[statkey][k] = getattr(np, statkey)(vals)

    # Also save as pandas
    results.df = pd.DataFrame(raw)

    return results



if __name__ == '__main__':

    #%% Define sim parameters
    pars = dict(
        n = 10_000,
        start_year = 1980,
        end_year = 2020,
    )

    # Run options
    repeats   = 3
    scen_year = 2005
    debug     = False

    #%% Define scenarios

    # Increased efficacy
    eff_scen = sc.objdict(
        eff={k:1.0 for k in method_names if k != 'None'}
    )
    eff = update_methods(scen_year, eff_scen)

    # Increased uptake
    uptake_scen = sc.objdict(
        eff = {'BTL':0.8},
        probs = [
            dict(
                source = 'None', # Source method, 'all' for all methods
                dest   = 'BTL', # Destination
                factor = None, # Factor by which to multiply existing probability
                value  = 0.2, # Alternatively, specify the absolute probability
                keys   = None, # Which age keys to modify -- if not specified, all
            ),
        ]
    )
    uptake = update_methods(scen_year, uptake_scen)


    #%% Run
    sims = sc.autolist()

    sims += make_sims(repeats=repeats, label='Baseline', **pars)
    sims += make_sims(repeats=repeats, interventions=eff, label='Increased efficacy', **pars)
    sims += make_sims(repeats=repeats, interventions=uptake, label='Increased uptake', **pars)

    if debug:
        sim = sims[4]
        sim.run()
        sim.plot()

    else:
        msim = run_sims(sims)
        msim.plot()

        # Analyze
        results = analyze_sims(msim)
        print(results.df)
        print(results.stats)

