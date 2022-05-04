'''
Class to define and run scenarios
'''

import numpy as np
import pandas as pd
import sciris as sc
import fpsim as fp
import fp_analyses as fa
from . import utils as fpu
from . import interentions as fpi

__all__ = ['Scenarios']


# Define basic things here
default_pars = fa.senegal_parameters.make_pars()
method_names = default_pars['methods']['names']



class Scenarios(sc.prettyobj):


    def make_sim(self, label='<no label>', **kwargs):
        ''' Create a single sim, with a label and updated parameters '''
        pars = fa.senegal_parameters.make_pars()
        pars.update(kwargs)
        sim = fp.Sim(pars=pars, label=label)
        return sim


    def make_sims(self, repeats=5, **kwargs):
        ''' Create a list of sims that are all identical except for the random seed '''
        sims = sc.autolist()
        for i in range(repeats):
            kwargs.setdefault('seed', 0)
            kwargs['seed'] += i
            sims += self.make_sim(**kwargs)
        return sims


    def run_sims(self, *args):
        ''' Actually run a list of sims '''
        msims = sc.autolist()
        for sims in args:
            msims += fp.MultiSim(sims)
        msim = fp.MultiSim.merge(*msims)
        msim.run()
        return msim


    def analyze_sims(self, msim, start_year=2010, end_year=2020):
        ''' Take a list of sims that have different labels and count the births in each '''

        def count_births(sim):
            year = sim.results['t']
            births = sim.results['births']
            inds = sc.findinds((year >= start_year), year < end_year)
            output = births[inds].sum()
            return output

        def method_failure(sim):
            year = sim.results['tfr_years']
            meth_fail = sim.results['method_failures_over_year']
            inds = sc.findinds((year >= start_year), year < end_year)
            output = meth_fail[inds].sum()
            return output

        def count_pop(sim):
            year = sim.results['tfr_years']
            popsize = sim.results['pop_size']
            inds = sc.findinds((year >= start_year), year < end_year)
            output = popsize[inds].sum()
            return output

        def mean_tfr(sim):
            year = sim.results['tfr_years']
            rates = sim.results['tfr_rates']
            inds = sc.findinds((year >= start_year), year < end_year)
            output = rates[inds].mean()
            return output

        # Split the sims up by scenario
        results = sc.objdict()
        results.sims = sc.objdict(defaultdict=sc.autolist)
        for sim in msim.sims:
            results.sims[sim.label] += sim

        # Count the births across the scenarios
        raw = sc.objdict(defaultdict=sc.autolist)
        for key,sims in results.sims.items():
            for sim in sims:
                n_births = count_births(sim)
                n_fails  = method_failure(sim)
                n_pop = count_pop(sim)
                n_tfr = mean_tfr(sim)
                raw.scenario += key      # Append scenario key
                raw.births   += n_births # Append births
                raw.fails    += n_fails  # Append failures
                raw.popsize  += n_pop    # Append population size
                raw.tfr      += n_tfr    # Append mean tfr rates

        # Calculate basic stats
        results.stats = sc.objdict()
        for statkey in ['mean', 'median', 'std', 'min', 'max']:
            results.stats[statkey] = sc.objdict()
            for k,vals in raw.items():
                if k != 'scenario':
                    results.stats[statkey][k] = getattr(np, statkey)(vals)

        # Also save as pandas
        results.df = pd.DataFrame(raw)

        return results


def key2ind(sim, key):
    """
    Take a method key and convert to an int, e.g. 'Condoms' → 7
    """
    ind = key
    if ind in [None, 'all']:
        ind = slice(None) # This is equivalent to ":" in matrix[:,:]
    elif isinstance(ind, str):
        ind = sim.pars['methods']['map'][key]
    return ind


def getval(v):
    ''' Handle different ways of supplying a value -- number, distribution, function '''
    if sc.isnumber(v):
        return v
    elif isinstance(v, dict):
        return fpu.sample(**v)[0]
    elif callable(v):
        return v()


class update_methods(fpi.Intervention):
    """
    Intervention to modify method efficacy and/or switching matrix.

    Attributes:
        self.year::float: The year we want to change the method.
        self.scen::dict: Has the following keys:

            probs::str
                An optional key with the value of a list of dictionaries where each dictionary has
                the following keys:

                source::str
                    The source method to be changed.
                dest::str
                    The destination method to be changed.
                factor::float
                    The factor by which to multiply existing probability.
                value::float
                    The value to replace the switching probability value.
                keys::list
                    A list of strings representing age groups to affect.

            eff::str
                An optional key for changing efficacy; its value is a dictionary with the following schema:

                    {method: efficacy}
                        Where method is the method to be changed, and efficacy is the new efficacy (can include multiple keys).

        self.matrix::str: One of ['probs_matrix', 'probs_matrix_1', 'probs_matrix_1-6'] where:

            probs_matrix:
                Changes the specified uptake at the corresponding year regardless of state.
            probs_matrix_1
                Changes the specified uptake for all individuals in their first month postpartum.
            probs_matrix_1-6
                Changes the specified uptake for all individuals that are in the first 6 months postpartum.
    """

    def __init__(self, year, scen, matrix='probs_matrix'):
        """
        Initializes self.year/scen/matrix from parameters
        """
        super().__init__()
        self.year   = year
        self.scen   = scen
        self.matrix = matrix
        valid_matrices = ['probs_matrix', 'probs_matrix_1', 'probs_matrix_1-6'] # TODO: be less subtle about the difference between normal and postpartum matrices
        if matrix not in valid_matrices:
            raise sc.KeyNotFoundError(f'Matrix must be one of {valid_matrices}, not "{matrix}"')
        return


    def apply(self, sim, verbose=True):
        """
        Applies the efficacy or contraceptive uptake changes if it is the specified year
        based on scenario specifications.
        """

        if sim.y >= self.year and not(hasattr(sim, 'modified')):
            sim.modified = True

            # Implement efficacy
            if 'eff' in self.scen:
                for k,rawval in self.scen.eff.items():
                    v = getval(rawval)
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
                        keys = sim.pars['methods']['probs_matrix'].keys()

                    for k in keys:
                        if self.matrix == 'probs_matrix':
                            matrices = sim.pars['methods']
                        else:
                            matrices = sim.pars['methods_postpartum']
                        matrix = matrices[self.matrix][k]
                        orig = matrix[source, dest]
                        if factor is not None:
                            matrix[source, dest] *= getval(factor)
                        elif value is not None:
                            matrix[source, dest] = getval(value)
                        if verbose:
                            print(f'At time {sim.y:0.1f}, matrix for age group {k} was changed from:\n{orig}\nto\n{matrix[source, dest]}')


        return