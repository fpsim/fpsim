'''
Class to define and run scenarios
'''

import numpy as np
import pandas as pd
import sciris as sc
from . import defaults as fpd
from . import sim as fps
from . import interventions as fpi

__all__ = ['make_scen', 'Scenario', 'Scenarios']


#%% Validation functions -- for internal use only

def check_not_none(obj, *args):
    ''' Check that all needed inputs are supplied '''
    for key in args:
        if obj[key] is None:
            errormsg = f'Entry "{key}" is not allowed to be None for scenario spec "{obj.which}"'
            raise ValueError(errormsg)
    return


def check_ages(ages):
    ''' Check that age keys are all valid '''
    valid_keys = list(fpd.method_age_map.keys()) + fpd.none_all_keys
    ages = sc.tolist(ages, keepnone=True)
    for age in ages:
        if age not in valid_keys:
            errormsg = f'Age "{age}" is not valid; choices are:\n{sc.newlinejoin(valid_keys)}'
            raise sc.KeyNotFoundError(errormsg)
    return


def check_method(methods):
    ''' Check that methods are valid '''
    valid_methods = list(fpd.method_map.keys()) + [None]
    for method in methods:
        if method not in valid_methods:
            errormsg = f'Method "{method}" is not valid; choices are:\n{sc.newlinejoin(valid_methods)}'
            raise sc.KeyNotFoundError(errormsg)
    return


#%% Scenario classes

class Scenario(sc.prettyobj, sc.dictobj):
    '''
    Store the specification for a single scenario (which may consist of multiple interventions).

    This function is intended to be as flexible as possible; as a result, it may
    be somewhat confusing. There are five different ways to call it -- method efficacy,
    method probability, method initiation/discontinuation, parameter, and custom intervention.

    Args (shared):
        spec   (dict): a pre-made specification of a scenario; see keyword explanations below (optional)
        args   (list): additional specifications (optional)
        label  (str): the sim label to use for this scenario
        year   (float): the year at which to activate efficacy and probability scenarios
        matrix (str): which set of probabilities to modify for probability scenarios (e.g. annual or postpartum)
        ages   (str/list): the age groups to modify the probabilities for

    Args (efficacy):
        year (float): as above
        eff  (dict): a dictionary of method names and new efficacy values

    Args (probablity):
        year   (float): as above
        matrix (str): as above
        ages   (str): as above
        source (str): the method to switch from
        dest   (str): the method to switch to
        factor (float): if supplied, multiply the [source, dest] probability by this amount
        value  (float): if supplied, instead of factor, replace the [source, dest] probability by this value

    Args (initiation/discontinuation):
        year   (float): as above
        matrix (str): as above
        ages   (str): as above
        method (str): the method for initiation/discontinuation
        init_factor    (float): as with "factor" above, for initiation (None → method)
        discont_factor (float): as with "factor" above, for discontinuation (method → None)
        init_value     (float): as with "value" above, for initiation (None → method)
        discont_value  (float): as with "value" above, for discontinuation (method → None)

    Args (parameter):
        par (str): the parameter to modify
        years (float/list): the year(s) at which to apply the modifications
        vals (float/list): the value(s) of the parameter for each year

    Args (custom):
        interventions (Intervention/list): any custom intervention(s) to be applied to the scenario

    Congratulations on making it this far.

    **Examples**::

        # Basic efficacy scenario
        s1 = fp.make_scen(eff={'Injectables':0.99}, year=2020)

        # Double rate of injectables initiation
        s2 = fp.make_scen(source='None', dest='Injectables', factor=2)

        # Double rate of injectables initiation -- alternate approach
        s3 = fp.make_scen(method='Injectables', init_factor=2)

        # More complex example: change condoms to injectables transition probability for 18-25 postpartum women
        s4 = fp.make_scen(source='Condoms', dest='Injectables', value=0.5, ages='18-25', matrix='pp1to6')

        # Parameter scenario: halve exposure
        s5 = fp.make_scen(par='exposure_factor', years=2010, vals=0.5)

        # Custom scenario
        def update_sim(sim): sim.updated = True
        s6 = fp.make_scen(interventions=update_sim)

        # Combining multiple scenarios: increase injectables initiation and reduce exposure factor
        s7 = fp.make_scen(
            dict(method='Injectables', init_factor=2),
            dict(par='exposure_factor', years=2010, vals=0.5)
        )

        # Scenarios can be combined
        s8 = s1 + s2
    '''
    def __init__(self, spec=None, *args, label=None, year=None, matrix=None, ages=None, # Basic settings
                 eff=None, probs=None, # Option 1
                 source=None, dest=None, factor=None, value=None, # Option 2
                 method=None, init_factor=None, discont_factor=None, init_value=None, discont_value=None, # Option 3
                 par=None, years=None, vals=None, # Option 4
                 interventions=None, # Option 5
                 ):

        # Handle input specification
        self.specs = sc.mergelists(*[Scenario(**spec).specs for spec in sc.mergelists(spec, *args)]) # Sorry
        self.label = label

        # Handle other keyword inputs
        eff_spec   = None
        prob_spec  = None
        par_spec   = None
        intv_specs = None

        # It's an efficacy scenario
        if eff is not None:
            eff_spec = sc.objdict(
                which  = 'eff',
                eff    = eff,
                year   = year,
            )
            check_not_none(eff_spec, 'year')

        # It's a method switching probability scenario
        prob_args = [probs, factor, value, init_factor, discont_factor, init_value, discont_value]
        if len(sc.mergelists(*prob_args)): # Check if any are non-None
            prob_spec = sc.objdict(
                which  = 'prob',
                year   = year,
                probs = probs if probs else dict(
                    matrix         = matrix,
                    ages           = ages,
                    source         = source,
                    dest           = dest,
                    factor         = factor,
                    value          = value,
                    method         = method,
                    init_factor    = init_factor,
                    discont_factor = discont_factor,
                    init_value     = init_value,
                    discont_value  = discont_value,
                )
            )
            check_not_none(prob_spec, 'year')
            check_ages(ages)
            check_method([source, dest, method])

        # It's a parameter change scenario
        if par is not None:
            par_spec = sc.objdict(
                which = 'par',
                par   = par,
                years = years,
                vals  = vals,
            )
            check_not_none(par_spec, 'years', 'vals')

        # It's a custom scenario(s)
        if interventions is not None:
            intv_specs = []
            for intv in sc.tolist(interventions):
                intv_specs.append(sc.objdict(
                    which        = 'intv',
                    intervention = intv,
                ))

        # Merge these different scenarios into the list, skipping None entries
        self.specs.extend(sc.mergelists(eff_spec, prob_spec, par_spec, intv_specs))

        # Finally, ensure all have a consistent label if supplied
        self.update_label()

        return


    def __add__(self, scen2):
        ''' Combine two scenario spec lists '''
        scen3 = sc.dcp(self)
        scen3.specs.extend(sc.dcp(scen2.specs))
        return scen3


    def __radd__(self, scen2):
        ''' Allows sum() to work correctly '''
        if not scen2: return self
        else:         return self.__add__(scen2)


    def update_label(self, label=None):
        ''' Ensure all specs have the correct label '''
        if label is None:
            label = self.label
        if label is not None:
            for spec in self.specs:
                spec['label'] = label
        self.label = label
        return



def make_scen(*args, **kwargs):
    ''' Alias for ``fp.Scenario()``. '''
    return Scenario(*args, **kwargs)

# Ensure the function ahs the same docstring as the class
make_scen.__doc__ +=  '\n\n' + Scenario.__doc__



class Scenarios(sc.prettyobj):
    '''
    Run different intervention scenarios.

    A "scenario" can be thought of as a list of sims, all with the same parameters
    except for the random seed. Usually, scenarios differ from each other only
    in terms of the interventions run (to compare other differences between sims,
    it's preferable to use a MultiSim object).

    Args:
        pars    (dict): parameters to pass to the sim
        repeats (int):  how many repeats of each scenario to run (default: 1)
        scens   (list): the list of scenarios to run; see also ``fp.make_scen()`` and ``Scenarios.add_scen()``
        kwargs  (dict): optional additional parameters to pass to the sim

    **Example**::

        scen1 = fp.make_scen(label='Baseline')
        scen2 = fp.make_scen(year=2002, eff={'Injectables':0.99}) # Basic efficacy scenario
        scens = fp.Scenarios(location='test', repeats=2, scens=[scen1, scen2])
        scens.run()
    '''

    def __init__(self, pars=None, repeats=None, scens=None, **kwargs):
        self.pars = sc.mergedicts(pars, kwargs)
        self.repeats = repeats if repeats is not None else 1
        self.scens = sc.dcp(sc.tolist(scens))
        self.simslist = []
        self.msims = []
        self.msim = None
        self.already_run = False
        return


    def add_scen(self, scen=None, label=None):
        ''' Add a scenario or scenarios to the Scenarios object '''
        if isinstance(scen, Scenario):
            scen.update_label(label)
            self.scens.append(scen)
        else:
            self.scens.append(Scenario(scen, label=label))
        return


    def make_sims(self, scenlabel, **kwargs):
        ''' Create a list of sims that are all identical except for the random seed '''
        if scenlabel is None:
            errormsg = 'Scenario label must be defined'
            raise ValueError(errormsg)
        sims = sc.autolist()
        for i in range(self.repeats):
            pars = sc.mergedicts(fpd.pars(self.pars.get('location')), self.pars, _copy=True)
            pars.update(kwargs)
            pars['seed'] += i
            sim = fps.Sim(pars=pars)
            sim.scenlabel = scenlabel # Special label for scenarios objects
            if sim.label is None:
                sim.label = scenlabel # Include here if no other label
            sims += sim
        return sims


    def make_scens(self):
        ''' Convert a scenario specification into a list of sims '''
        for i,scen in enumerate(self.scens):
            simlabel = None
            interventions = sc.autolist()
            for spec in sc.tolist(scen.specs):

                # Figure ou what type of spec this is
                spec = sc.dcp(spec)
                if 'which' not in spec:
                    errormsg = f'Invalid scenario spec: key "which" not found among: {sc.strjoin(spec.keys())}'
                    raise ValueError(errormsg)
                else:
                    which = spec.pop('which')

                # Handle update_methods
                if which == 'eff':
                    eff = spec.pop('eff')
                    year = spec.pop('year')
                    interventions += fpi.update_methods(eff=eff, year=year)
                elif which == 'prob':
                    probs = spec.pop('probs')
                    year = spec.pop('year')
                    interventions += fpi.update_methods(probs=probs, year=year)
                elif which == 'par':
                    par = spec.pop('par')
                    years = spec.pop('years')
                    vals = spec.pop('vals')
                    interventions += fpi.change_par(par=par, years=years, vals=vals)
                elif which == 'intv':
                    intv = spec.pop('intervention')
                    interventions += intv

                # Handle label
                label  = spec.pop('label', None)
                assert len(spec)==0, f'Unrecognized scenario key(s) {sc.strjoin(spec.keys())}'
                if simlabel is None:
                    simlabel = label
                else:
                    if label != simlabel:
                        print('Warning, new sim label {label} does not match existing sim label {simlabel}')

            if simlabel is None:
                simlabel = f'Scenario {i}'
            sims = self.make_sims(interventions=interventions, scenlabel=simlabel)
            self.simslist.append(sims)
        return


    def run(self, *args, **kwargs):
        ''' Actually run a list of sims '''

        # Check that it's set up
        if not self.scens:
            errormsg = 'No scenarios are defined'
            raise ValueError(errormsg)
        if not self.simslist:
            self.make_scens()

        # Create msim
        msims = sc.autolist()
        for sims in self.simslist:
            msims += fps.MultiSim(sims)
        self.msim = fps.MultiSim.merge(*msims)

        # Run
        self.msim.run(**kwargs)
        self.already_run = True

        # Process
        self.msim_merged =self.msim.remerge()
        self.analyze_sims()
        return

    def check_run(self):
        ''' Give a meaningful error message if the scenarios haven't been run '''
        if not self.already_run:
            errormsg = 'Scenarios have not yet been run; please run first via scens.run()'
            raise RuntimeError(errormsg)
        return


    def plot_sims(self, **kwargs):
        ''' Plot each sim as a separate line across all senarios '''
        self.check_run()
        return self.msim.plot(plot_sims=True, **kwargs)


    def plot_scens(self, **kwargs):
        ''' Plot the scenarios with bands '''
        self.check_run()
        return self.msim_merged.plot(plot_sims=True, **kwargs)


    def plot_cpr(self, **kwargs):
        ''' Plot the CPR with bands '''
        self.check_run()
        return self.msim_merged.plot_cpr(**kwargs)


    def analyze_sims(self, start=None, end=None):
        ''' Take a list of sims that have different labels and count the births in each '''
        self.check_run()

        # Pull out first sim and parameters
        sim0 = self.msim.sims[0]
        if start is None: start = sim0.pars['start_year']
        if end   is None: end   = sim0.pars['end_year']

        def count_births(sim):
            year = sim.results['t']
            births = sim.results['births']
            inds = sc.findinds((year >= start), year < end)
            output = births[inds].sum()
            return output

        def method_failure(sim):
            year = sim.results['tfr_years']
            meth_fail = sim.results['method_failures_over_year']
            inds = sc.findinds((year >= start), year < end)
            output = meth_fail[inds].sum()
            return output

        def count_pop(sim):
            year = sim.results['tfr_years']
            popsize = sim.results['pop_size']
            inds = sc.findinds((year >= start), year < end)
            output = popsize[inds].sum()
            return output

        def mean_tfr(sim):
            year = sim.results['tfr_years']
            rates = sim.results['tfr_rates']
            inds = sc.findinds((year >= start), year < end)
            output = rates[inds].mean()
            return output

        # Split the sims up by scenario
        results = sc.objdict()
        results.sims = sc.objdict(defaultdict=sc.autolist)
        for sim in self.msim.sims:
            try:
                label = sim.scenlabel
            except:
                errormsg = f'Warning, could not extract scenlabel from sim {sim.label}; using default...'
                print(errormsg)
                label = sim.label
            results.sims[label] += sim

        # Count the births across the scenarios
        raw = sc.ddict(list)
        for key,sims in results.sims.items():
            for sim in sims:
                n_births = count_births(sim)
                n_fails  = method_failure(sim)
                n_pop = count_pop(sim)
                n_tfr = mean_tfr(sim)
                raw['scenario'] += [key]      # Append scenario key
                raw['births']   += [n_births] # Append births
                raw['fails']    += [n_fails]  # Append failures
                raw['popsize']  += [n_pop]    # Append population size
                raw['tfr']      += [n_tfr]    # Append mean tfr rates

        # Calculate basic stats
        results.stats = sc.objdict()
        for statkey in ['mean', 'median', 'std', 'min', 'max']:
            results.stats[statkey] = sc.objdict()
            for k,vals in raw.items():
                if k != 'scenario':
                    results.stats[statkey][k] = getattr(np, statkey)(vals)

        # Also save as pandas
        results.df = pd.DataFrame(raw)
        self.results = results

        return