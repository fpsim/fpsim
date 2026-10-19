'''
Specify the core interventions available in FPsim. Other interventions can be
defined by the user by inheriting from these classes.
'''
import numpy as np
import sciris as sc
import starsim as ss
from . import utils as fpu
from . import methods as fpm
from . import defaults as fpd

_make_repr = fpu.make_recreatable_intervention_repr
#>> Generic intervention classes

__all__ = ['change_par', 'update_methods', 'add_method', 'change_people_state',
           'change_initiation_prob', 'change_initiation', 'method_switching']


##%% Helper functions for JSON serialization

class change_par(ss.Intervention):
    '''
    Change a parameter at a specified point in time.

    Args:
        par   (str): the parameter to change
        years (float/arr): the year(s) at which to apply the change
        vals  (any): a value or list of values to change to (if a list, must have the same length as years); or a dict of year:value entries

    If any value is ``'reset'``, reset to the original value.

    **Example**::

        ec0 = fp.change_par(par='exposure_factor', years=[2000, 2010], vals=[0.0, 2.0]) # Reduce exposure factor
        ec0 = fp.change_par(par='exposure_factor', vals={2000:0.0, 2010:2.0}) # Equivalent way of writing
        sim = fp.Sim(interventions=ec0).run()
    '''
    def __init__(self, par, years=None, vals=None, verbose=False, **kwargs):
        super().__init__(**kwargs)
        self.par   = par
        self.verbose = verbose
        if isinstance(years, dict): # Swap if years is supplied as a dict, so can be supplied first
            vals = years
        if vals is None:
            errormsg = 'Values must be supplied'
            raise ValueError(errormsg)
        if isinstance(vals, dict):
            years = sc.dcp(list(vals.keys()))
            vals = sc.dcp(list(vals.values()))
        else:
            if years is None:
                errormsg = 'If vals is not supplied as a dict, then year(s) must be supplied'
                raise ValueError(errormsg)
            else:
                years = sc.toarray(sc.dcp(years))
                vals = sc.dcp(vals)
                if sc.isnumber(vals):
                    vals = sc.tolist(vals) # We want to be careful not to take something that might already be an array and interpret different values as years
                n_years = len(years)
                n_vals = len(vals)
                if n_years != n_vals:
                    errormsg = f'Number of years ({n_years}) does not match number of values ({n_vals})'
                    raise ValueError(errormsg)

        self.years = years
        self.vals = vals

        return

    def init_pre(self, sim):
        super().init_pre(sim)

        # Validate parameter name
        if self.par not in sim.pars.fp:
            errormsg = f'Parameter "{self.par}" is not a valid sim parameter'
            raise ValueError(errormsg)

        # Validate years and values
        years = self.years
        min_year = min(years)
        max_year = max(years)
        if min_year < sim.pars.start:
            errormsg = f'Intervention start {min_year} is before the start of the simulation'
            raise ValueError(errormsg)
        if max_year > sim.pars.stop:
            errormsg = f'Intervention end {max_year} is after the end of the simulation'
            raise ValueError(errormsg)
        if years != sorted(years):
            errormsg = f'Years {years} should be monotonic increasing'
            raise ValueError(errormsg)

        # Convert intervention years to sim timesteps
        self.counter = 0
        self.inds = sc.autolist()
        for y in years:
            self.inds += sc.findnearest(sim.t.yearvec, y)

        # Store original value
        self.orig_val = sc.dcp(sim.pars.fp[self.par])

        return

    def step(self):
        sim = self.sim
        if len(self.inds) > self.counter:
            ind = self.inds[self.counter] # Find the current index
            if sim.ti == ind: # Check if the current timestep matches
                curr_val = sc.dcp(sim.pars.fp[self.par])
                val = self.vals[self.counter]
                if val == 'reset':
                    val = self.orig_val
                sim.pars.fp[self.par] = val  # Update the parameter value -- that's it!
                if self.verbose:
                    label = f'Sim "{sim.label}": ' if sim.label else ''
                    print(f'{label}On {sim.t.year}, change {self.counter+1}/{len(self.inds)} applied: "{self.par}" from {curr_val} to {sim.pars.fp[self.par]}')
                self.counter += 1
        return

    def finalize(self):
        # Check that all changes were applied
        n_counter = self.counter
        n_vals = len(self.vals)
        if n_counter != n_vals:
            errormsg = f'Not all values were applied ({n_vals} ≠ {n_counter})'
            raise RuntimeError(errormsg)
        super().finalize()
        return


def add_method(*args, **kwargs):
    """
    Not ported to Starsim 4 yet: a method's state has to be declared when the contraception module is made. The
    declared form is a method present from the start with `weight_<method>` 0 until the year, then the share of
    `copy_from` split between them (an `ss.modify` of the two weights), with the new method's switching rows and
    column copied from `copy_from`.
    """
    raise NotImplementedError('add_method() is not ported to Starsim 4 yet: declare the method on the contraception module and give it weight 0 until the year (see the docstring)')


class change_people_state(ss.Intervention):
    """
    Intervention to modify values of a People's boolean state at one specific
    point in time.

    Args:
        state_name  (string): name of the People's state that will be modified
        new_val     (bool, float): the new state value eligible people will have
        years       (list, float): The year we want to start the intervention.
                     if years is None, uses start and end years of sim as defaults
                     if years is a number or a list with a single element, eg, 2000.5, or [2000.5],
                     this is interpreted as the start year of the intervention, and the
                     end year of intervention will be the end of the simulation
        eligibility (inds/callable): indices OR callable that returns inds
        prop        (float): a value between 0 and 1 indicating the x% of eligible people
                     who will have the new state value
        annual      (bool): whether the increase, prop, represents a "per year" increase, or per time step

    **Example**::

        # Move 10% of eligible women to urban in 2010
        def eligible(sim): return sim.people.female & (sim.people.age > 18)
        intv = fp.change_people_state('urban', new_val=True, years=2010,
                                      eligibility=eligible, prop=0.1)
        sim = fp.Sim(location='kenya', interventions=intv).run()
    """

    def __init__(self, state_name, new_val, years=None, eligibility=None, prop=1.0, annual=False, **kwargs):
        super().__init__(**kwargs)
        self.define_pars(
            state_name=state_name,
            new_val=new_val,
            years=years,
            eligibility=eligibility,
            prop=prop,
            annual=annual
        )
        self.module_name = None
        if '.' in state_name:
            self.module_name, self.pars.state_name = state_name.split('.')
        self.annual_perc = None
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        self._validate_pars()

        # Lastly, adjust the probability by the sim's timestep, if it's an annual probability
        if self.pars.annual:
            # per timestep/monthly growth rate or perc of eligible women who will be made to choose contraception
            self.annual_perc = self.pars.prop
            self.pars.prop = ((1 + self.annual_perc) ** sim.dt.years)-1
        # Validate years and values
        if self.pars.years is None:
            # f'Intervention start and end years not provided. Will use sim start an end years'
            self.pars.years = [sim.pars['start'], sim.pars['stop']]
        if sc.isnumber(self.pars.years) or len(self.pars.years) == 1:
            self.pars.years = sc.promotetolist(self.pars.years)
            # Assumes that start year has been specified, append end of the simulation as end year of the intervention
            self.pars.years.append(sim.pars['stop'])

        min_year = min(self.pars.years)
        max_year = max(self.pars.years)
        if min_year < sim.pars['start']:
            errormsg = f'Intervention start {min_year} is before the start of the simulation.'
            raise ValueError(errormsg)
        if max_year > sim.pars['stop']:
            errormsg = f'Intervention end {max_year} is after the end of the simulation.'
            raise ValueError(errormsg)
        if self.pars.years != sorted(self.pars.years):
            errormsg = f'Years {self.pars.years} should be monotonically increasing'
            raise ValueError(errormsg)

        return

    def _validate_pars(self):
        # Validation
        if self.pars.state_name is None:
            errormsg = 'A state name must be supplied.'
            raise ValueError(errormsg)
        if self.pars.new_val is None:
            errormsg = 'A new value must be supplied.'
            raise ValueError(errormsg)
        if self.pars.eligibility is None:
            errormsg = 'Eligibility needs to be provided'
            raise ValueError(errormsg)
        return

    def check_eligibility(self):
        """
        Return an array of uids of agents eligible
        """
        if callable(self.pars.eligibility):
            eligible_uids = self.pars.eligibility(self.sim)
        elif sc.isarray(self.pars.eligibility):
            eligible_uids = self.pars.eligibility
        else:
            errormsg = 'Eligibility must be a function or an array of uids'
            raise ValueError(errormsg)

        return ss.uids(eligible_uids)

    def step(self):
        if self.pars.years[0] <= self.sim.t.year <= self.pars.years[1]:  # Inclusive range
            eligible_uids = self.check_eligibility()
            if self.module_name is not None:
                self.sim.people[self.module_name][self.pars.state_name][eligible_uids] = self.pars.new_val
            else:
                self.sim.people[self.pars.state_name][eligible_uids] = self.pars.new_val
        return


def _modify(cm, key, start, name=None, **kwargs):
    """ A modifier of a parameter of the contraception module, from a year """
    return ss.modify(f'{cm}.pars.{key}', start=start, name=name, **kwargs)


def update_methods(year, eff=None, dur_use=None, p_use=None, method_mix=None, method_choice_pars=None, name=None, contraception='contraception', methods=None, verbose=False, label=None):
    """
    Change methods from a year: efficacy, duration of use, the probability of use (RandomChoice), or the method mix

    Returns a list of `ss.modify` (give it to `interventions=`), so it runs as agents and as compartments.

    Args:
        year (float): the year of the change
        eff (dict): {method name or label: efficacy}
        dur_use (dict): {method name or label: a multiplier on its duration of use}
        p_use (float): the probability of use at every decision (RandomChoice only: SimpleChoice's depends on age and ever use)
        method_mix (array): weights on the share choosing each method (one per method, excluding none), renormalized among
            methods, so the probability of using some method is unchanged (Starsim 3 FPsim's method_weights)
        methods (ndict): the methods, to look up labels (default: the standard methods)

    **Example**::

        intv = fp.update_methods(year=2010, eff={'Injectables': 0.99}, method_mix=[1, 1, 2, 1, 1, 1, 1, 1, 1])
        sim = fp.Sim(location='kenya', interventions=intv).run()
    """
    if all(x is None for x in [eff, dur_use, p_use, method_mix, method_choice_pars]):
        raise ValueError('At least one of eff, dur_use, p_use, method_mix, or method_choice_pars must be supplied')
    if method_choice_pars is not None:
        raise NotImplementedError('A new switching matrix mid-run is not ported to Starsim 4 yet; use method_mix to reweight methods')
    methods = methods if methods is not None else fpm.make_methods()
    def lookup(key):
        for m in methods.values():
            if key in [m.name, m.label]:
                return m.name
        raise ValueError(f'Method "{key}" not found')
    tag = f'_{name}' if name else ''
    mods = []
    for k, v in (eff or {}).items():
        m = lookup(k)
        mods.append(_modify(contraception, f'eff_{m}', year, set=float(v), name=f'eff_{m}{tag}'))
    for k, v in (dur_use or {}).items():
        m = lookup(k)
        if not sc.isnumber(v):
            raise NotImplementedError('dur_use is a multiplier on the duration of use; a new distribution mid-run is not ported to Starsim 4')
        src = 'none' if m == 'none' else m
        mods.append(ss.modify(f'{contraception}.stop_{src}', multiply=1/float(v), start=year, name=f'dur_{m}{tag}'))
    if p_use is not None:
        mods.append(_modify(contraception, 'p_use', year, set=float(p_use), name=f'p_use{tag}'))
    if method_mix is not None:
        names = [m for m in methods if m != 'none']
        mix = np.asarray(method_mix, dtype=float)
        if len(mix) != len(names):
            raise ValueError(f'method_mix has {len(mix)} values, but there are {len(names)} methods ({names})')
        mix = mix/mix.mean()
        for m, w in zip(names, mix):
            mods.append(_modify(contraception, f'weight_{m}', year, set=float(w), name=f'weight_{m}{tag}'))
    return mods


def change_initiation_prob(year=None, prob_use_intercept=0.0, contraception='contraception', name=None, verbose=False):
    """
    Set the intercept added to the logit of the probability of use, from a year (a list with one `ss.modify`)

    **Example**::

        intv = fp.change_initiation_prob(year=2010, prob_use_intercept=0.5)
        sim = fp.Sim(location='kenya', interventions=intv).run()
    """
    return [_modify(contraception, 'prob_use_intercept', year, set=float(prob_use_intercept), name=name or 'change_initiation_prob')]


class change_initiation(ss.Intervention):
    """
    Start a share of non-users on contraception each step

    Women not using contraception (past debut, not pregnant, not within `dur_postpartum` of a birth, under 50)
    are chosen at random and start a method: `target_method`, or one drawn from the switching shares of
    non-users by age (weighted by the current method weights). Their duration of use is drawn as for anyone
    starting the method.

    Args:
        years (list, float): the start year (to the end of the sim), or [start, end]; default the whole sim
        perc (float): the extra share of women to start, relative to the users at the start of the intervention (or of
            the eligible women, with `perc_of_eligible`), per year if `annual`
        annual (bool): whether `perc` is per year
        force_theoretical (bool): also make up the shortfall from the expected number of users
        age_range (tuple): (min_age, max_age) of the women eligible
        perc_of_eligible (bool): whether `perc` is a share of the eligible women
        target_method (str): the method new users start (default: drawn)
        final_perc (float): if given, `perc` changes linearly to this by the end year

    **Example**::

        intv = fp.change_initiation(years=2010, perc=0.05)
        sim = fp.Sim(location='kenya', interventions=intv).run()
    """
    def __init__(self, years=None, eligibility=None, perc=0.0, annual=True, force_theoretical=False,
                 age_range=None, perc_of_eligible=False, target_method=None, final_perc=None, verbose=False, **kwargs):
        super().__init__(**kwargs)
        if eligibility is not None:
            raise NotImplementedError('eligibility= is not ported: use age_range=')
        self.years = years
        self.perc = perc
        self.annual = annual
        self.force_theoretical = force_theoretical
        self.age_range = age_range
        self.perc_of_eligible = perc_of_eligible
        self.target_method = target_method
        self.final_perc = final_perc
        self.verbose = verbose
        self.init_women_oncontra = None
        self.expected_women_oncontra = None
        self.current_women_oncontra = None
        self.n_started = 0
        self._select = ss.random()
        self._method = ss.random()
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        start, stop = float(sim.pars.start), float(sim.pars.stop)
        years = [start, stop] if self.years is None else sc.tolist(self.years)
        if len(years) == 1:
            years.append(stop)
        years = [float(y) for y in years]
        if years[0] < start:
            raise ValueError(f'Intervention start {years[0]} is before the start of the simulation.')
        if years[1] > stop:
            raise ValueError(f'Intervention end {years[1]} is after the end of the simulation.')
        if years != sorted(years):
            raise ValueError(f'Years {years} should be monotonically increasing')
        if self.age_range is not None and len(self.age_range) != 2:
            raise ValueError('age_range must be a tuple of (min_age, max_age)')
        self.years = years
        cm = sim.connectors.contraception
        if self.target_method is not None and self.target_method not in cm.method_names:
            raise ValueError(f'Target method "{self.target_method}" not found')
        dt = sim.t.dt_year
        self.step_perc = lambda p: ((1 + p)**dt - 1) if self.annual else p
        return

    def current_perc(self, year):
        if self.final_perc is None:
            return self.step_perc(self.perc)
        y0, y1 = self.years
        frac = np.clip((year - y0)/(y1 - y0), 0, 1) if y1 > y0 else 1
        return self.step_perc(self.perc + frac*(self.final_perc - self.perc))

    def eligible(self):
        sim = self.sim
        cm = sim.connectors.contraception
        f = sim.demographics.fp
        ppl = sim.people
        ok = (cm['never'].raw | cm['none'].raw) & ppl.female.raw & ppl.alive.raw & (ppl.age.raw < fpd.max_age_preg) & ~f.postpartum.raw
        if self.age_range is not None:
            ok &= (ppl.age.raw >= self.age_range[0]) & (ppl.age.raw < self.age_range[1])
        return ss.uids(np.flatnonzero(ok))

    def step(self):
        sim = self.sim
        year = sim.t.now('year')
        cm = sim.connectors.contraception
        users = lambda: int(np.sum((cm.method_raw() > 0) & sim.people.alive.raw))
        if self.expected_women_oncontra is None and year >= self.years[0] - 1e-9: # The first step of the intervention (Starsim 3 compared years exactly)
            self.expected_women_oncontra = self.init_women_oncontra = users()
        if not (self.years[0] - 1e-9 <= year <= self.years[1] + 1e-9):
            return
        self.current_women_oncontra = users()
        uids = self.eligible()
        if not len(uids):
            return
        perc = self.current_perc(year)
        if self.perc_of_eligible:
            n_new = perc*len(uids)
        else:
            step_new = perc*self.expected_women_oncontra
            n_new = step_new + max(self.expected_women_oncontra - self.current_women_oncontra, 0) if self.force_theoretical else perc*self.current_women_oncontra
            self.expected_women_oncontra += step_new
        n_new = min(n_new, len(uids))
        chosen = uids[self._select.rvs(uids) < n_new/len(uids)]
        if not len(chosen):
            return
        self.start_methods(chosen)
        self.n_started += len(chosen)
        return

    def start_methods(self, uids):
        """ Move women from non-use to a method, as an early end of non-use (TODO(v4-rs): a delivery moving people along a branch chosen by the split) """
        sim = self.sim
        cm = sim.connectors.contraception
        names = cm.method_names
        if self.target_method is not None:
            method = np.full(len(uids), names.index(self.target_method))
        else:
            age = sim.people.age.raw[uids]
            groups = np.clip(np.digitize(age, fpm.age_edges) - 1, 0, len(fpm.age_edges) - 1)
            shares = cm.shares('pp0', 'none')
            w = np.array([cm.value(f'weight_{m}') for m in names])
            table = np.array([shares[m] for m in names]).T*w # Age groups by methods
            cdf = np.cumsum(table/table.sum(1, keepdims=True), axis=1)[groups]
            method = np.minimum((cdf < self._method.rvs(uids)[:, None]).sum(1), len(names) - 1)
        graph = cm.graph
        moves = [(src, j, uids[cm[src].raw[uids] & (method == j)]) for src in ['never', 'none'] for j in range(len(names))] # Before anyone moves
        for src, j, sel in moves:
            if len(sel):
                cm.engine.force(sel, graph.transitions[f'stop_{src}'], graph.states[names[j]])
        return


def method_switching(*args, **kwargs):
    """ Not ported to Starsim 4 yet: the switching shares are tables in the declared transitions; use update_methods(method_mix=) to reweight methods """
    raise NotImplementedError('method_switching() is not ported to Starsim 4 yet; use update_methods(method_mix=...) to reweight methods')
