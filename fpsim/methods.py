"""
Contraceptive methods and method choice, as a declared module

Each method is a state. A woman leaves it at the end of her duration of use (drawn on entry, by method and age)
and decides then, with her age at that moment, whether to keep using contraception and which method: one
transition per method with several destinations. Pregnancy pauses contraception, and the decisions after a
birth (at 1 and 6 months) and after a loss are transitions triggered by the pregnancy module's events.
"""

# %% Imports
import copy
import numpy as np
import sciris as sc
import starsim as ss
import pandas as pd
from scipy.stats import fisk
from . import defaults as fpd
import fpsim.locations.data_utils as fpld

__all__ = ['Method', 'make_method_list', 'ContraPars', 'make_contra_pars', 'make_methods', 'ContraceptiveChoice',
           'RandomChoice', 'SimpleChoice', 'StandardChoice', 'age_edges']

age_edges = [0, 10, 15, 18, 20, 25, 30, 35, 40, 45, 50] # Age groups of the contraception module: contain the method-choice edges and 15-49
choice_edges = [18, 20, 25, 35] # FPsim's method-choice age groups: <18, 18-20, 20-25, 25-35, >35
choice_keys = ['<18', '18-20', '20-25', '25-35', '>35']
jitter = 1e-4 # Zero switching probabilities are given this weight (Starsim 3 FPsim: |N(0, 1e-4)|, drawn each time)


# %% Methods
class Method:
    """
    A contraceptive method

    Each method is a state of the contraception module. A woman leaves it after ``dur_use`` (drawn when she
    starts, by her age), and chooses again then.

    Args:
        name (str): short identifier used in code, e.g. ``'inj'``; the name of the method's state
        label (str): display name, e.g. ``'Injectables'``
        idx (int): position in the method list; set automatically
        efficacy (float): proportional reduction in the annual probability of conception while using
        modern (bool): whether this counts towards modern contraceptive prevalence (mCPR)
        dur_use (Dist): distribution of duration of use (typed, e.g. in months); a parameter may be an ``ss.by(age=...)``
        rel_dur_use (float): multiplier on ``dur_use``
        csv_name (str): name used for this method in the input data files

    **Example**::

        dmpasc = fp.Method(name='dmpasc', label='DMPA-SC', efficacy=0.983, modern=True, dur_use=ss.lognorm_ex(ss.years(2), ss.years(1)))
    """
    def __init__(self, name=None, label=None, idx=None, efficacy=None, modern=None, dur_use=None, rel_dur_use=1, csv_name=None, _source_dur=None):
        self.name = name
        self.label = label or name
        self.csv_name = csv_name or label or name
        self.idx = idx
        self.efficacy = efficacy
        self.modern = modern
        self.dur_use = dur_use
        self.rel_dur_use = rel_dur_use
        self._source_dur = _source_dur

    def __repr__(self):
        return f'Method({self.name}, eff={self.efficacy}, modern={self.modern})'

    @property
    def efficacy(self):
        """ The efficacy: in a contraception module, its current value (with any modifiers, e.g. from update_methods()) """
        mod = self.__dict__.get('_mod')
        if mod is not None and mod.__dict__.get('pars') is not None and f'eff_{self.name}' in mod.pars:
            return mod.value(f'eff_{self.name}')
        return self.__dict__.get('_efficacy')

    @efficacy.setter
    def efficacy(self, value):
        self.__dict__['_efficacy'] = value
        return

    def __deepcopy__(self, memo):
        """ A copy of a method alone is not in a module; a copy of the module it is in is in the copy of the module """
        new = Method.__new__(Method)
        memo[id(self)] = new
        for key, val in self.__dict__.items():
            if key != '_mod':
                new.__dict__[key] = copy.deepcopy(val, memo)
        mod = self.__dict__.get('_mod')
        if mod is not None and id(mod) in memo:
            new.__dict__['_mod'] = memo[id(mod)]
        return new

    def __setattr__(self, key, value):
        """ Once in a contraception module, changing the efficacy or duration of use changes its parameters, as in Starsim 3 FPsim """
        super().__setattr__(key, value)
        mod = self.__dict__.get('_mod')
        if mod is not None and key in ['efficacy', 'dur_use', 'rel_dur_use']:
            mod.method_changed(self, key)
        return


def make_method_list(methods_df):
    """ Create methods from a DataFrame with columns name, label, csv_name, efficacy, modern """
    return [Method(name=row['name'], label=row['label'], csv_name=row['csv_name'], idx=idx, efficacy=float(row['efficacy']), modern=bool(row['modern']))
            for idx, row in methods_df.iterrows()]


def make_methods(method_list=None, method_df=None):
    """ The default method list (an ndict keyed by name), or one built from a list or a dataframe """
    if method_list is None:
        if method_df is None:
            method_df = make_contra_pars().methods_df
        method_list = make_method_list(method_df)
    for i, method in enumerate(method_list): # Methods made by hand get their position
        if method.idx is None:
            method.idx = i
    return ss.ndict(method_list, type=Method)


class Fisk(ss.Dist):
    """ The log-logistic distribution (scipy's fisk) """
    def __init__(self, c=0.0, scale=1.0, **kwargs):
        super().__init__(distname='fisk', dist=fisk, c=c, scale=scale, **kwargs)
        return


def method_bins(edges=None):
    """ The method-choice age group (0-4) of each age group of the module """
    edges = age_edges if edges is None else edges
    return np.digitize(edges, choice_edges)


def dur_use_pars(df, method, edges=None):
    """
    A method's duration of use from the fitted survival model (months), with the age factors on its location
    parameter: (form, parameters, each one per age group). Parameters are as reported by flexsurv's `res.t`
    (shapes, scales, rates and sdlog on the log scale; meanlog as is), so the shape is exponentiated for every
    form. Starsim 3 FPsim matched 'lognorm' against the data's 'lnorm' (so these methods kept a 1 +/- 1 year
    lognormal), read sdlog as a standard deviation in months, and used the Weibull shape unexponentiated.
    """
    thisdf = df.loc[df.method == method.csv_name]
    if not len(thisdf):
        raise ValueError(f'No duration of use for {method.csv_name}')
    form = thisdf.functionform.iloc[0]
    coef = lambda key: float(thisdf.estimate[thisdf.coef == key].values[0])
    af = np.zeros(5)
    i0 = sc.findfirst(thisdf.coef.values, 'age_grp_fact(0,18]')
    if i0 is not None:
        vals = thisdf.estimate.values[i0:].astype(float)
        af[:len(vals)] = vals
    af = af[method_bins(edges)] # One per age group
    n = len(af)
    if form in ['lognormal', 'lnorm']:
        return 'lognormal', dict(meanlog=coef('meanlog') + af, sdlog=np.full(n, np.exp(coef('sdlog'))))
    elif form == 'gamma':
        return 'gamma', dict(a=np.full(n, np.exp(coef('shape'))), scale=1/np.exp(coef('rate') + af))
    elif form == 'llogis':
        return 'llogis', dict(c=np.full(n, np.exp(coef('shape'))), scale=np.exp(coef('scale') + af))
    elif form == 'weibull':
        return 'weibull', dict(c=np.full(n, np.exp(coef('shape'))), scale=np.exp(coef('scale') + af))
    elif form == 'exponential':
        return 'exponential', dict(scale=1/np.exp(coef('rate') + af))
    raise ValueError(f'Duration of use distribution {form} not recognized')


def dur_use_dist(df, method, edges=None):
    """ A method's duration of use (see `dur_use_pars()`), as a distribution whose parameters are by age group """
    form, p = dur_use_pars(df, method, edges)
    months = lambda arr: ss.by(age=[ss.months(float(v)) for v in arr])
    if form == 'lognormal':
        mean = np.exp(p['meanlog'] + p['sdlog']**2/2)
        return ss.lognorm_ex(mean=months(mean), std=months(mean*np.sqrt(np.exp(p['sdlog']**2) - 1)))
    elif form == 'gamma':
        return ss.gamma(a=p['a'][0], scale=months(p['scale']))
    elif form == 'llogis':
        return Fisk(c=p['c'][0], scale=months(p['scale']))
    elif form == 'weibull':
        return ss.weibull(c=p['c'][0], scale=months(p['scale']))
    return ss.expon(scale=months(p['scale']))


def dur_use_survival(df, method, t, edges=None):
    """ The share still using a method `t` months after starting, by age group """
    import scipy.stats as sps
    form, p = dur_use_pars(df, method, edges)
    if form == 'lognormal':
        return sps.lognorm(s=p['sdlog'], scale=np.exp(p['meanlog'])).sf(t)
    elif form == 'gamma':
        return sps.gamma(a=p['a'], scale=p['scale']).sf(t)
    elif form == 'llogis':
        return sps.fisk(c=p['c'], scale=p['scale']).sf(t)
    elif form == 'weibull':
        return sps.weibull_min(c=p['c'], scale=p['scale']).sf(t)
    return np.exp(-t/p['scale'])


# %% Parameters
class ContraPars(ss.Pars):
    def __init__(self, **kwargs):
        super().__init__()

        # Methods
        self.methods = None  # Will be populated from methods_df
        self.methods_file = sc.makepath('methods.csv', fpld.sd_dir)  # Store the provenance
        self.methods_df = pd.read_csv(self.methods_file, keep_default_na=False, na_values=['NaN'])
        self.method_column_order = None

        # Probabilities and choices
        self.p_use = 0.5  # RandomChoice: probability of use at each decision
        self.method_weights = None  # Weights on each method's share when choosing; renormalized among methods
        self.switches = None  # Pairs (from, to) with a switching parameter: see ContraceptiveChoice.add_switching()

        # mCPR trend: added to the logit of the probability of use
        self.prob_use_year = 2000
        self.prob_use_intercept = 0.0
        self.prob_use_trend_par = 0.0

        # Data pars, all None by default and populated with data
        self.age_spline = None
        self.init_dist = None
        self.dur_use_df = None
        self.contra_use_pars = None
        self.method_choice_pars = None
        self.intent_pars = None
        self.max_dur = ss.years(100)
        self.update(kwargs)
        return


def make_contra_pars():
    """ Shortcut for making a new instance of ContraPars """
    return ContraPars()


def _fp_event(name, fp_name='fp'):
    """ An event of the sim's FP module, by name, for a module made before it """
    return ss.conditions.Event('tr', ss.conditions.Named(fp_name, name))


# %% The contraception module
class ContraceptiveChoice(ss.Connector, ss.Disease):
    """
    Contraceptive use and method choice, as declared states and transitions

    States: `pre` (before sexual debut; everyone starts here), `never` (never used),
    `none` (not using, has used), one state per method, `preg0`/`preg1` (pregnant, by ever use), `pp0`/`pp1` (the
    first month after a birth) and `wait0`/`wait1` (months 1-6 after a birth, not using). Leaving a method or
    non-use at the end of its duration is one transition per state that splits between not using and each method,
    by the probability of use (a logistic by age, ever use and year) and the switching shares (by age and the
    method left, weighted by `method_weights`, renormalized among methods). Debut, pregnancy, delivery and losses
    come from the FP module (`fp`).

    Subclasses give the probability of use and the switching shares: see `SimpleChoice` and `RandomChoice`.

    Args:
        pars (dict): parameters (see `ContraPars`)
        methods (list): the methods (default from `methods.csv`)
        fp_name (str): the name of the FP module whose events drive pauses and postpartum decisions
    """
    def __init__(self, pars=None, methods=None, fp_name='fp', name='contraception', **kwargs):
        defaults = dict(ContraPars())
        pars = sc.mergedicts(defaults, dict(pars or {}), {k: kwargs.pop(k) for k in list(kwargs) if k in defaults})
        if methods is not None:
            pars['methods'] = methods
        method_list = pars['methods'] if pars['methods'] is not None else make_method_list(pars['methods_df'])
        self.methods = make_methods(sc.dcp(list(method_list.values()) if isinstance(method_list, dict) else method_list))
        self.fp_name = fp_name
        self._init_kwargs = dict(fp_name=fp_name, name=name, **kwargs) # To build the module again with another method: see add_method()
        self._data = pars
        n = self.n_methods
        if pars['method_weights'] is None:
            pars['method_weights'] = np.ones(n)
        self.prepare(pars)
        states, trs, graph_pars = self.make_graph(pars)
        effs = {k: v for k, v in graph_pars.items() if k.startswith('eff_')} # Read by the FP module, not by a transition
        super().__init__(*states, **trs, **effs, name=name, **kwargs)
        self.stratify(age=age_edges)
        self.define_pars(**{k: v for k, v in pars.items() if k not in self.pars and k not in graph_pars}) # Data read when the graph is built and by the code, not by the graph
        for method in self.methods.values():
            method._mod = self
        return

    def add_method(self, method, copy_from=None, weight=1.0):
        """
        Add a method, before the sim is initialized: each method is a state, so the module is built again with it

        Args:
            method (Method): the method
            copy_from (str): the method whose switching shares it takes (to it and from it); SimpleChoice needs one
            weight (float): its weight (`weight_<name>`); 0 makes it available only once an `ss.modify` raises it (see `fp.add_method`)
        """
        if self.sim is not None and self.sim.initialized:
            raise RuntimeError(f'A method can only be added to "{self.name}" before the sim is initialized: use fp.add_method(year=...) to make it available from a year')
        if method.name in self.methods:
            raise ValueError(f'There is already a method named "{method.name}"')
        method = sc.dcp(method)
        method.idx = max(m.idx for m in self.methods.values()) + 1
        source = self.get_method(copy_from) if copy_from is not None else None
        pars = self._data
        self.extend_data(pars, method, source)
        pars['method_weights'] = np.append(pars['method_weights'][:self.n_methods], weight)
        pars['_new_methods'] = list(self.methods.values()) + [method]
        self.rebuild(pars)
        return self.methods[method.name]

    def extend_data(self, pars, method, source):
        """ The data for a new method (subclasses: see SimpleChoice) """
        return

    def rebuild(self, pars):
        """ Make the module again with changed data (before the sim is initialized), keeping the current values of its parameters """
        if self.sim is not None and self.sim.initialized:
            raise RuntimeError(f'"{self.name}" can only be changed before the sim is initialized')
        pars['method_weights'] = np.array([self.value(f'weight_{m}') for m in self.method_names] + list(pars['method_weights'][self.n_methods:]))
        for key in [k for k in ['prob_use_intercept', 'prob_use_trend_par', 'p_use'] if isinstance(self.pars.get(k), ss.par)]: # Their current values
            pars[key] = self.value(key)
        methods = list(pars.pop('_new_methods', None) or self.methods.values())
        for m in methods: # Their current efficacies
            if m.name != 'none' and f'eff_{m.name}' in self.pars:
                m.__dict__['_efficacy'] = self.value(f'eff_{m.name}')
        self._lock_attrs = False # UPSTREAM: a declared module cannot add a state or parameter once made, so it is made again
        ContraceptiveChoice.__init__(self, pars=pars, methods=methods, **self._init_kwargs)
        return

    def add_switching(self, from_method, to_method):
        """
        Add a parameter, `switch_<from>_<to>` (0 by default), for the share of women choosing a method after `from_method`
        (a method name, or 'birth' for the choice 1 month after a birth) who choose `to_method` instead; before the sim is
        initialized. See `fp.method_switching`.
        """
        frm = 'birth' if from_method == 'birth' else self.get_method(from_method).name
        to = self.get_method(to_method).name
        key = f'switch_{frm}_{to}'
        if key not in self.pars:
            pars = self._data
            pars['switches'] = list(pars.get('switches') or []) + [(frm, to)]
            self.rebuild(pars)
        return key

    def switching_row(self, postpartum, from_method=None):
        """ The current shares of each method (weighted, renormalized among methods) by method-choice age group, for a decision after `postpartum` months (0, 1, or 6) from a method """
        event = {0: 'pp0', 1: 'pp1', 6: 'pp6'}[postpartum]
        src = None if event == 'pp1' else self.get_method(from_method or 'none').name
        edges = [0] + choice_edges # One per method-choice age group
        sh = self.shares(event, src, edges=edges) if 'edges' in self.shares.__code__.co_varnames else self.shares(event, src)
        w = {m: self.value(f'weight_{m}') for m in self.method_names}
        total = sum(w[m]*np.asarray(sh[m], dtype=float)[:len(edges)] for m in self.method_names)
        out = {key: {m: float(w[m]*sh[m][i]/total[i]) if total[i] > 0 else 0.0 for m in self.method_names} for i, key in enumerate(choice_keys)}
        for frm, to in self._data.get('switches') or []: # See add_switching()
            if frm == ('birth' if event == 'pp1' else src):
                sw = self.group_value(f'switch_{frm}_{to}')
                out = {key: {m: (1 - sw[i])*v + (sw[i] if m == to else 0) for m, v in row.items()} for i, (key, row) in enumerate(out.items())}
        return out

    def group_value(self, key):
        """ The current value of a parameter in each method-choice age group """
        val = self.value(key)
        if np.ndim(val) == 0:
            return np.full(len(choice_keys), float(val))
        ppl = self.sim.people # As agents, one value per agent (by UID): that of the agents in each group, who all have the same
        age, auids = ppl.age.raw, ppl.auids
        bins = np.digitize(age[auids], choice_edges)
        return np.array([val[auids[bins == i][0]] if np.any(bins == i) else 0.0 for i in range(len(choice_keys))]) # UPSTREAM: no value of a parameter by level of a stratification, only by agent

    def get_switching_prob(self, from_method, to_method, postpartum, age_grp):
        """ The current probability of choosing `to_method`, among those choosing a method, for a woman in an age group (e.g. '18-20') leaving `from_method` """
        return self.switching_row(postpartum, from_method)[age_grp][self.get_method(to_method).name]

    def get_switching_matrix(self, postpartum, from_method=None):
        """ The current shares by age group: {age group: array over the methods (excluding none)} """
        row = self.switching_row(postpartum, from_method)
        return {key: np.array([vals[m] for m in self.method_names]) for key, vals in row.items()}

    def method_changed(self, method, key):
        """ A method's efficacy or duration of use was changed: update its parameter """
        if key == 'efficacy' and method.name != 'none':
            self.pars.update({f'eff_{method.name}': float(method.efficacy)})
        elif key in ['dur_use', 'rel_dur_use'] and f'dur_{method.name}' in self.pars:
            self.pars.update({f'dur_{method.name}': self.dur_use(method)})
        return

    # %% Building the graph
    @property
    def n_options(self):
        """ Number of options, including none """
        return len(self.methods)

    @property
    def n_methods(self):
        return len(self.method_names)

    @property
    def method_names(self):
        return [m for m in self.methods if m != 'none']

    def prepare(self, pars):
        """ Process data into the tables the graph needs (subclasses) """
        return

    def logit_use(self, event, ever):
        """ The logit of the probability of use, by age group (bare numbers; subclasses) """
        raise NotImplementedError

    def prob_use(self, event, ever, trend, intercept):
        """ The probability of use at a decision: a logistic of the age-group table, the trend in time, and the intercept """
        return 1/(1 + np.exp(-(ss.by(age=list(self.logit_use(event, ever))) + trend + intercept)))

    def shares(self, event, src):
        """ The switching shares of each method by age group, for a decision of `event` from method `src` (unweighted; subclasses) """
        raise NotImplementedError

    def dur_use(self, method):
        """ The distribution of duration of use of a method """
        dist = method.dur_use if method.dur_use is not None else ss.lognorm_ex(mean=ss.years(1), std=ss.years(1))
        if method.rel_dur_use != 1:
            dist = sc.dcp(dist)
            for k, v in dist.pars.items():
                if k in ['mean', 'std', 'scale'] and v is not None:
                    dist.pars[k] = v*method.rel_dur_use if not isinstance(v, ss.by) else v.map(lambda x: x*method.rel_dur_use)
        return dist

    def make_graph(self, pars):
        """ The states and transitions """
        names = self.method_names
        S = sc.objdict({k: ss.State(k) for k in ['pre', 'never', 'none'] + names + ['preg0', 'preg1', 'pp0', 'pp1', 'wait0', 'wait1']})
        meth = [S[m] for m in names]
        gp = dict(prob_use_intercept=ss.par(float(pars['prob_use_intercept']), name='prob_use_intercept', doc='Added to the logit of the probability of use'),
                  prob_use_trend_par=ss.par(float(pars['prob_use_trend_par']), name='prob_use_trend_par', doc='Change in the logit of the probability of use per year'))
        w = {m: ss.par(float(pars['method_weights'][i]), name=f'weight_{m}', doc=f'Weight on the share choosing {m} (renormalized among methods)') for i, m in enumerate(names)}
        eff = {m: ss.par(float(self.methods[m].efficacy), name=f'eff_{m}', doc=f'Efficacy of {m} against conception') for m in names}
        gp.update({f'weight_{m}': w[m] for m in names})
        gp.update({f'eff_{m}': eff[m] for m in names})
        self._weights = w
        self._effs = eff
        year0 = float(pars['prob_use_year'])
        trend = gp['prob_use_trend_par']*ss.timeseries({1800: 1800 - year0, 2300: 2300 - year0}, mode='linear') # (year - prob_use_year), exact
        intercept = gp['prob_use_intercept']

        def p_use(event, ever):
            return self.prob_use(event, ever, trend, intercept)

        switches = {}
        for frm, to in pars.get('switches') or []: # See add_switching()
            switches[(frm, to)] = gp[f'switch_{frm}_{to}'] = ss.par(0.0, name=f'switch_{frm}_{to}', doc=f'Share of those choosing a method after {frm} who choose {to} instead (by age group)')

        def split(event, src, ever, rest, base=None):
            """ [rest, method 1, ...]: not using w.p. 1 - p_use, else each method by its weighted share (and the switches to a method, see add_switching()) """
            pu = p_use(event, ever)
            sh = base if base is not None else self.shares(event, src)
            R = {m: ss.by(age=list(sh[m])) for m in names}
            total = None
            for m in names:
                total = w[m]*R[m] if total is None else total + w[m]*R[m]
            share = {m: w[m]*R[m]/total for m in names}
            for (frm, to), sw in switches.items():
                if frm == ('birth' if event == 'pp1' else src):
                    share = {m: (1 - sw)*share[m] + (sw if m == to else 0) for m in names}
            return [rest, *meth], [1 - pu, *[pu*share[m] for m in names]]

        fpn = self.fp_name
        age, female = ss.ppl.age, ss.ppl.female
        debuted = female & (age >= ss.state(f'{fpn}.fated_debut')) & (age < fpd.max_age_preg)
        pregnant = ss.state(f'{fpn}.P')
        trs = sc.objdict()
        durs = {}
        dst, p = split('pp0', 'none', 0, S.never)
        trs.debut = ss.Transition(S.pre, dst, p=p, on=debuted) # Everyone starts in pre, so women past their debut decide at the start (as in Starsim 3 FPsim)
        for src in ['never', 'none'] + names:
            if src == 'btl':
                continue # Permanent
            ever = int(src != 'never')
            dst, p = split('pp0', 'none' if src == 'never' else src, ever, S.never if src == 'never' else S.none)
            method = self.methods['none'] if src in ['never', 'none'] else self.methods[src]
            if method.name not in durs:
                durs[method.name] = ss.par(self.dur_use(method), name=f'dur_{method.name}', doc=f'Duration of use of {method.name}' if method.name != 'none' else 'Duration of non-use')
            trs[f'stop_{src}'] = ss.Transition(S[src], dst, p=p, dur=durs[method.name])
        losses = ss.conditions.Event('tr', ss.conditions.Named(fpn, ('miscarriage', 'stillbirth')))
        for e, srcs in [(0, [S.pre, S.never, S.pp0, S.wait0]), (1, [S.none, S.pp1, S.wait1])]:
            rest = S.never if e == 0 else S.none
            trs[f'pause{e}'] = ss.Transition(srcs, S[f'preg{e}'], on=pregnant) # Starsim 3: method = 0 at conception (not for abortions)
            if e == 1:
                trs.fail = ss.Transition(meth, S.preg1, on=pregnant) # A continuing pregnancy on a method
            trs[f'deliver{e}'] = ss.Transition(S[f'preg{e}'], S[f'pp{e}'], on=_fp_event('delivery', fpn))
            dst, p = split('pp0', 'none', e, rest)
            trs[f'loss{e}'] = ss.Transition(S[f'preg{e}'], dst, p=p, on=losses) # Starsim 3: the next step
            dst, p = split('pp1', None, e, S[f'wait{e}'])
            trs[f'pp1_{e}'] = ss.Transition(S[f'pp{e}'], dst, p=p, dur=ss.constant(ss.months(1))) # Starsim 3: the step after delivery
            dst, p = split('pp6', 'none', e, rest)
            trs[f'pp6_{e}'] = ss.Transition(S[f'wait{e}'], dst, p=p, dur=ss.constant(ss.months(5)))
        return list(S.values()), dict(trs), gp

    @property
    def decisions(self):
        """ The transitions that are decisions about contraception (the end of a method or of non-use, a loss, and 1 and 6 months after a birth) """
        return [k for k in self.graph.transitions if k == 'debut' or k.startswith(('stop_', 'loss', 'pp1_', 'pp6_'))]

    def step(self):
        ss.Disease.step_state(self)
        ss.Disease.step(self)
        return

    # %% Methods and parameters
    def get_method(self, identifier):
        """ A method by name (first) or label """
        if identifier in self.methods:
            return self.methods[identifier]
        for method in self.methods.values():
            if method.label == identifier:
                return method
        raise ValueError(f'Method "{identifier}" not found. Available names: {list(self.methods.keys())}, labels: {[m.label for m in self.methods.values()]}')

    def get_method_by_label(self, method_label):
        return self.get_method(method_label)

    def efficacy(self, name):
        """ The current efficacy of a method (with modifiers) """
        return 0.0 if name == 'none' else self.value(f'eff_{name}')

    # %% Per-agent views (agents only)
    def method_raw(self):
        """ The index of each agent's method (0 if none), by UID """
        idx = np.array([self.methods[m].idx for m in self.method_names] + [0]) # -1, in no method, is the last
        return idx[self.current_state(self.method_names).raw]

    def as_arr(self, raw, cls):
        """ A per-agent array (by UID) as an ss.Arr, so it can be indexed like a state; read-only, since it is computed from the states (writing to it would change nothing) """
        out = self.sim.people.age.asnew(raw, cls=cls)
        out.raw.flags.writeable = False
        return out

    @property
    def method(self):
        """ The index of each agent's method (0 if none), as in Starsim 3 FPsim; setting it moves women to that method (see set_method()) """
        out = self.as_arr(self.method_raw(), MethodArr)
        out.cm = self
        return out

    def to_uids(self, key):
        """ The UIDs that an index of a per-agent array selects """
        auids = self.sim.people.auids
        if isinstance(key, ss.uids):
            return key
        if isinstance(key, ss.BoolArr):
            return key.uids
        if isinstance(key, slice):
            return auids[key]
        key = np.asarray(key)
        if key.dtype == bool:
            return auids[key] if len(key) == len(auids) else ss.uids(np.flatnonzero(key))
        return ss.uids(key)

    def set_method(self, uids, idx):
        """ Move women who are deciding about contraception (never-users, non-users, and users) to a method by its index (0: not using), as if they had chosen it now """
        uids = ss.uids(uids)
        idx = np.broadcast_to(np.asarray(idx, dtype=int), (len(uids),))
        by_idx = {m.idx: m.name for m in self.methods.values()}
        src = ['never', 'none'] + self.method_names
        moved = []
        for i in np.unique(idx):
            these = uids[idx == i]
            if by_idx.get(i) in [None, 'none']: # Stopping: never-users stay never-users
                moved.append(self.move_to(these, 'none', src=self.method_names))
            else:
                moved.append(self.move_to(these, by_idx[i], src=[s for s in src if s != by_idx[i]]))
        return ss.uids.concatenate(moved) if moved else ss.uids()

    def set_ever_used(self, uids, value=True):
        """ Make women ever-users (or never-users), in whatever state of contraception or pregnancy they are in """
        pairs = [('never', 'none'), ('preg0', 'preg1'), ('pp0', 'pp1'), ('wait0', 'wait1')]
        uids = ss.uids(uids)
        for a, b in pairs:
            frm, to = (a, b) if value else (b, a)
            self.move_to(uids, to, src=frm)
        return

    @property
    def on_contra(self):
        return self.as_arr(self.method_raw() > 0, ss.BoolArr)

    @property
    def ti_contra(self):
        """ The step of each agent's next decision about contraception (the scheduled end of her method or of non-use; NaN if none is scheduled), as in Starsim 3 FPsim """
        out = np.full(len(self.sim.people.uid.raw), np.nan)
        for name in self.graph.states:
            key = f'ti_exit_{name}'
            if hasattr(self, key): # A state with a scheduled exit
                inside = self[name].raw
                out[inside] = np.ceil(self[key].raw[inside])
        return self.as_arr(out, ss.FloatArr)

    @property
    def ever_used_contra(self):
        """ Whether each woman has ever used contraception; setting it moves women between the states of never and ever users (see set_ever_used()) """
        out = np.ones(len(self.sim.people.uid.raw), dtype=bool)
        for k in ['pre', 'never', 'preg0', 'pp0', 'wait0']:
            out &= ~self[k].raw
        out = self.as_arr(out, EverUsedArr)
        out.cm = self
        return out

    def eff_array(self):
        """ Each agent's contraceptive efficacy """
        out = np.zeros(len(self.sim.people.uid.raw))
        for m in self.method_names:
            out[self[m].raw] = self.efficacy(m)
        return out

    # %% Results
    def init_results(self):
        super().init_results()
        kw = dict(scale=False, dtype=float)
        self.define_results(
            ss.Result('n_at_risk_non_users', scale=True, label='Non-users at risk of pregnancy (aCPR, expected)'),
            ss.Result('n_at_risk_users', scale=True, label='Users at risk of pregnancy (aCPR, expected)'),
            ss.Result('n_non_users', scale=True, label='Non-users (CPR)'),
            ss.Result('n_mod_users', scale=True, label='Modern contraceptive users (mCPR)'),
            ss.Result('n_users', scale=True, label='Contraceptive users (CPR)'),
            ss.Result('mcpr', label='Modern contraceptive prevalence rate (mCPR)', **kw),
            ss.Result('cpr', label='Contraceptive prevalence rate (CPR)', **kw),
            ss.Result('acpr', label='Active contraceptive prevalence rate (aCPR, expected)', **kw),
        )
        self.method_mix = np.zeros((self.n_options, self.t.npts))
        return

    def update_results(self):
        """ mCPR, CPR and aCPR among women 15-49 (DHS definition: LAM not counted); method mix among women 15-50 """
        super().update_results()
        ppl = self.sim.people
        ti = self.ti
        res = self.results
        women = ppl.female.raw & ppl.alive.raw
        age = ppl.age.raw
        denom = women & (age >= fpd.min_age) & (age < fpd.max_age_preg)
        method = self.method_raw()
        modern = np.isin(method, [m.idx for m in self.methods.values() if m.modern])
        scale = self.sim._current_scale
        n = denom.sum()
        res.n_non_users[ti] = np.sum(denom & (method == 0))*scale
        res.n_mod_users[ti] = np.sum(denom & modern)*scale
        res.n_users[ti] = np.sum(denom & (method > 0))*scale
        res.mcpr[ti] = sc.safedivide(np.sum(denom & modern), n)
        res.cpr[ti] = sc.safedivide(np.sum(denom & (method > 0)), n)
        fpm = self.sim.demographics.get(self.fp_name) if self.sim.demographics else None
        active = fpm.p_sexually_active() if hasattr(fpm, 'p_sexually_active') else None
        if active is not None: # Expected values, since activity is not drawn per woman: the chance each woman is active
            at_risk = denom & ~fpm.P.raw
            a = np.where(at_risk, active, 0)
            res.n_at_risk_users[ti] = np.sum(a*(method > 0))*scale
            res.n_at_risk_non_users[ti] = np.sum(a*(method == 0))*scale
            res.acpr[ti] = sc.safedivide(np.sum(a*(method > 0)), a.sum())
        mix = women & (age >= fpd.min_age) & (age <= fpd.max_age_preg)
        counts = np.bincount(method[mix], minlength=self.n_options)[:self.n_options]
        self.method_mix[:, ti] = counts/counts.sum() if counts.sum() else 0
        return


class MethodArr(ss.IntArr):
    """ Each woman's method (`ContraceptiveChoice.method`): setting some moves them to that method """
    def __setitem__(self, key, value):
        self.cm.set_method(self.cm.to_uids(key), value)
        return


class EverUsedArr(ss.BoolArr):
    """ Whether each woman has ever used contraception (`ContraceptiveChoice.ever_used_contra`): setting some moves them between never and ever users """
    def __setitem__(self, key, value):
        uids = self.cm.to_uids(key)
        value = np.broadcast_to(np.asarray(value, dtype=bool), (len(uids),))
        self.cm.set_ever_used(uids[value], True)
        self.cm.set_ever_used(uids[~value], False)
        return


class RandomChoice(ContraceptiveChoice):
    """
    Choose a method uniformly at random, with a probability of use of `p_use` (0.5) at every decision, and durations
    of use of 1 +/- 1 year: for testing model mechanics, not for realistic method mix
    """
    def __init__(self, pars=None, **kwargs):
        super().__init__(pars=pars, **kwargs)
        return

    def prob_use(self, event, ever, trend, intercept):
        """ The same at every decision: `p_use`, a parameter (so update_methods(p_use=) is an ss.modify of it) """
        if not hasattr(self, '_p_use'):
            self._p_use = ss.par(float(self._data['p_use']), name='p_use', doc='Probability of use at each decision')
        return self._p_use

    def shares(self, event, src):
        return {m: np.ones(len(age_edges)) for m in self.method_names}

    def dur_use(self, method):
        return ss.lognorm_ex(mean=ss.years(1), std=ss.years(1))


class SimpleChoice(ContraceptiveChoice):
    """
    Choose a method based on age and previous method, from location-specific fitted data

    The probability of use is a logistic regression on age group, ever use and year (`contra_use_pars`; one set
    each for decisions at the end of a method, 1 month after a birth, and 6 months after it); the method chosen
    follows the switching matrix (`method_choice_pars`) for the method left and the age group; durations of use
    follow each method's fitted distribution by age group (`dur_use_df`).

    Args:
        pars (dict): parameters (see `ContraPars`)
        location (str): location whose fitted coefficients to load, e.g. ``'kenya'``
        data (dict): pre-loaded data, used instead of loading from ``location``
        contra_mod (str): which fitted coefficient set to use ('simple')

    **Example**::

        sim = fp.Sim(location='kenya', contraception_module=fp.SimpleChoice(location='kenya'))
    """
    def __init__(self, pars=None, location=None, data=None, contra_mod='simple', **kwargs):
        if data is None and location is not None:
            data = fpd.get_dataloader(location).load_contra_data(contra_mod, return_data=True)
        self.location = location
        pars = sc.mergedicts(data, pars)
        super().__init__(pars=pars, **kwargs)
        return

    def prepare(self, pars):
        mcp = pars.get('method_choice_pars')
        if mcp is None or pars.get('contra_use_pars') is None:
            raise ValueError('SimpleChoice needs method_choice_pars and contra_use_pars: give a location or data')
        names = self.method_names
        idx = {m.idx: m.name for m in self.methods.values()}
        self._cols = [idx[i] for i in mcp[0]['method_idx']] # Method of each column of the switching rows
        if sorted(self._cols) != sorted(names):
            raise ValueError(f'The switching data have methods {self._cols}, but the methods are {names}')
        if pars.get('dur_use_df') is not None:
            for m in self.methods.values():
                if m.name != 'btl' and m.dur_use is None:
                    m.dur_use = dur_use_dist(pars['dur_use_df'], m)
        if pars.get('init_dist') is None:
            pars['init_dist'] = sc.objdict({k: mcp[0][k]['none']/mcp[0][k]['none'].sum() for k in choice_keys})
        return

    def extend_data(self, pars, method, source):
        """ A new method's column (its share as a destination) and row (its shares as a source) are copies of `source`'s """
        if source is None:
            raise ValueError(f'{type(self).__name__} needs copy_from= to add a method: the method whose switching shares it takes')
        mcp = sc.dcp(pars['method_choice_pars'])
        for key, rows in mcp.items():
            col = list(rows['method_idx']).index(source.idx)
            rows['method_idx'] = np.append(rows['method_idx'], method.idx)
            for age in choice_keys:
                if key == 1:
                    rows[age] = np.append(rows[age], rows[age][col])
                else:
                    for frm in list(rows[age].keys()):
                        rows[age][frm] = np.append(rows[age][frm], rows[age][frm][col])
                    if source.name in rows[age]:
                        rows[age][method.name] = np.append(rows[age][source.name][:-1], 0.0) # Not to itself
        pars['method_choice_pars'] = mcp
        pars['init_dist'] = None # Made again from the switching shares
        return

    def logit_use(self, event, ever, edges=None):
        """ Starsim 3 SimpleChoice.get_prob_use, by age group, without the trend and intercept """
        p = self._data['contra_use_pars'][dict(pp0=0, pp1=1, pp6=2)[event]]
        ab = method_bins(edges)
        out = p.intercept + np.asarray(p.age_factors)[ab]
        if ever:
            aeuf = np.asarray(p.age_ever_user_factors)
            out = out + p.fp_ever_user + np.where(ab > 1, aeuf[np.maximum(ab - 1, 0)], 0)
        return out

    def _rows(self, rows, edges=None):
        """ Shares by method name, one per age group, with zeros jittered """
        out = {}
        ab = method_bins(edges)
        for j, m in enumerate(self._cols):
            vals = np.array([rows[choice_keys[b]][j] for b in ab], dtype=float)
            out[m] = np.where(vals > 0, vals, jitter)
        return out

    def shares(self, event, src, edges=None):
        mcp = self._data['method_choice_pars']
        if event == 'pp1':
            return self._rows({k: mcp[1][k] for k in choice_keys}, edges)
        key = 0 if event == 'pp0' else 6
        return self._rows({k: mcp[key][k][src] for k in choice_keys}, edges)


class StandardChoice(SimpleChoice):
    """
    FPsim's default: the probability of use is a logistic regression on age (a spline, by single year of age), ever
    use, education, urban residence, wealth quintile, parity and year; method choice and durations are as in
    `SimpleChoice`, from the same location's fitted data (`contra_mod='mid'`)

    The probability of use is a declared expression of the women's own arrays (`people.urban`,
    `people.wealthquintile`, `fp.parity`, `edu.attainment`), so it needs an education module (`fp.Education`,
    which `fp.Sim` adds by default).

    Args:
        pars (dict): parameters (see `ContraPars`)
        location (str): location whose fitted coefficients to load, e.g. ``'kenya'``
        data (dict): pre-loaded data, used instead of loading from ``location``
        contra_mod (str): which fitted coefficient set to use ('mid')
        edu_name (str): the name of the education module

    **Example**::

        sim = fp.Sim(location='kenya', contraception_module=fp.StandardChoice(location='kenya'))
    """
    def __init__(self, pars=None, location=None, data=None, contra_mod='mid', edu_name='edu', **kwargs):
        self.edu_name = edu_name
        super().__init__(pars=pars, location=location, data=data, contra_mod=contra_mod, **kwargs)
        return

    def prepare(self, pars):
        super().prepare(pars)
        if pars.get('age_spline') is None:
            raise ValueError('StandardChoice needs age_spline: give a location or data')
        return

    def age_term(self, event, ever):
        """
        The age terms of the logit (the spline, and its interaction with ever use), by single year of age from 0 to 49
        (ages are clipped to 15-49, as in Starsim 3 FPsim): a parameter for each decision and ever use
        """
        if not hasattr(self, '_age_terms'):
            self._age_terms = {}
        key = f'age_use_{event}_{ever}'
        if key not in self._age_terms:
            p = self._data['contra_use_pars'][dict(pp0=0, pp1=1, pp6=2)[event]]
            ages = np.arange(fpd.max_age_preg)
            knots = self._data['age_spline'].loc[np.clip(ages, fpd.min_age, fpd.max_age_preg - 1), ['knot_1', 'knot_2', 'knot_3']].values
            coefs = np.asarray(p.age_factors, dtype=float) + ever*np.asarray(p.age_ever_user_factors, dtype=float)
            self._age_terms[key] = ss.par(ss.by(dict(age=list(ages), value=knots @ coefs)), name=key, doc=f'Age terms of the logit of the probability of use ({event}, ever user: {bool(ever)})')
        return self._age_terms[key]

    def prob_use(self, event, ever, trend, intercept):
        """ Starsim 3 StandardChoice.get_prob_use, as an expression; `ever` is known from the state the decision is made in """
        p = self._data['contra_use_pars'][dict(pp0=0, pp1=1, pp6=2)[event]]
        att = ss.state(f'{self.edu_name}.attainment')
        logit = (p.intercept + ever*p.ever_used_contra
                 + p.urban*ss.ppl.urban + p.wealthquintile*ss.ppl.wealthquintile + p.parity*ss.state(f'{self.fp_name}.parity')
                 + self.age_term(event, ever)
                 + p.edu_factors[0]*((att > 1) & (att <= 6)) + p.edu_factors[1]*(att > 6)
                 + trend + intercept)
        return 1/(1 + np.exp(-logit))
