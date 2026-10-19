"""
Contraceptive methods and method choice, as a declared module

Each method is a state. A woman leaves it at the end of her duration of use (drawn on entry, by method and age)
and decides then, with her age at that moment, whether to keep using contraception and which method: one
transition per method with several destinations. Pregnancy pauses contraception, and the decisions after a
birth (at 1 and 6 months) and after a loss are transitions triggered by the pregnancy module's events.
"""

# %% Imports
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
        self._data = pars
        n = self.n_methods
        if pars['method_weights'] is None:
            pars['method_weights'] = np.ones(n)
        self.prepare(pars)
        states, trs, graph_pars = self.make_graph(pars)
        effs = {k: v for k, v in graph_pars.items() if k.startswith('eff_')} # Read by the FP module, not by a transition
        super().__init__(*states, **trs, **effs, name=name, **kwargs)
        self.stratify(age=age_edges)
        self.define_pars(**{k: v for k, v in pars.items() if k not in self.pars and k not in graph_pars})
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

        def split(event, src, ever, rest, base=None):
            """ [rest, method 1, ...]: not using w.p. 1 - p_use, else each method by its weighted share """
            pu = p_use(event, ever)
            sh = base if base is not None else self.shares(event, src)
            R = {m: ss.by(age=list(sh[m])) for m in names}
            total = None
            for m in names:
                total = w[m]*R[m] if total is None else total + w[m]*R[m]
            return [rest, *meth], [1 - pu, *[pu*w[m]*R[m]/total for m in names]]

        fpn = self.fp_name
        age, female = ss.people.age, ss.people.female
        debuted = female & (age >= ss.conditions.Cond('array', fpn, 'fated_debut')) & (age < fpd.max_age_preg)
        pregnant = ss.state(f'{fpn}.P')
        trs = sc.objdict()
        dst, p = split('pp0', 'none', 0, S.never)
        trs.debut = ss.Transition(S.pre, dst, p=p, on=debuted) # Everyone starts in pre, so women past their debut decide at the start (as in Starsim 3 FPsim)
        for src in ['never', 'none'] + names:
            if src == 'btl':
                continue # Permanent
            ever = int(src != 'never')
            dst, p = split('pp0', 'none' if src == 'never' else src, ever, S.never if src == 'never' else S.none)
            method = self.methods['none'] if src in ['never', 'none'] else self.methods[src]
            trs[f'stop_{src}'] = ss.Transition(S[src], dst, p=p, dur=self.dur_use(method))
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
        out = np.zeros(len(self.sim.people.uid.raw), dtype=int)
        for m in self.method_names:
            out[self[m].raw] = self.methods[m].idx
        return out

    def as_arr(self, raw, cls):
        """ A per-agent array (by UID) as an ss.Arr, so it can be indexed like a state """
        return self.sim.people.age.asnew(raw, cls=cls)

    @property
    def method(self):
        """ The index of each agent's method (0 if none), as in Starsim 3 FPsim """
        return self.as_arr(self.method_raw(), ss.IntArr)

    @property
    def on_contra(self):
        return self.as_arr(self.method_raw() > 0, ss.BoolArr)

    @property
    def ever_used_contra(self):
        out = np.ones(len(self.sim.people.uid.raw), dtype=bool)
        for k in ['pre', 'never', 'preg0', 'pp0', 'wait0']:
            out &= ~self[k].raw
        return self.as_arr(out, ss.BoolArr)

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
    """ Starsim 3 FPsim's default (age, education, wealth, parity and prior use): not ported to Starsim 4 yet; use SimpleChoice """
    def __init__(self, *args, **kwargs):
        raise NotImplementedError('StandardChoice (education, wealth, urban and parity in the probability of use) is not ported to Starsim 4 yet: use fp.SimpleChoice')
