"""
The compartmental form of FPsim: one declared module whose dimensions are age, parity, and contraceptive method,
so it runs as agents and as an ODE. It is a different model from fp.FPmod + fp.SimpleChoice, written so a
compartmental solver can take it: time since a birth is a chain of postpartum states, method durations are
exponential (matched to 12-month continuation), personal fecundity and activity are their means, and births are
an inflow of daughters at the expected delivery rate.
"""
import numpy as np
import sciris as sc
import starsim as ss
import fpsim as fp
from . import defaults as fpd
from . import methods as fpm

__all__ = ['FPCompartmental', 'make_compartmental_sim']

STATES = ['NP', 'INF', 'P1', 'P2', 'PA', 'PB', 'PC', 'PD', 'PE'] # Not pregnant, infertile, trimester 1, 2-3, postpartum 0-1, 1-6, 6-18, 18-39, 39-51 months
ROUTES = ['NP', 'PA', 'PB', 'PC', 'PD', 'PE'] # The states women conceive from
NOMETHOD = ['pre', 'never', 'none', 'pause0', 'pause1', 'wait0', 'wait1'] # Levels of the method dimension that are not a method (pause, wait: by ever use)


def binned(arr, edges, top=None):
    """ The mean of a single-year-of-age array over each age group (the last group is edges[-1] to top) """
    arr = np.asarray(arr, dtype=float)
    top = top or len(arr)
    hi = list(edges[1:]) + [top]
    return np.array([arr[lo:min(h, len(arr))].mean() if lo < len(arr) else arr[-1] for lo, h in zip(edges, hi)])


def p_month(raw):
    """ Starsim 3 FPmod.make_p_conceive: an annual probability, capped at 1, as a monthly one """
    return 1 - (1 - np.minimum(raw, 1))**(1/12)


class FPCompartmental(ss.Disease):
    """
    FPsim as one declared module of women, with age, parity and method as dimensions, for agents and compartments

    Args:
        location (str): the location whose data to use
        edges (list): age group edges; must contain the method-choice edges in use (18, 20, 25, 35) and 15 and 50
        parity (list): parity levels (the last is that many or more)
        pars (dict): FP parameters to override the location's
        contra_pars (dict): contraception parameters to override the location's

    **Example**::

        sim = fp.make_compartmental_sim('kenya', solver='ode', total_pop=10_000)
        sim.run()
    """
    def __init__(self, location='kenya', edges=None, parity=None, pars=None, contra_pars=None, name='fp', **kwargs):
        edges = list(edges if edges is not None else fpm.age_edges)
        parity = list(parity if parity is not None else ['0', '1', '2', '3', '4+'])
        self.edges, self.parity_levels = edges, parity
        P, cm = self.load(location, pars, contra_pars)
        self.data = sc.objdict(P=P)
        self.methods = cm.methods
        names = cm.method_names
        levels = NOMETHOD + names
        self.levels = levels
        mb = fpm.method_bins(edges)
        yr = lambda per_month: 12*np.asarray(per_month) # Rates are bare numbers per year inside ss.by

        # Conception: one transition per (route, method level), its rate the method's; move= starts the pause
        F = self.fertility(P, cm, edges, parity)
        S = sc.objdict({s: ss.State(s) for s in STATES})
        pab = float(P['abortion_prob'])
        tr = sc.objdict()
        for route in ROUTES:
            for lev in levels:
                if lev == 'pre' or lev.startswith('pause') != (route == 'PA') or (lev.startswith('wait') and route != 'PB'):
                    continue # Nobody there: before debut, pause levels only in the first month postpartum, wait levels only in months 1-6
                to = 'pause0' if lev in ['never', 'pause0', 'wait0'] else 'pause1'
                tr[f'c_{route}_{lev}'] = ss.Transition(S[route], S.P1, rate=ss.by(yr(F.haz[(route, lev)])*(1 - pab), dims=['age', 'parity']), move=dict(method=f'{lev} -> {to}'))

        # Pregnancy and its outcomes; after a loss, back to (ever-)non-use
        mc = binned(P['miscarriage_rates'][:51], edges)
        sr = P['stillbirth_rate']
        sb_age = np.array([sr['age_probs'][np.argmin(np.abs(np.asarray(sr['ages']) - lo))] for lo in edges])
        dur = P['dur_pregnancy']
        mean_preg = ss.weeks(float(np.sum(np.asarray(dur.pars.a.weeks)*np.asarray(dur.pars.p))))
        mean_p2 = ss.weeks(mean_preg.weeks - 13)
        tr.tri1 = ss.Transition(S.P1, S.P2, dur=ss.constant(ss.weeks(13)))
        for k, (pz, back) in enumerate([('pause0', 'never'), ('pause1', 'none')]):
            tr[f'miscarriage_{k}'] = ss.Transition(S.P1, S.NP, rate=ss.by(age=-np.log1p(-np.minimum(mc, .999))/ss.weeks(13).years), move=dict(method=f'{pz} -> {back}'))
            tr[f'stillbirth_{k}'] = ss.Transition(S.P2, S.NP, rate=ss.by(age=-np.log1p(-sr['probs'][0]*sb_age)/mean_p2.years), move=dict(method=f'{pz} -> {back}'))
        tr.maternal = ss.Transition(S.P2, ss.null, rate=ss.peryear(-np.log1p(-P['maternal_mortality']['probs'][0])/mean_p2.years))
        twins = float(P['twins_prob'])
        tr.deliver = ss.Transition(S.P2, S.PA, rate=ss.peryear((1 - twins)/mean_p2.years), move=dict(parity='+1'))
        tr.deliver_twins = ss.Transition(S.P2, S.PA, rate=ss.peryear(twins/mean_p2.years), move=dict(parity='+2'))

        # Postpartum: the 1- and 6-month decisions are the transitions out of PA and PB, split by destination as competing hazards
        pu = {k: {e: 1/(1 + np.exp(-cm.logit_use(k, e, edges))) for e in [0, 1]} for k in ['pp0', 'pp1', 'pp6']}
        share = lambda event, src: self.weighted(cm, cm.shares(event, src, edges))
        rows1, rows6 = share('pp1', None), share('pp6', 'none')
        for e in [0, 1]:
            tr[f'pp1_wait{e}'] = ss.Transition(S.PA, S.PB, rate=ss.by(age=yr(1 - pu['pp1'][e])), move=dict(method=f'pause{e} -> wait{e}'))
            tr[f'pp6_none{e}'] = ss.Transition(S.PB, S.PC, rate=ss.by(age=yr((1 - pu['pp6'][e])/5)), move=dict(method=f'wait{e} -> {"none" if e else "never"}'))
            for m in names:
                tr[f'pp1_{m}{e}'] = ss.Transition(S.PA, S.PB, rate=ss.by(age=yr(pu['pp1'][e]*rows1[m])), move=dict(method=f'pause{e} -> {m}'))
                tr[f'pp6_{m}{e}'] = ss.Transition(S.PB, S.PC, rate=ss.by(age=yr(pu['pp6'][e]*rows6[m]/5)), move=dict(method=f'wait{e} -> {m}'))
        for m in names: # Women on a method from 1 month postpartum: out of PB at the same rate as the decisions (one dwell out of a state)
            tr[f'pp6_{m}'] = ss.Transition(S.PB, S.PC, rate=ss.peryear(12/5), move=dict(method=f'{m} -> {m}'))
        tr.pp18 = ss.Transition(S.PC, S.PD, rate=ss.peryear(1)) # Exponential: a conveyor of 12-21 monthly levels would multiply the compartments
        tr.pp39 = ss.Transition(S.PD, S.PE, rate=ss.peryear(12/21))
        tr.pp51 = ss.Transition(S.PE, S.NP, rate=ss.peryear(1))
        im = P['infant_mortality']
        imr = im['probs'][np.searchsorted(im['year'], 2000)]
        p_inf = P['p_infertile'].pars.p if isinstance(P['p_infertile'], ss.bernoulli) else float(P['p_infertile'])
        tr.births = ss.Transition(ss.null, (1 - p_inf)*S.NP + p_inf*S.INF, rate=ss.peryear(0.5*(1 + twins)*(1 - imr)/mean_p2.years)*S.P2.n) # Surviving daughters
        tr.deaths = ss.death(rate=ss.by(age=-np.log1p(-binned(self.mortality, edges, top=100))))

        # Switching: Starsim 3's decision at the end of a dwell, as competing hazards p*h, with h matched to 12-month continuation
        nonpreg = ['NP', 'INF', 'PA', 'PB', 'PC', 'PD', 'PE']
        trans = sc.objdict()
        h_debut = self.debut_hazard(P, edges)
        rows0 = share('pp0', 'none')
        trans.debut_never = ss.transfer(method='pre -> never', rate=ss.by(age=h_debut*(1 - pu['pp0'][0])), states=nonpreg)
        for m in names:
            trans[f'debut_{m}'] = ss.transfer(method=f'pre -> {m}', rate=ss.by(age=h_debut*pu['pp0'][0]*rows0[m]), states=nonpreg)
        self.exit_rates = sc.objdict()
        for frm in ['never', 'none'] + [m for m in names if m != 'btl']:
            key = 'none' if frm == 'never' else frm
            e = 0 if frm == 'never' else 1
            h = self.exit_rate(cm, key, edges)
            self.exit_rates[frm] = h
            rows = share('pp0', key)
            if frm not in ['never', 'none']:
                trans[f'{frm}_to_none'] = ss.transfer(method=f'{frm} -> none', rate=ss.by(age=(1 - pu['pp0'][e])*h), states=nonpreg)
            for m in names:
                if m != frm:
                    trans[f'{frm}_to_{m}'] = ss.transfer(method=f'{frm} -> {m}', rate=ss.by(age=pu['pp0'][e]*rows[m]*h), states=nonpreg)

        # Initial state: women past their debut decide as never-users (Starsim 3 FPsim at the start), the rest before debut
        Fd = self.debut_cdf(P, edges)
        split = np.zeros((len(edges), len(levels)))
        split[:, 0] = 1 - Fd
        split[:, 1] = Fd*(1 - pu['pp0'][0])
        split[:, len(NOMETHOD):] = (Fd*pu['pp0'][0])[:, None]*np.array([rows0[m] for m in names]).T
        split[:, 0] += 1 - split.sum(1)
        pi = self.init_parity(P, edges, parity)

        super().__init__(*S.values(), **tr, init=dict(INF=p_inf), name=name, **kwargs)
        self.stratify(parity=parity) # Its split, by age, is in the joint split below (one split can cover several dimensions)
        joint = pi[:, :, None]*split[:, None, :] # (age, parity, method): parity and method independent given age
        self.stratify(method=levels, split=ss.by(joint, dims=['age', 'parity', 'method']), **trans)
        self.pab = pab
        return

    # %% Data
    def load(self, location, pars, contra_pars):
        """ The FP module's parameters (as fp.Sim gives them) and a SimpleChoice for the method tables """
        dl = fpd.get_dataloader(location)
        data = dl.load(contra_mod='simple')
        calib = fpd.get_calib_pars(location, verbose=0) or {}
        for key in fp.sim.uncalibrated_keys:
            calib.pop(key, None)
        fpp = fp.mergepars(fp.FPPars(), data['fp'], {k: v for k, v in calib.items() if k in fp.FPPars()}, pars)
        P = fp.FPmod.process_pars(fpp)
        cm = fp.SimpleChoice(pars=fp.mergepars(data['contra'], {k: v for k, v in calib.items() if k in fpm.ContraPars()}, contra_pars))
        self.mortality = data['deaths']['age_mortality']['f_spline']
        self.pyramid = data['people']['age_pyramid']
        return P, cm

    @staticmethod
    def weighted(cm, shares):
        w = {m: float(cm.pars[f'weight_{m}'].raw) for m in cm.method_names}
        total = sum(w[m]*shares[m] for m in cm.method_names)
        return {m: w[m]*shares[m]/total for m in cm.method_names}

    def fertility(self, P, cm, edges, parity):
        """
        Monthly hazards of conception by (age, parity), for each route (reproductive state) and method level. Efficacy
        multiplies the annual probability before it is capped and converted, as in Starsim 3 FPsim
        """
        fec = binned(P['age_fecundity'][:51], edges)*0.5*(P['fecundity_low'] + P['fecundity_high'])*P['exposure_factor']*binned(P['exposure_age'][:51], edges)
        nullip = binned(P['fecundity_ratio_nullip'][:51], edges)
        k = len(parity) - 1
        ep = np.asarray(P['exposure_parity'])
        exp_par = np.array([ep[i] for i in range(k)] + [ep[k:k+4].mean()])
        raw = fec[:, None]*exp_par[None, :]*np.where(np.arange(len(parity)) == 0, nullip[:, None], 1) # Annual probability, (age, parity)
        sa = binned(P['sexual_activity'][:51], edges)
        sa_pp = np.asarray(P['sexual_activity_pp']['percent_active'])
        lam = np.asarray(P['lactational_amenorrhea']['rate'])*P['LAM_efficacy']
        sp = P['spacing_pref']
        space = lambda m0, m1: np.asarray(sp['preference'])[np.minimum(np.arange(m0, m1)//int(sp['interval']), int(sp['n_bins']) - 1)].mean()
        debuted = self.debut_cdf(P, edges)
        never_act = np.where(debuted > 0, sa, 0) # Never users have debuted
        routes = sc.objdict( # Activity, LAM effect, spacing weight; the postpartum bands are months since delivery
            NP = (np.where(np.arange(len(parity)) == 0, 1, space(51, 52))[None, :]*sa[:, None], 0, 1),
            PA = (sa_pp[0], lam[0], 1),
            PB = (sa_pp[1:6].mean(), lam[1:6].mean(), 1),
            PC = (sa_pp[6:18].mean(), 0, 1),
            PD = (sa[:, None], 0, space(18, 39)),
            PE = (sa[:, None], 0, space(39, 51)),
        )
        eff = {m: 0.0 for m in NOMETHOD}
        eff.update({m: float(cm.methods[m].efficacy) for m in cm.method_names})
        haz = {}
        for name, (act, lam_eff, spacing) in routes.items():
            a = np.broadcast_to(np.asarray(act, dtype=float)*spacing, raw.shape) if np.ndim(act) else np.full(raw.shape, act*spacing)
            for lev, e in eff.items():
                haz[(name, lev)] = -np.log1p(-np.minimum(a*p_month(raw*(1 - lam_eff)*(1 - e)), 1 - 1e-9))
        return sc.objdict(haz=haz)

    def exit_rate(self, cm, m, edges):
        """
        An exponential hazard of leaving a method (per year), by age group, matched to its 12-month continuation:
        matching the mean fails for heavy tails (TODO(v4-rs): a 2-phase hyperexponential when the CV is above 1)
        """
        method = cm.methods[m]
        surv = fpm.dur_use_survival(cm.pars['dur_use_df'], method, 12, edges)
        return -np.log(np.maximum(surv, 1e-9))

    @staticmethod
    def debut_cdf(P, edges):
        """ Share of women past their fated debut age, at the middle of each age group """
        db = P['debut_age']
        cdf = np.cumsum(db['probs'])/np.sum(db['probs'])
        mids = [(lo + hi)/2 for lo, hi in zip(edges, list(edges[1:]) + [edges[-1] + 10])]
        return np.interp(mids, np.asarray(db['ages']) + 0.5, cdf, left=0, right=1)

    @staticmethod
    def debut_hazard(P, edges):
        """ The hazard of debut (per year) in each age group, from the fated-debut distribution (exact for a cohort) """
        db = P['debut_age']
        surv = lambda a: 1 - np.interp(a, np.asarray(db['ages']) + 1, np.cumsum(db['probs'])/np.sum(db['probs']), left=0, right=1)
        hi = list(edges[1:]) + [edges[-1] + 10]
        return np.array([np.log(max(surv(lo), 1e-9)/max(surv(h), 1e-9))/(h - lo) if surv(h) > 1e-9 else 5.0 for lo, h in zip(edges, hi)])

    def init_parity(self, P, edges, parity):
        """ The parity split within each age group at the start, from the location's DHS table """
        out = np.zeros((len(edges), len(parity)))
        out[:, 0] = 1
        ap = P.get('init_parity')
        if ap is None:
            return out
        for i, lo in enumerate(edges):
            for a0, a1, probs in ap:
                if a0 <= lo + 1e-9 < a1 or (lo >= 50 and a1 >= 200):
                    p = np.zeros(len(parity))
                    for j, v in enumerate(probs):
                        p[min(j, len(parity) - 1)] += v
                    out[i] = p
                    break
        return out

    def age_split(self, edges):
        """ Women by age group, from the location's age pyramid (5-year bins) """
        pyr = np.asarray(self.pyramid, dtype=float)
        single = np.repeat(pyr[:, 2]/5, 5)
        split = np.array([single[lo:hi].sum() for lo, hi in zip(edges, list(edges[1:]) + [len(single)])])
        return split/split.sum()

    # %% Results
    def summarize(self, sim):
        """ mCPR (women 15-49), TFR, and the method mix at the end, from the results by age and method, under any solver """
        r = sim.results[self.name]
        edges = self.edges
        rep = [i for i, lo in enumerate(edges) if 15 <= lo < 50]
        A = lambda res: np.asarray(res.by('age').values)
        method = lambda m: sum(A(r[f'n_{s}'].select(method=m)) for s in STATES)
        w = sum(A(r[f'n_{s}']) for s in STATES)
        modern = [m for m in self.methods if m != 'none' and self.methods[m].modern]
        mcpr = sum(method(m) for m in modern)[:, rep].sum(1)/np.maximum(w[:, rep].sum(1), 1e-9)
        deliv = A(r.deliver) + A(r.deliver_twins)
        widths = np.diff(list(edges) + [edges[-1] + 5])
        yv = np.asarray(sim.t.yearvec)
        out = sc.objdict(mcpr=mcpr, tfr=dict(), mix=dict())
        for y in np.unique(np.floor(yv[:-1])).astype(int):
            i0, i1 = np.searchsorted(yv, y), np.searchsorted(yv, y + 1)
            asfr = deliv[i0:i1].sum(0)/np.maximum(w[i0:i1].mean(0), 1e-9)
            out.tfr[y] = float((asfr[rep]*widths[rep]).sum())
        for m in self.methods:
            if m != 'none':
                out.mix[m] = float(method(m)[-1, rep].sum()/w[-1, rep].sum())
        out.parity = np.asarray(sum(np.asarray(r[f'n_{s}'].by('parity').values) for s in STATES))[-1]
        out.parity = out.parity/out.parity.sum()
        return out


def make_compartmental_sim(location='kenya', solver='ode', n_agents=5000, total_pop=None, edges=None, parity=None, start=2000, stop=2020, **kwargs):
    """ A sim of the compartmental form of FPsim (see `FPCompartmental`), as agents (`solver='abm'`) or compartments """
    fpc = FPCompartmental(location=location, edges=edges, parity=parity)
    kw = dict(start=start, stop=stop, dt=ss.months(1), unit='year', verbose=0, solver=solver, stratify=dict(age=fpc.edges, split=fpc.age_split(fpc.edges)))
    kw.update(dict(total_pop=total_pop or n_agents) if solver != 'abm' else dict(n_agents=n_agents))
    kw.update(kwargs)
    return ss.Sim(fpc, **kw)
