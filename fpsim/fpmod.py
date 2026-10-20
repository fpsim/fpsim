"""
The FP module: the declared ss.Pregnancy, with FPsim's conception, losses, infant mortality, and results
"""

# %% Imports
import numpy as np
import sciris as sc
import starsim as ss
import fpsim as fp
from . import defaults as fpd
from . import arrays as fpa

__all__ = ['FPmod', 'Lactation']

P, NP = ss.Pregnancy.P, ss.Pregnancy.NP


def _fp_event(name):
    """ An event the FP module (named "fp") records on the mother """
    return ss.conditions.Event('tr', ss.conditions.Named('fp', name))


class Lactation(ss.Lactation):
    """ ss.Lactation driven by the FP module's events (ss.Lactation listens to a module named "pregnancy") """
    # TODO(v4-rs): ss.Lactation should find the sim's pregnancy module by type, not by the name "pregnancy"
    start   = ss.Transition(ss.Lactation.N, [ss.Lactation.L, ss.stay], p=[ss.Lactation.p_breastfeed, None], on=_fp_event('live_birth'))
    restart = ss.Transition(ss.Lactation.L, [ss.Lactation.L, ss.Lactation.N], p=[ss.Lactation.p_breastfeed, None], on=_fp_event('live_birth'))
    stop    = ss.Transition(ss.Lactation.L, ss.Lactation.N, on=_fp_event('infant_death'))


class FPmod(ss.Pregnancy):
    """
    The FP module: pregnancy as declared states (`NP`, `P`), with FPsim's model of conception

    Conception is one transition (`conception`), whose hazard is FPsim's: fecundity by age, personal fecundity,
    nulliparity and calibrated exposure by age and parity, times (1 - efficacy) of the contraceptive method and of
    lactational amenorrhea, capped at 1 per year, converted to a chance per step, and times the chance of being
    sexually active (by age; by month postpartum; with the birth spacing preference; 0 before debut for
    nulliparous women). Activity and LAM are their expected values, so are not drawn per woman. Abortions are a
    branch at conception; miscarriages (before `loss_threshold`) and stillbirths are hazards by gestational age
    and maternal age; maternal and infant deaths are at delivery. Parity, births and the time since the last live
    birth are kept per woman; parity is set at the start from the location's distribution by age.

    Takes FPsim's parameters (see `fp.FPPars`), by the same names. Agents only: see `fp.FPCompartmental` for
    the compartmental form.

    Args:
        pars (dict): FP parameters to override defaults
        location (str): location name for loading data (e.g. 'kenya')
        data (dict): pre-loaded FP data; if None, loaded from location
        name (str): module name (default 'fp')
    """
    mc_hazard = ss.par(None, doc='Hazard of a miscarriage, by gestational and maternal age: see FPmod.miscarriage_rate()')
    sb_hazard = ss.par(None, doc='Hazard of a stillbirth, by gestational and maternal age and year: see FPmod.stillbirth_rate()')
    miscarriage = ss.Transition(P, NP, rate=mc_hazard)
    stillbirth  = ss.Transition(P, NP, rate=sb_hazard)

    def __init__(self, pars=None, location=None, data=None, name='fp', **kwargs):
        if data is None and location is not None:
            data = fp.get_dataloader(location).load_fp_data(return_data=True)
        allpars = fp.mergepars(fp.FPPars(), data, pars, kwargs)
        allpars = self.process_pars(allpars)
        v3 = ss.PregnancyPars()
        preg = {k: v for k, v in allpars.items() if k in v3}
        preg['p_abortion'] = float(allpars['abortion_prob'] or 0)
        super().__init__(pars=preg, name=name, lactation=False, fetus=False)
        own = {k: v for k, v in allpars.items() if k not in self.pars}
        self.define_pars(**own)
        self.code_pars = list(self.code_pars) + list(own) # Read by FPmod's code, not by the graph
        self.pars.update(mc_hazard=self.miscarriage_rate, sb_hazard=self.stillbirth_rate)
        P = self.pars
        db = P['debut_age']
        self.define_states(
            ss.FloatArr('fated_debut', default=ss.choice(a=db['ages'], p=db['probs']), label='Age of sexual debut'),
            ss.FloatArr('personal_fecundity', default=ss.uniform(low=P['fecundity_low'], high=P['fecundity_high']), label='Personal fecundity'),
            *[ss.FloatArr(k, default=0, label=k) for k in ['n_births', 'n_twinbirths', 'n_stillbirths', 'n_miscarriages', 'n_abortions']],
            ss.FloatArr('first_birth_age', label='Age at first birth'),
            *[ss.FloatArr(f'ti_{k}', label=f'Time of the last {k}') for k in ['abortion', 'miscarriage', 'stillbirth']],
            ss.FloatArr('ti_last_interval', label='Interval between the last two live births (steps)'),
            *[fpa.TwoDimensionalArr(k, ncols=fpd.max_parity) for k in ['birth_ages', 'stillborn_ages', 'miscarriage_ages', 'abortion_ages']], # Ages at each event, as in Starsim 3
        )
        self._p_inf_mort = ss.random() # Infant deaths at delivery
        self._init_parity = ss.random() # Parity at the start
        self.method_mix = None
        self.parity_dist = None
        return

    @staticmethod
    def process_pars(pars):
        """ Starsim 3 FPsim's parameters as the declared module takes them """
        pars = sc.dcp(pars)
        dp = pars['dur_pregnancy']
        if isinstance(dp, ss.choice) and not isinstance(np.asarray(dp.pars.a).flat[0], ss.dur): # Bare numbers are weeks
            pars['dur_pregnancy'] = ss.choice(a=ss.weeks(np.asarray(dp.pars.a, dtype=float)), p=dp.pars.p)
        bf = pars['dur_breastfeed']
        if isinstance(bf, ss.lognorm_ex) and sc.isnumber(bf.pars.mean): # Bare numbers are months (Starsim 3: steps of a month)
            pars['dur_breastfeed'] = ss.lognorm_ex(mean=ss.months(bf.pars.mean), std=ss.months(bf.pars.std))
        twins = pars.get('twins_prob')
        if twins is not None:
            pars['embryos_per_pregnancy'] = ss.choice(a=np.array([1, 2]), p=np.array([1 - twins, twins]))
        for key in ['exposure_age', 'exposure_parity']: # Calibration points to one value per age or parity
            val = pars[key]
            if isinstance(val, dict):
                val = np.array([v for v in val.values()])
            val = np.asarray(val, dtype=float)
            if val.ndim == 2:
                pars[key] = fp.data2interp(val, fpd.spline_preg_ages if key == 'exposure_age' else fpd.spline_parities)
        dpp = pars['dur_postpartum']
        if sc.isnumber(dpp):
            pars['dur_postpartum'] = ss.months(dpp)
        pars.pop('fecundity', None) # Made from fecundity_low and fecundity_high
        return pars

    def init_pre(self, sim):
        super().init_pre(sim)
        if not any(isinstance(m, ss.Lactation) for m in sim.pars.demographics.values()):
            pass # Lactation is added by fp.Sim; without it there is no LAM
        return

    def init_post(self):
        super().init_post()
        self.init_parity()
        return

    def init_parity(self):
        """ Parity at the start, by age group, from the location's DHS distribution (Starsim 3 FPsim started everyone at 0) """
        ap = self.pars.get('init_parity')
        if ap is None:
            return
        ppl = self.sim.people
        women = (ppl.female & ppl.alive).uids
        age = ppl.age[women]
        rvs = self._init_parity.rvs(women)
        for lo, hi, probs in ap:
            these = (age >= lo) & (age < hi)
            cdf = np.cumsum(probs)/np.sum(probs)
            self.parity[women[these]] = np.searchsorted(cdf, rvs[these], side='right').astype(float)
        return

    # %% Lookups
    @property
    def lactation(self):
        return next((m for m in self.sim.demographics() if isinstance(m, ss.Lactation)), None) if self.sim is not None else None

    @property
    def contraception(self):
        return self.sim.connectors.get('contraception') if self.sim is not None else None

    def int_age(self):
        """ Each agent's age in whole years, clipped to the tables (0 to max_age) """
        return np.clip(np.nan_to_num(self.sim.people.age.raw), 0, int(self.pars.max_age)).astype(int)

    def months_since_birth(self):
        """ Whole months since each agent's last live birth (NaN if none) """
        out = np.floor((self.ti - self.ti_live_birth.raw)*self.t.dt.months + 1e-9)
        out[self.just_delivered()] = 0
        return out

    def just_delivered(self):
        """
        Women who delivered this step but whose live birth is not recorded yet: the hazards drawn on entering NP come
        before the children are made (TODO(v4-rs): live_birth, and so ti_live_birth, is recorded after the entry
        hazards of mothers back in NP, so conception would read the previous birth)
        """
        out = np.zeros(len(self.sim.people.uid.raw), dtype=bool)
        if self.engine is not None and getattr(self.engine, '_started', False):
            uids = self.delivery.uids
            out[uids[self.ti_live_birth.raw[uids] != self.ti]] = True
        return out

    def year_value(self, key):
        """ The value of a yearly series parameter in the current year (nearest year, as in Starsim 3 FPsim) """
        d = self.pars[key]
        return d['probs'][sc.findnearest(d['year'], self.t.now('year'))]

    # %% Rates
    def p_sexually_active(self):
        """ The chance each woman is sexually active this step (Starsim 3 FPmod.check_sexually_active) """
        pars = self.pars
        ppl = self.sim.people
        ia = self.int_age()
        parity = self.parity.raw
        msb = self.months_since_birth()
        has_birth = np.isfinite(msb)
        msb0 = np.where(has_birth, msb, 0).astype(int)
        sa = np.asarray(pars['sexual_activity'])[ia]
        sa_pp = np.asarray(pars['sexual_activity_pp']['percent_active'])
        pp = has_birth & (msb0 <= pars['dur_postpartum'].months)
        pref = pars['spacing_pref']
        spacing = np.asarray(pref['preference'])[np.minimum(msb0//int(pref['interval']), int(pref['n_bins']) - 1)]
        parous = np.where(pp, sa_pp[np.minimum(msb0, len(sa_pp) - 1)], sa)*spacing
        debuted = ppl.age.raw >= self.fated_debut.raw
        out = np.where(parity > 0, np.where(has_birth, parous, sa), np.where(debuted & ~pp, sa, 0))
        return out

    def p_lam(self):
        """ The chance each woman has lactational amenorrhea this step: breastfeeding, up to max_lam_dur months after a birth """
        lact = self.lactation
        n = len(self.sim.people.uid.raw)
        if lact is None:
            return np.zeros(n)
        pars = self.pars
        msb = self.months_since_birth()
        rate = np.asarray(pars['lactational_amenorrhea']['rate'])
        bf = lact.L.raw | self.just_delivered() # Lactation starts on the live birth (see just_delivered())
        on = bf & np.isfinite(msb) & (msb <= pars['max_lam_dur']) & (msb < len(rate))
        return np.where(on, rate[np.where(on, msb, 0).astype(int)], 0)

    def conception_rate(self, sim):
        """
        The hazard of conception per year for each agent (Starsim 3 FPmod.make_p_conceive, with sexual activity
        and LAM as their expected values): the annual probability is capped at 1 before it is converted
        """
        pars = self.pars
        ppl = sim.people
        age = ppl.age.raw
        ia = self.int_age()
        parity = self.parity.raw
        fecund = ppl.female.raw & ppl.alive.raw & (age >= pars.min_age) & (age < pars.max_age) & ~self.infertile.raw
        cm = self.contraception
        eff = cm.eff_array() if cm is not None else 0
        raw = np.asarray(pars['age_fecundity'])[ia]*self.personal_fecundity.raw*(1 - eff)
        raw = raw*np.where(parity == 0, np.asarray(pars['fecundity_ratio_nullip'])[ia], 1)
        raw = raw*pars['exposure_factor']*np.asarray(pars['exposure_age'])[ia]*np.asarray(pars['exposure_parity'])[np.clip(np.nan_to_num(parity), 0, fpd.max_parity).astype(int)]
        dt = self.t.dt_year
        pm = lambda x: 1 - (1 - np.minimum(x, 1))**dt
        lam = self.p_lam()
        active = self.p_sexually_active()
        p = np.where(fecund, active, 0)*((1 - lam)*pm(raw) + lam*pm(raw*(1 - pars['LAM_efficacy'])))
        p = np.where(fecund, np.minimum(p, 1 - 1e-12), 0)
        return ss.peryear(-np.log1p(-p)/dt)

    @property
    def loss_threshold_weeks(self):
        lt = self.pars.loss_threshold
        lt = lt.raw if isinstance(lt, ss.par) else lt
        return lt.weeks if isinstance(lt, ss.dur) else float(lt)

    def gestation_steps(self):
        """ Whole steps since conception, for each agent (NaN if not pregnant) """
        return np.where(self.P.raw, self.ti - self.ti_P.raw, np.nan)

    def miscarriage_rate(self, sim):
        """
        Starsim 3 FPmod.progress_pregnancies: the first-trimester risk by maternal age, spread over the steps to
        the end of the first trimester; then the stillbirth risk, labelled a miscarriage before loss_threshold
        """
        return self._loss_rate(sim, stillbirth=False)

    def stillbirth_rate(self, sim):
        """ The stillbirth risk by maternal age and year, spread over the steps from the end of the first trimester to 40 weeks, from loss_threshold """
        return self._loss_rate(sim, stillbirth=True)

    def _loss_rate(self, sim, stillbirth):
        pars = self.pars
        dtw = self.t.dt.weeks
        n1 = round(pars.trimesters[0].weeks/dtw)
        n2 = round((40 - pars.trimesters[0].weeks)/dtw)
        k = self.gestation_steps()
        ga = k*dtw # Gestational age at the start of the step, in weeks
        tri1 = (k >= 1) & (k <= n1)
        later = k > n1
        ia = self.int_age()
        mc = np.asarray(pars['miscarriage_rates'])[ia]
        sr = pars['stillbirth_rate']
        ages = np.asarray(sr['ages'])
        age = sim.people.age.raw
        ind = np.searchsorted(ages, age, side='left') # Nearest age, as in Starsim 3 FPmod._get_stillbirth_prob
        lo, hi = ages[np.maximum(ind - 1, 0)], ages[np.minimum(ind, len(ages) - 1)]
        ind = np.where((ind == len(ages)) | (np.abs(age - lo) < np.abs(age - hi)), ind - 1, ind)
        sb = self.year_value('stillbirth_rate')*np.asarray(sr['age_probs'])[np.clip(ind, 0, len(ages) - 1)]
        h1 = -np.log1p(-np.minimum(mc, 1 - 1e-9))/n1
        h2 = -np.log1p(-np.minimum(sb, 1 - 1e-9))/n2
        h = np.where(tri1, h1, np.where(later, h2, 0)) # Per step
        thr = ga >= self.loss_threshold_weeks
        h = np.where(thr if stillbirth else ~thr, h, 0)
        return ss.peryear(np.nan_to_num(h)/self.t.dt_year)

    # %% Deliveries
    def step(self):
        self.pars.p_maternal_death.set(p=self.year_value('maternal_mortality')*self.pars.maternal_mortality_factor)
        super().step()
        self.count_events()
        return

    def count_events(self):
        """ Per-woman counts of this step's losses (ss.Pregnancy counts n_pregnancies) """
        ages = dict(abortion='abortion_ages', miscarriage='miscarriage_ages', stillbirth='stillborn_ages')
        for tr, key in [('abortion', 'n_abortions'), ('miscarriage', 'n_miscarriages'), ('stillbirth', 'n_stillbirths')]:
            uids = self.engine.fresh(f'count_{key}', tr)
            if len(uids):
                self.record_age(ages[tr], uids, self[key].raw[uids])
                self[key][uids] += 1
                self[f'ti_{tr}'][uids] = self.ti
                if tr == 'stillbirth':
                    self.parity[uids] += 1 # Starsim 3 FPsim: stillbirths count in parity
        return

    def make_children(self, tr, parents, ti):
        """ The children of a delivery (see ss.Pregnancy), then twins in parity, birth ages and intervals, and infant deaths """
        ppl = self.sim.people
        prev = self.ti_live_birth.raw[parents].copy()
        new = super().make_children(tr, parents, ti)
        counts = np.bincount(ppl.parent.raw[new].astype(int), minlength=len(ppl.uid.raw))[parents]
        twins = parents[counts > 1]
        self.parity[twins] += counts[counts > 1] - 1
        self.record_age('birth_ages', parents, self.n_births.raw[parents])
        self.n_births[parents] += 1
        self.n_twinbirths[twins] += 1
        first = parents[np.isnan(self.first_birth_age.raw[parents])]
        self.first_birth_age[first] = ppl.age.raw[first]
        self.ti_last_interval[parents] = ti - prev
        self._short = np.count_nonzero((ti - prev)*self.t.dt_year < self.pars.short_int.years)
        self._n_live = len(parents)

        # Infant mortality (Starsim 3 FPmod.check_infant_mortality): by year and the mother's age
        im = self.pars['infant_mortality']
        p = self.year_value('infant_mortality')*np.asarray(im['age_probs'])[sc.findnearest(im['ages'], ppl.age.raw[parents])]
        moms = parents[self._p_inf_mort.rvs(parents) < p]
        kids = new[np.isin(ppl.parent.raw[new], moms)]
        if len(kids):
            ppl.request_death(kids)
        self._infant_deaths = len(kids)
        return new

    def record_age(self, key, uids, counts):
        """ Each woman's age at her nth event (the nth column), up to max_parity """
        ok = counts < fpd.max_parity
        uids, counts = uids[ok], counts[ok].astype(int)
        for c in np.unique(counts):
            these = uids[counts == c]
            self[key][these, c] = self.sim.people.age.raw[these]
        return

    # %% Results
    def init_results(self):
        super().init_results()
        kw = dict(dtype=float, scale=True, auto_plot=False)
        nkw = dict(dtype=float, scale=False)
        results = [
            ss.Result('infant_deaths', label='Infant deaths', summarize_by='sum', **kw),
            ss.Result('imr', label='Infant mortality rate', summarize_by='mean', **nkw),
            ss.Result('method_failures', label='Pregnancies on a method', summarize_by='sum', **kw),
            ss.Result('short_intervals', label='Births after a short interval', summarize_by='sum', **kw),
            ss.Result('p_short_interval', label='Proportion of short interval births', summarize_by='mean', **nkw),
            ss.Result('total_births', label='Live births and stillbirths', summarize_by='sum', **kw),
            ss.Result('n_fecund', label='Women 15-49', summarize_by='mean', **kw),
        ]
        results += [ss.Result(f'cum_{key}', label=f'Cumulative {key}', dtype=float, scale=False, summarize_by='last') for key in fpd.event_counts + fpd.inherited_counts]
        self.define_results(*[r for r in results if r.name not in self.results])
        self.parity_bins = np.arange(fpd.max_parity + 2)
        self.parity_dist = np.zeros((len(self.parity_bins) - 1, self.t.npts))
        self._short = self._n_live = self._infant_deaths = 0
        return

    def update_results(self):
        """ Infant deaths, short intervals, women 15-49, the parity distribution, the method mix (TFR and ASFR: see ss.Pregnancy) """
        super().update_results()
        ti = self.ti
        ppl = self.sim.people
        res = self.results
        scale = self.sim._current_scale
        women = ppl.female.raw & ppl.alive.raw
        age = ppl.age.raw
        res.infant_deaths[ti] = self._infant_deaths*scale
        res.short_intervals[ti] = self._short*scale
        res.p_short_interval[ti] = sc.safedivide(self._short, self._n_live)
        self._short = self._n_live = self._infant_deaths = 0
        cm = self.contraception
        fecund = women & (age >= fpd.min_age) & (age < fpd.max_age_preg)
        res.n_fecund[ti] = fecund.sum()*scale
        counts = np.bincount(np.minimum(self.parity.raw[fecund], len(self.parity_bins) - 2).astype(int), minlength=len(self.parity_bins) - 1)
        self.parity_dist[:, ti] = counts/max(counts.sum(), 1)
        if cm is not None:
            self.method_mix = cm.method_mix
        return

    def finalize_results(self):
        super().finalize_results()
        res = self.results
        res.total_births[:] = res.births[:] + res.stillbirths[:]
        res.imr[:] = sc.safedivide(res.infant_deaths[:], res.births[:])*1e3
        cm = self.contraception
        if cm is not None and 'fail' in cm.results:
            res.method_failures[:] = cm.results.fail[:] # Continuing pregnancies of women on a method
        for key in fpd.event_counts + fpd.inherited_counts:
            if key in res:
                res[f'cum_{key}'][:] = np.cumsum(res[key][:])
        return

    # %% Starsim 3 FPsim names
    @property
    def method(self):
        return self.contraception.method

    @property
    def on_contra(self):
        return self.contraception.on_contra

    @property
    def ever_used_contra(self):
        return self.contraception.ever_used_contra

    @property
    def end_tri1_uids(self):
        """ Women in the last step of their first trimester (read by fp.Education) """
        ga = self.gestation.raw[self.P.uids] # Weeks
        t1 = self.pars.trimesters[0].weeks
        return self.P.uids[(ga <= t1) & (ga + self.t.dt.weeks > t1)]

    @property
    def postpartum(self):
        """ Women within dur_postpartum of a live birth, not pregnant """
        msb = self.months_since_birth()
        pp = ~self.P.raw & np.isfinite(msb) & (msb <= self.pars.dur_postpartum.months)
        return self.sim.people.age.asnew(pp, cls=ss.BoolArr)
