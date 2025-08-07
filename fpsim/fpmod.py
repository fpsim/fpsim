"""
Defines the FPmod class
"""

# %% Imports
import numpy as np  # Needed for a few things not provided by pl
import sciris as sc
import fpsim as fp
from . import utils as fpu
from . import defaults as fpd
import starsim as ss

# Specify all externally visible things this file defines
__all__ = ['FPmod']


# %% Define classes

class FPmod(ss.Module):
    """
    Class for storing and updating FP-related events
    """

    def __init__(self, pars=None, location=None, name='fp', **kwargs):
        super().__init__(name=name)
        default_pars = fp.FPPars()
        if location is not None:
            default_pars.update_location(location)  # Update location-specific parameters
        self.define_pars(**default_pars)
        self.update_pars(pars, **kwargs)

        # Define states

        return

    def init_vals(self, uids=None):
        # Parameters on sexual and reproductive history
        self.fertile[uids] = fpu.n_binomial(1 - fp_pars['primary_infertility'], len(uids))

        # Sexual activity
        # Default initialization for fated_debut; subnational debut initialized in subnational.py otherwise
        self.fated_debut[uids] = fp_pars['debut_age']['ages'][fpu.n_multinomial(fp_pars['debut_age']['probs'], len(uids))]
        fecund = self.female & (self.age < fp_pars['age_limit_fecundity'])
        self.check_sexually_active(uids[fecund[uids]])
        self.update_time_to_choose(uids)

        # Fecundity variation
        fv = [fp_pars['fecundity_var_low'], fp_pars['fecundity_var_high']]
        fac = (fv[1] - fv[0]) + fv[0]  # Stretch fecundity by a factor bounded by [f_var[0], f_var[1]]
        self.personal_fecundity[uids] = np.random.random(len(uids)) * fac
        return

    def init_results(self):
        """
        Initialize result storage. Most default results are either arrays or lists; these are
        all stored in defaults.py. Any other results with different formats can also be added here.
        """
        super().init_results()  # Initialize the base results

        scaling_kw = dict(dtype=int, scale=True)
        nonscaling_kw = dict(dtype=float, scale=False)

        # Add event counts - these are all integers, and are scaled by the number of agents
        # We compute new results for each event type, and also cumulative results
        for key in fpd.event_counts:
            self.results += ss.Result(key, label=key, **scaling_kw)
            self.results += ss.Result(f'cum_{key}', label=key, dtype=int, scale=False)  # TODO, check

        # Add people counts - these are all integers, and are scaled by the number of agents
        # However, for these we do not include cumulative totals
        for key in fpd.people_counts:
            self.results += ss.Result(key, label=key, **scaling_kw)

        for key in fpd.rate_results:
            self.results += ss.Result(key, label=key, **nonscaling_kw)

        for key in fpd.dict_annual_results:
            if key == 'method_usage':
                self.results[key] = ss.Results(module=self)
                for i, method in enumerate(self.sim.connectors.contraception.methods):
                    self.results[key] += ss.Result(method, label=method, **scaling_kw)

        # Store age-specific fertility rates
        self.results['asfr'] = ss.Results(module=self)  # ['asfr'] = {}
        for key in fpd.age_bin_map.keys():
            self.results.asfr += ss.Result(key, label=key, **nonscaling_kw)
            self.results += ss.Result(f"tfr_{key}", label=key, **nonscaling_kw)

        return

    def update_time_to_choose(self, uids=None):
        """
        Initialise the counter to determine when girls/women will have to first choose a method.
        """
        if uids is None:
            uids = self.alive.uids

        fecund = uids[(self.female[uids] == True) & (self.age[uids] < self.sim.fp_pars['age_limit_fecundity'])]

        time_to_debut = (self.fated_debut[fecund]-self.age[fecund])/self.sim.t.dt

        # If ti_contra is less than one timestep away, we want to also set it to 0 so floor time_to_debut.
        self.ti_contra[fecund] = np.maximum(np.floor(time_to_debut), 0)

        # Validation
        time_to_set_contra = self.ti_contra[fecund] == 0
        if not np.array_equal(((self.age[fecund] - self.fated_debut[fecund]) > -self.sim.t.dt), time_to_set_contra):
            errormsg = 'Should be choosing contraception for everyone past fated debut age.'
            raise ValueError(errormsg)
        return

    def update_mothers(self, uids):
        """
        Add link between newly added individuals and their mothers
        """
        if uids is None:
            uids = self.alive.uids

        for mother_index, postpartum in enumerate(uids[self.postpartum[uids]]):
            if postpartum and self.postpartum_dur[uids][mother_index] < 2:
                for child in self.children[uids][mother_index]:
                    self.mothers[uids][child] = mother_index
        return

    def decide_death_outcome(self, uids):
        """ Decide if person dies at a timestep """
        sim = self.sim
        timestep = sim.t.dt_year * fpd.mpy # timestep in months
        trend_val = sim.fp_pars['mortality_probs']['gen_trend']
        age_mort = sim.fp_pars['age_mortality']
        f_spline = age_mort['f_spline'] * trend_val
        m_spline = age_mort['m_spline'] * trend_val
        over_one = self.age[uids] >= 1
        female = uids[over_one & self.female[uids]]
        male = uids[over_one & self.male[uids]]
        f_ages = self.int_age(female)
        m_ages = self.int_age(male)

        f_mort_prob = fpu.annprob2ts(f_spline[f_ages], timestep)
        m_mort_prob = fpu.annprob2ts(m_spline[m_ages], timestep)

        self.binom.set(p=f_mort_prob)
        f_died = self.binom.filter(female)

        self.binom.set(p=m_mort_prob)
        m_died = self.binom.filter(male)

        for died in [f_died, m_died]:
            self.pregnant[died] = False,
            self.gestation[died] = False,
            self.sexually_active[died] = False,
            self.lactating[died] = False,
            self.postpartum[died] = False,
            self.lam[died] = False,
            self.breastfeed_dur[died] = 0,
            sim.results['deaths'][sim.ti] += len(died)
            self.request_death(died)
            self.step_die() # to match the order of ops from earlier FPsim version
            self.remove_dead()

        return

    def check_conception(self, uids):
        """
        Decide if person (female) becomes pregnant at a timestep.
        """
        if uids is None:
            uids = self.alive.uids

        active_uids = uids[(self.sexually_active[uids] & self.fertile[uids])]
        lam = self.lam[active_uids]
        lam_uids = active_uids[lam]
        nonlam = ~self.lam[active_uids]
        nonlam_uids = active_uids[nonlam]
        preg_probs = np.zeros(len(active_uids))  # Use full array

        # Find monthly probability of pregnancy based on fecundity and use of contraception including LAM - from data
        pars = self.sim.fp_pars  # Shorten
        preg_eval_lam = pars['age_fecundity'][self.int_age_clip(lam_uids)] * self.personal_fecundity[lam_uids]
        preg_eval_nonlam = pars['age_fecundity'][self.int_age_clip(nonlam_uids)] * self.personal_fecundity[nonlam_uids]

        # Get each woman's degree of protection against conception based on her contraception or LAM
        cm = self.sim.connectors.contraception
        eff_array = np.array([m.efficacy for m in cm.methods.values()])
        method_eff = eff_array[self.method[nonlam_uids].astype(int)]
        lam_eff = pars['LAM_efficacy']

        # Change to a monthly probability and set pregnancy probabilities
        lam_probs = fpu.annprob2ts((1 - lam_eff) * preg_eval_lam, self.sim.t.dt_year * fpd.mpy)
        nonlam_probs = fpu.annprob2ts((1 - method_eff) * preg_eval_nonlam, self.sim.t.dt_year * fpd.mpy)
        preg_probs[lam] = lam_probs
        preg_probs[nonlam] = nonlam_probs

        # Adjust for decreased likelihood of conception if nulliparous vs already gravid - from PRESTO data
        nullip = self.parity[active_uids] == 0
        nullip_uids = active_uids[nullip]
        preg_probs[nullip] *= pars['fecundity_ratio_nullip'][self.int_age_clip(nullip_uids)]

        # Adjust for probability of exposure to pregnancy episode at this timestep based on age and parity.
        # This encapsulates background factors and is experimental and tunable.
        preg_probs *= pars['exposure_factor']
        preg_probs *= pars['exposure_age'][self.int_age_clip(active_uids)]
        preg_probs *= pars['exposure_parity'][np.minimum(self.parity[active_uids], fpd.max_parity).astype(int)]

        # Use a single binomial trial to check for conception successes this month
        self.binom.set(p=preg_probs)
        conceived = self.binom.filter(active_uids)

        self.sim.results['pregnancies'][self.sim.ti] += len(conceived)  # track all pregnancies
        unintended = conceived[self.method[conceived] != 0]
        self.sim.results['method_failures'][self.sim.ti] += len(unintended)  # unintended pregnancies due to method failure

        # Check for abortion
        self.binom.set(p=pars['abortion_prob'])
        abort, preg = self.binom.filter(conceived, both=True)

        # Update states
        n_aborts = len(abort)
        self.sim.results['abortions'][self.sim.ti] = n_aborts
        if n_aborts:
            for abort_uid in abort:
                # put abortion age in first nan slot
                abortion_age_index = np.where(np.isnan(self.abortion_ages[abort_uid]))[0][0]
                self.abortion_ages[abort_uid, abortion_age_index] = self.age[abort_uid]
            self.postpartum[abort] = False
            self.abortion[abort] += 1  # Add 1 to number of abortions agent has had
            self.postpartum_dur[abort] = 0

        # Make selected agents pregnant
        self.make_pregnant(preg)

        return

    def make_pregnant(self, uids):
        """
        Update the selected agents to be pregnant. This also sets their method to no contraception.
        """
        pregdur = [self.sim.fp_pars['preg_dur_low'], self.sim.fp_pars['preg_dur_high']]
        self.pregnant[uids] = True
        self.gestation[uids] = 1  # Start the counter at 1
        self.preg_dur[uids] = np.random.randint(pregdur[0], pregdur[1] + 1, size=len(uids))  # Duration of this pregnancy
        self.postpartum[uids] = False
        self.postpartum_dur[uids] = 0
        self.reset_breastfeeding(uids)  # Stop lactating if becoming pregnant
        self.on_contra[uids] = False  # Not using contraception during pregnancy
        self.method[uids] = 0  # Method zero due to non-use
        return

    def check_lam(self, uids):
        """
        Check to see if postpartum agent meets criteria for
        Lactation amenorrhea method (LAM) LAM in this time step
        """
        max_lam_dur = self.sim.fp_pars['max_lam_dur']
        lam_candidates = uids[(self.postpartum[uids]) * (self.postpartum_dur[uids] <= max_lam_dur)]
        if len(lam_candidates) > 0:
            probs = self.sim.fp_pars['lactational_amenorrhea']['rate'][(self.postpartum_dur[lam_candidates]).astype(int)]

            self.binom.set(p=probs)
            lam_true, lam_false = self.binom.filter(lam_candidates, both=True)
            self.lam[lam_false] = False
            self.lam[lam_true] = True

        not_postpartum = uids[self.postpartum[uids] == 0]
        over5mo = self.postpartum_dur[uids] > max_lam_dur
        not_breastfeeding = self.breastfeed_dur[uids] == 0
        not_lam = uids[not_postpartum & over5mo & not_breastfeeding]
        self.lam[not_lam] = False

        return

    def update_breastfeeding(self, uids):
        """
        Track breastfeeding, and update time of breastfeeding for individual pregnancy.
        Agents are randomly assigned a duration value based on a truncated normal distribution drawn
        from the 2018 DHS variable for breastfeeding months.
        The mean and the std dev are both drawn from that distribution in the DHS data.
        """
        mean, sd = self.sim.fp_pars['breastfeeding_dur_mean'], self.sim.fp_pars['breastfeeding_dur_sd']
        a, b = 0, 50  # Truncate at 0 to ensure positive durations
        breastfeed_durs = fpu.sample(dist='normal_int', par1=mean, par2=sd, size=len(uids))
        breastfeed_durs = np.clip(breastfeed_durs, a, b)
        breastfeed_finished = uids[self.breastfeed_dur[uids] >= breastfeed_durs]
        breastfeed_continue = uids[self.breastfeed_dur[uids] < breastfeed_durs]
        self.reset_breastfeeding(breastfeed_finished)
        self.breastfeed_dur[breastfeed_continue] += self.sim.t.dt_year * fpd.mpy
        return

    def update_postpartum(self, uids):
        """
        Track duration of extended postpartum period (0-24 months after birth).
        Only enter this function if agent is postpartum.
        """

        # Stop postpartum episode if reach max length (set to 24 months)
        pp_done = uids[(self.postpartum_dur[uids] >= self.sim.fp_pars['postpartum_dur'])]
        self.postpartum[pp_done] = False
        self.postpartum_dur[pp_done] = 0

        # Count the state of the agent for postpartum -- # TOOD: refactor, what is this loop doing?
        postpart = uids[(self.postpartum[uids] == True)]
        for key, (pp_low, pp_high) in fpd.postpartum_map.items():
            this_pp_bin = postpart[(self.postpartum_dur[postpart] >= pp_low) & (self.postpartum_dur[postpart] < pp_high)]
            self.sim.results[key][self.sim.ti] += len(this_pp_bin)
        self.postpartum_dur[postpart] += self.sim.t.dt_year * fpd.mpy

        return

    def progress_pregnancy(self, uids):
        """ Advance pregnancy in time and check for miscarriage """

        preg = uids[self.pregnant[uids] == True]
        self.gestation[preg] += self.sim.t.dt_year * fpd.mpy

        # Check for miscarriage at the end of the first trimester
        end_first_tri = preg[(self.gestation[preg] == self.sim.fp_pars['end_first_tri'])]
        miscarriage_probs = self.sim.fp_pars['miscarriage_rates'][self.int_age_clip(end_first_tri)]
        self.binom.set(p=miscarriage_probs)
        miscarriage = self.binom.filter(end_first_tri)

        # Reset states and track miscarriages
        n_miscarriages = len(miscarriage)
        self.sim.results['miscarriages'][self.sim.ti] = n_miscarriages

        if n_miscarriages:
            for miscarriage_uid in miscarriage:
                # put miscarriage age in first nan slot
                miscarriage_age_index = np.where(np.isnan(self.miscarriage_ages[miscarriage_uid]))[0][0]
                self.miscarriage_ages[miscarriage_uid, miscarriage_age_index] = self.age[miscarriage_uid]
            self.pregnant[miscarriage] = False
            self.miscarriage[miscarriage] += 1  # Add 1 to number of miscarriages agent has had
            self.postpartum[miscarriage] = False
            self.gestation[miscarriage] = 0  # Reset gestation counter
            self.ti_contra[miscarriage] = self.sim.ti+1  # Update contraceptive choices

        return

    def reset_breastfeeding(self, uids):
        """
        Stop breastfeeding, calculate total lifetime duration so far, and reset lactation episode to zero
        """
        self.lactating[uids] = False
        self.breastfeed_dur_total[uids] += self.breastfeed_dur[uids]
        self.breastfeed_dur[uids] = 0
        return

    def check_maternal_mortality(self, uids):
        """
        Check for probability of maternal mortality
        """
        prob = self.sim.fp_pars['mortality_probs']['maternal'] * self.sim.fp_pars['maternal_mortality_factor']
        self.binom.set(p=prob)
        death = self.binom.filter(uids)
        self.request_death(death)
        self.sim.results['maternal_deaths'][self.sim.ti] += len(death)
        self.sim.results['deaths'][self.sim.ti] += len(death)
        return death

    def check_infant_mortality(self, uids):
        """
        Check for probability of infant mortality (death < 1 year of age)
        """
        death_prob = (self.sim.fp_pars['mortality_probs']['infant'])
        if len(uids) > 0:
            age_inds = sc.findnearest(self.sim.fp_pars['infant_mortality']['ages'], self.age[uids])
            death_prob = death_prob * (self.sim.fp_pars['infant_mortality']['age_probs'][age_inds])
        self.binom.set(p=death_prob)
        death = self.binom.filter(uids)

        self.sim.results['infant_deaths'][self.sim.ti] += len(death)
        self.reset_breastfeeding(death)
        self.ti_contra[death] = self.sim.ti + 1  # Trigger update to contraceptive choices following infant death
        return death

    def process_delivery(self, uids):
        """
        Decide if pregnant woman gives birth and explore maternal mortality and child mortality
        """
        sim = self.sim
        fp_pars = sim.fp_pars
        ti = sim.ti

        # Update states
        deliv = uids[(self.gestation[uids] == self.preg_dur[uids])]
        if len(deliv):  # check for any deliveries
            self.pregnant[deliv] = False
            self.gestation[deliv] = 0  # Reset gestation counter
            self.lactating[deliv] = True
            self.postpartum[deliv] = True  # Start postpartum state at time of birth
            self.breastfeed_dur[deliv] = 0  # Start at 0, will update before leaving timestep in separate function
            self.postpartum_dur[deliv] = 0
            self.ti_contra[deliv] = ti + 1  # Trigger a call to re-evaluate whether to use contraception when 1month pp

            # Handle stillbirth
            still_prob = fp_pars['mortality_probs']['stillbirth']
            rate_ages = fp_pars['stillbirth_rate']['ages']

            age_ind = np.searchsorted(rate_ages, self.age[deliv], side="left")
            prev_idx_is_less = ((age_ind == len(rate_ages)) | (
                    np.fabs(self.age[deliv] - rate_ages[np.maximum(age_ind - 1, 0)]) < np.fabs(
                self.age[deliv] - rate_ages[np.minimum(age_ind, len(rate_ages) - 1)])))
            age_ind[prev_idx_is_less] -= 1  # adjusting for quirks of np.searchsorted
            still_prob = still_prob * (fp_pars['stillbirth_rate']['age_probs'][age_ind]) if len(self) > 0 else 0

            self.binom.set(p=still_prob)
            stillborn, live = self.binom.filter(deliv, both=True)

            self.stillbirth[stillborn] += 1  # Track how many stillbirths an agent has had
            self.lactating[stillborn] = False  # Set agents of stillbith to not lactate
            sim.results['stillbirths'][ti] = len(stillborn)

            # Handle twins
            self.binom.set(fp_pars['twins_prob'])
            twin, single = self.binom.filter(live, both=True)
            sim.results['births'][ti] += 2 * len(twin)  # only add births to population if born alive
            sim.results['births'][ti] += len(single)

            # Record ages of agents when live births / stillbirths occur
            for parity in np.unique(self.parity[single]):
                single_uids = single[self.parity[single] == parity]
                # for uid in single_uids:
                self.birth_ages[ss.uids(single_uids), int(parity)] = self.age[ss.uids(single_uids)]
                if parity == 0: self.first_birth_age[single_uids] = self.age[single_uids]
            for parity in np.unique(self.parity[twin]):
                twin_uids = twin[self.parity[twin] == parity]
                # for uid in twin_uids:
                self.birth_ages[twin_uids, int(parity)] = self.age[twin_uids]
                self.birth_ages[twin_uids, int(parity) + 1] = self.age[twin_uids]
                if parity == 0: self.first_birth_age[twin_uids] = self.age[twin_uids]
            for parity in np.unique(self.parity[stillborn]):
                uids = stillborn[self.parity[stillborn] == parity]
                # for uid in uids:
                self.stillborn_ages[uids, int(parity)] = self.age[uids]

            self.parity[single] += 1
            self.parity[twin] += 2  # Add 2 because matching DHS "total children ever born (alive) v201"

            # Calculate short intervals
            prev_birth_single = single[self.parity[single] > 1]
            prev_birth_twins = twin[self.parity[twin] > 2]
            if len(prev_birth_single):
                pidx = (self.parity[prev_birth_single] - 1).astype(int)
                all_ints = [self.birth_ages[r, pidx] - self.birth_ages[r, pidx-1] for r in prev_birth_single]
                latest_ints = np.array([r[~np.isnan(r)][-1] for r in all_ints])
                short_ints = np.count_nonzero(latest_ints < (fp_pars['short_int']/fpd.mpy))
                sim.results['short_intervals'][ti] += short_ints
            if len(prev_birth_twins):
                pidx = (self.parity[prev_birth_twins] - 2).astype(int)
                all_ints = [self.birth_ages[r, pidx] - self.birth_ages[r, pidx-1] for r in prev_birth_twins]
                latest_ints = np.array([r[~np.isnan(r)][-1] for r in all_ints])
                short_ints = np.count_nonzero(latest_ints < (fp_pars['short_int']/fpd.mpy))
                sim.results['short_intervals'][ti] += short_ints

            # Calculate total births
            sim.results['total_births'][ti] = len(stillborn) + sim.results['births'][ti]

            live_age = self.age[live]
            for key, (age_low, age_high) in fpd.age_bin_map.items():
                match_low_high = live[(self.age[live] >=age_low) & (self.age[live] < age_high)]
                birth_bins = len(match_low_high)

                sim.results[f'total_births_{key}'][ti] += birth_bins

            # Check mortality
            maternal_deaths = self.check_maternal_mortality(live)  # Mothers of only live babies eligible to match definition of maternal mortality ratio
            i_death = self.check_infant_mortality(live)

            new_uids = self.grow(len(live) - len(i_death))

            # Be sure to reset the new agents to the correct states!
            self.init_vals(new_uids)
            self.age[new_uids] = 0

            if new_uids is not None:
                return new_uids

        return

    def update_age_bin_totals(self, uids):
        """
        Count how many total live women in each 5-year age bin 10-50, for tabulating ASFR
        """
        if uids is None:
            uids = self.alive.uids

        for key, (age_low, age_high) in fpd.age_bin_map.items():
            this_age_bin = uids[(self.age[uids] >= age_low) & (self.age[uids] < age_high)]
            self.sim.results[f'total_women_{key}'][self.sim.ti] += len(this_age_bin)

        return

    def step(self):
        """
        Perform all updates to people within a single timestep
        """

        # normally SS handles deaths at end of timestep, but to match the previous version's logic, we start it here.
        # dead agents are removed so we don't have to filter for alive after this.
        self.decide_death_outcome(self.alive.uids)

        # Update pregnancy with maternal mortality outcome
        self.process_delivery(self.pregnant.uids)  # Deliver with birth outcomes if reached pregnancy duration

        # Reselect for live agents after exposure to maternal mortality and infant mortality
        fecund = ((self.female) & (self.age < self.sim.fp_pars['age_limit_fecundity'])).uids

        nonpreg = fecund[self.pregnant[fecund] == False]
        lact = fecund[self.lactating[fecund] == True]

        # Check who has reached their age at first partnership and set partnered attribute to True.
        self.start_partnership(self.female.uids)

        # Complete all updates. Note that these happen in a particular order!
        self.progress_pregnancy(self.pregnant.uids)  # Advance gestation in timestep, handle miscarriage

        # Update mothers
        if self.sim.fp_pars['track_children']:
            self.update_mothers()

        # Check if agents are sexually active, and update their intent to use contraception
        self.check_sexually_active(nonpreg)

        # Update methods for those who are eligible
        ready = nonpreg[self.ti_contra[nonpreg] <= self.sim.ti]
        if len(ready):
            self.update_method(ready)
            self.sim.results['switchers'][self.sim.ti] = len(ready)  # Track how many people switch methods (incl on/off)

        methods_ok = np.array_equal(self.on_contra.nonzero()[-1], self.method.nonzero()[-1])
        if not methods_ok:
            errormsg = 'Agents not using contraception are not the same as agents who are using None method'
            raise ValueError(errormsg)

        self.update_postpartum(nonpreg)  # Updates postpartum counter if postpartum
        self.update_breastfeeding(lact)
        self.check_lam(nonpreg)
        self.check_conception(nonpreg)  # Decide if conceives and initialize gestation counter at 0

        # Update results
        self.update_age_bin_totals(fecund)

        # Add check for ti contra
        if (self.ti_contra < 0).any():
            errormsg = f'Invalid values for ti_contra at timestep {self.ti}'
            raise ValueError(errormsg)

        return

    def update_results(self):
        super().update_results()
        sim = self.sim
        res = sim.results
        ti = sim.ti
        age_min = self.age >= fp.min_age
        age_max = self.age < sim.pars.fp['age_limit_fecundity']

        res['n_fecund'][ti] = np.sum(self.female * age_min * age_max)
        res['n_urban'][ti] = np.sum(self.urban * self.female)
        res['ever_used_contra'][ti] = np.sum(self.ever_used_contra * self.female) / np.sum(self.female) * 100
        res['parity0to1'][ti] = np.sum((self.parity <= 1) & self.female) / np.sum(self.female) * 100
        res['parity2to3'][ti] = np.sum((self.parity >= 2) & (self.parity <= 3) & self.female) / np.sum(self.female) * 100
        res['parity4to5'][ti] = np.sum((self.parity >= 4) & (self.parity <= 5) & self.female) / np.sum(self.female) * 100
        res['parity6plus'][ti] = np.sum((self.parity >= 6) & self.female) / np.sum(self.female) * 100

        percent0to5 = (res.pp0to5[ti] / res.n_fecund[ti]) * 100
        percent6to11 = (res.pp6to11[ti] / res.n_fecund[ti]) * 100
        percent12to23 = (res.pp12to23[ti] / res.n_fecund[ti]) * 100
        nonpostpartum = ((res.n_fecund[ti] - res.pp0to5[ti] - res.pp6to11[ti] - res.pp12to23[ti]) / res.n_fecund[ti]) * 100

        # Store results
        res['pp0to5'][ti] = percent0to5
        res['pp6to11'][ti] = percent6to11
        res['pp12to23'][ti] = percent12to23
        res['nonpostpartum'][ti] = nonpostpartum

