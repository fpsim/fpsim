"""
Defines the People class
"""

# %% Imports
import numpy as np  # Needed for a few things not provided by pl
import pylab as pl
import seaborn as sns
import sciris as sc
import pandas as pd
from .settings import options as fpo
from . import utils as fpu
from . import defaults as fpd
from . import base as fpb
from . import parameters as fpp

# Specify all externally visible things this file defines
__all__ = ['People']


# %% Define classes
def arr(n=None, val=0):
    """
    Shortcut for defining an empty array with the correct value and data type
    """
    if isinstance(val, np.ndarray):
        assert len(val) == n
        arr = val
    elif isinstance(val, list):
        arr = [[] for _ in range(n)]
    else:
        dtype = object if isinstance(val, str) else None
        arr = np.full(shape=n, fill_value=val, dtype=dtype)
    return arr


class People(fpb.BasePeople):
    """
    Class for all the people in the simulation.
    """

    def __init__(self, pars, n=None, **kwargs):

        # Initialization
        super().__init__()
        self.pars = pars  # Set parameters
        d = sc.mergedicts(fpd.person_defaults, kwargs)  # d = defaults
        if n is None:
            n = int(self.pars['n_agents'])

        # Basic states
        init_states = dir(self)
        self.uid = arr(n, np.arange(n))
        self.age = arr(n, np.float64(d['age']))  # Age of the person (in years)
        self.age_by_group = arr(n,
                                np.float64(d['age_by_group']))  # Age by which method bin the age falls into, as integer
        self.sex = arr(n, d['sex'])  # Female (0) or male (1)
        self.parity = arr(n, d['parity'])  # Number of children
        self.method = arr(n,
                          d['method'])  # Contraceptive method 0-9, see pars['methods']['map'], excludes LAM as method
        self.barrier = arr(n, d['barrier'])  # Reason for non-use
        self.alive = arr(n, d['alive'])
        self.pregnant = arr(n, d['pregnant'])
        self.fertile = arr(n, d['fertile'])  # assigned likelihood of remaining childfree throughout reproductive years

        # Sexual and reproductive history
        self.sexually_active = arr(n, d['sexually_active'])
        self.sexual_debut = arr(n, d['sexual_debut'])
        self.sexual_debut_age = arr(n, np.float64(
            d['sexual_debut_age']))  # Age at first sexual debut in years, If not debuted, -1
        self.fated_debut = arr(n, np.float64(d['debut_age']))
        self.first_birth_age = arr(n, np.float64(d['first_birth_age']))  # Age at first birth.  If no births, -1
        self.lactating = arr(n, d['lactating'])
        self.gestation = arr(n, d['gestation'])
        self.preg_dur = arr(n, d['preg_dur'])
        self.stillbirth = arr(n, d['stillbirth'])  # Number of stillbirths
        self.miscarriage = arr(n, d['miscarriage'])  # Number of miscarriages
        self.abortion = arr(n, d['abortion'])  # Number of abortions
        self.pregnancies = arr(n, d['pregnancies'])  # Number of conceptions (before abortion)
        self.months_inactive = arr(n, d[
            'months_inactive'])  # Number of months an agents has been sexually inactive once debuted
        self.postpartum = arr(n, d['postpartum'])
        self.mothers = arr(n, d['mothers'])
        self.short_interval = arr(n, d['short_interval'])  # Number of short birth intervals
        self.secondary_birth = arr(n, d['secondary_birth'])  # Number of secondary live birth

        self.postpartum_dur = arr(n, d['postpartum_dur'])  # Tracks # months postpartum
        self.lam = arr(n,
                       d['lam'])  # Separately tracks lactational amenorrhea, can be using both LAM and another method
        self.breastfeed_dur = arr(n, d['breastfeed_dur'])
        self.breastfeed_dur_total = arr(n, d['breastfeed_dur_total'])

        self.children = arr(n, [])  # Indices of children -- list of lists
        self.dobs = arr(n, [])  # Dates of births -- list of lists
        self.still_dates = arr(n, [])  # Dates of stillbirths -- list of lists
        self.miscarriage_dates = arr(n, [])  # Dates of miscarriages -- list of lists
        self.abortion_dates = arr(n, [])  # Dates of abortions -- list of lists
        self.short_interval_dates = arr(n, [])  # age of agents at short birth interval -- list of lists

        # Fecundity variation
        fv = [self.pars['fecundity_var_low'], self.pars['fecundity_var_high']]
        self.personal_fecundity = arr(n, np.random.random(n) * (fv[1] - fv[0]) + fv[
            0])  # Stretch fecundity by a factor bounded by [f_var[0], f_var[1]]
        self.remainder_months = arr(n, d['remainder_months'])

        # Empowerment-related sociodemographic attributes
        self.partnered = arr(n, d['partnered'])  # Whether a person is in a relationship or not
        self.partnership_age = arr(n, d['partnership_age'])  # Age at first partnership in years, initialised from data
        self.urban = arr(n, d['urban'])  # Whether a person lives in rural or urban setting
        self.paid_employment = arr(n, d['paid_employment'])  # Whether a person has a paid job or not
        self.decision_wages = arr(n,
                                  d['decision_wages'])  # Decision making autonomy over major household purchases/wages
        self.decision_health = arr(n, d['decision_health'])  # Decision making autonomy over her health
        self.sexual_autonomy = arr(n, d['sexual_autonomy'])  # Ability to refuse sex

        # Empowerment-education attributes
        self.edu_objective = arr(n, d[
            'edu_objective'])  # Highest-ideal level of education to be completed (in years), could be individualised or constant across agents
        self.edu_attainment = arr(n, d['edu_attainment'])  # Current level of education achieved in years
        self.edu_dropout = arr(n, d[
            'edu_dropout'])  # Whether a person has dropped out of the edu system, before reaching their goal
        self.edu_interrupted = arr(n, d[
            'edu_interrupted'])  # Whether a person/woman has had their education temporarily interrupted, but can resume
        self.edu_completed = arr(n, d['edu_completed'])  # Whether a person/woman has reached their education goals
        self.edu_started = arr(n, d['edu_started'])  # Whether a person/woman has started thier education
        # Store keys
        final_states = dir(self)
        self._keys = [s for s in final_states if s not in init_states]

        return

    def update_method(self):
        """
        Uses a switching matrix from DHS data to decide based on a person's original method their probability of changing to a
        new method and assigns them the new method. Currently allows switching on whole calendar years to enter function.
        Matrix serves as an initiation, discontinuation, continuation, and switching matrix. Transition probabilities are for 1 year and
        only for women who have not given birth within the last 6 months.
        """
        methods = self.pars['methods']
        method_map = methods['map']
        annual = methods['adjusted']['annual']
        orig_methods = self.method
        m = len(method_map)
        switching_events = np.zeros((m, m), dtype=int)
        switching_events_ages = {}
        for key in fpd.method_age_map.keys():
            switching_events_ages[key] = np.zeros((m, m), dtype=int)

        # Method switching depends both on agent age and also on their current method, so we need to loop over both
        for key, (age_low, age_high) in fpd.method_age_map.items():
            match_low = (self.age >= age_low)  # CK: TODO: refactor into single method
            match_high = (self.age < age_high)
            match_low_high = match_low * match_high
            for m in method_map.values():
                match_m = (orig_methods == m)
                match = match_m * match_low_high
                this_method = self.filter(match)
                old_method = this_method.method.copy()

                matrix = annual[key]
                choices = matrix[m]
                choices = choices / choices.sum()
                new_methods = fpu.n_multinomial(choices, match.sum())
                this_method.method = new_methods

                for i in range(len(old_method)):
                    x = old_method[i]
                    y = new_methods[i]
                    switching_events[x, y] += 1
                    switching_events_ages[key][x, y] += 1

        if self.pars['track_switching']:
            self.step_results_switching[
                'annual'] += switching_events  # CK: TODO: remove this extra result and combine with step_results
            for key in fpd.method_age_map.keys():
                self.step_results['switching_annual'][key] += switching_events_ages[key]

        return

    def update_method_pp(self):
        """
        Utilizes data from birth to allow agent to initiate a method postpartum coming from birth by
        3 months postpartum and then initiate, continue, or discontinue a method by 6 months postpartum.
        Next opportunity to switch methods will be on whole calendar years, whenever that falls.
        """
        # TODO- Probabilities need to be adjusted for postpartum women on the next annual draw in "get_method" since they may be less than one year

        # Probability of initiating a postpartum method at 0-3 months postpartum
        # Transitional probabilities are for the first 3 month time period after delivery from DHS data

        methods = self.pars['methods']
        pp0to1 = methods['adjusted']['pp0to1']
        pp1to6 = methods['adjusted']['pp1to6']
        methods_map = methods['map']
        orig_methods = self.method

        m = len(methods_map)
        switching_events = np.zeros((m, m), dtype=int)
        switching_events_ages = {}
        for key in fpd.method_age_map.keys():
            switching_events_ages[key] = np.zeros((m, m), dtype=int)

        postpartum1 = (self.postpartum_dur == 0)
        postpartum6 = (self.postpartum_dur == 6)

        # In first time step after delivery, choice is by age but not previous method (since just gave birth)
        # All women are coming from birth and on no method to start, either will stay on no method or initiate a method
        for key, (age_low, age_high) in fpd.method_age_map.items():
            match_low = (self.age >= age_low)
            match_high = (self.age < age_high)
            low_parity = (self.parity < self.pars['high_parity'])
            high_parity = (self.parity >= self.pars['high_parity'])
            match = (self.postpartum * postpartum1 * match_low * match_high * low_parity)
            match_high_parity = (self.postpartum * postpartum1 * match_low * match_high * high_parity)
            this_method = self.filter(match)
            this_method_high_parity = self.filter(match_high_parity)
            old_method = this_method.method.copy()
            old_method_high_parity = sc.dcp(this_method_high_parity.method)

            choices = pp0to1[key]
            choices_high_parity = sc.dcp(choices)
            choices_high_parity[0] *= self.pars['high_parity_nonuse']
            choices_high_parity = choices_high_parity / choices_high_parity.sum()
            new_methods = fpu.n_multinomial(choices, len(this_method))
            new_methods_high_parity = fpu.n_multinomial(choices_high_parity, len(this_method_high_parity))
            this_method.method = np.array(new_methods, dtype=np.int64)
            this_method_high_parity.method = np.array(new_methods_high_parity, dtype=np.int64)
            for i in range(len(old_method)):
                x = old_method[i]
                y = new_methods[i]
                switching_events[x, y] += 1
                switching_events_ages[key][x, y] += 1

            for i in range(len(old_method_high_parity)):
                x = old_method_high_parity[i]
                y = new_methods_high_parity[i]
                switching_events[x, y] += 1
                switching_events_ages[key][x, y] += 1

        # At 6 months, choice is by previous method and by age
        # Allow initiation, switching, or discontinuing with matrix at 6 months postpartum
        # Transitional probabilities are for 5 months, 1-6 months after delivery from DHS data
        for key, (age_low, age_high) in fpd.method_age_map.items():
            match_low = (self.age >= age_low)
            match_high = (self.age < age_high)
            match_postpartum_age = self.postpartum * postpartum6 * match_low * match_high
            for m in methods_map.values():
                match_m = (orig_methods == m)
                match = match_m * match_postpartum_age
                this_method = self.filter(match)
                old_method = self.method[match].copy()

                matrix = pp1to6[key]
                choices = matrix[m]
                new_methods = fpu.n_multinomial(choices, match.sum())
                this_method.method = new_methods
                for i in range(len(old_method)):
                    x = old_method[i]
                    y = new_methods[i]
                    switching_events[x, y] += 1
                    switching_events_ages[key][x, y] += 1

        if self.pars['track_switching']:
            self.step_results_switching['postpartum'] += switching_events
            for key in fpd.method_age_map.keys():
                self.step_results['switching_postpartum'][key] += switching_events_ages[key]

        return

    def update_methods(self):
        """
        If eligible (age 15-49 and not pregnant), choose new method or stay with current one
        """

        if not (self.i % self.pars['method_timestep']):  # Allow skipping timesteps
            postpartum = (self.postpartum) * (self.postpartum_dur <= 6)
            pp = self.filter(postpartum)
            non_pp = self.filter(~postpartum)

            pp.update_method_pp()  # Update method for

            age_diff = non_pp.ceil_age - non_pp.age
            whole_years = ((age_diff < (1 / fpd.mpy)) * (age_diff > 0))
            birthdays = non_pp.filter(whole_years)
            birthdays.update_method()

        return

    def check_mortality(self):
        '''Decide if person dies at a timestep'''

        timestep = self.pars['timestep']
        trend_val = self.pars['mortality_probs']['gen_trend']
        age_mort = self.pars['age_mortality']
        f_spline = age_mort['f_spline'] * trend_val
        m_spline = age_mort['m_spline'] * trend_val
        over_one = self.filter(self.age >= 1)
        female = over_one.filter(over_one.is_female)
        male = over_one.filter(over_one.is_male)
        f_ages = female.int_age
        m_ages = male.int_age

        f_mort_prob = fpu.annprob2ts(f_spline[f_ages], timestep)
        m_mort_prob = fpu.annprob2ts(m_spline[m_ages], timestep)

        f_died = female.binomial(f_mort_prob, as_filter=True)
        m_died = male.binomial(m_mort_prob, as_filter=True)
        for died in [f_died, m_died]:
            died.alive = False,
            died.pregnant = False,
            died.gestation = False,
            died.sexually_active = False,
            died.lactating = False,
            died.postpartum = False,
            died.lam = False,
            died.breastfeed_dur = 0,
            self.step_results['deaths'] += len(died)

        return

    def check_partnership(self):
        '''
        Decide if an agent has reached their age at first partnership. Age-based data from DHS.
        '''

        is_not_partnered = self.partnered == 0
        reached_partnership_age = self.age >= self.partnership_age
        first_timers = self.filter(is_not_partnered * reached_partnership_age)
        first_timers.partnered = True

    def check_sexually_active(self):
        '''
        Decide if agent is sexually active based either on month postpartum or age if
        not postpartum.  Postpartum and general age-based data from DHS.
        '''
        # Set postpartum probabilities
        match_low = self.postpartum_dur >= 0
        match_high = self.postpartum_dur <= self.pars['postpartum_dur']
        pp_match = self.postpartum * match_low * match_high
        non_pp_match = ((self.age >= self.fated_debut) * (~pp_match))
        pp = self.filter(pp_match)
        non_pp = self.filter(non_pp_match)

        # Adjust for postpartum women's birth spacing preferences
        pref = self.pars['spacing_pref']  # Shorten since used a lot
        spacing_bins = pp.postpartum_dur / pref['interval']  # Main calculation -- divide the duration by the interval
        spacing_bins = np.array(np.minimum(spacing_bins, pref['n_bins']),
                                dtype=int)  # Convert to an integer and bound by longest bin
        probs_pp = self.pars['sexual_activity_pp']['percent_active'][pp.postpartum_dur]
        probs_pp *= pref['preference'][
            spacing_bins]  # Actually adjust the probability -- check the overall probability with print(pref['preference'][spacing_bins].mean())

        # Set non-postpartum probabilities
        probs_non_pp = self.pars['sexual_activity'][non_pp.int_age]

        # Evaluate likelihood in this time step of being sexually active
        # Can revert to active or not active each timestep
        pp.sexually_active = fpu.binomial_arr(probs_pp)
        non_pp.sexually_active = fpu.binomial_arr(probs_non_pp)

        # Set debut to True if sexually active for the first time
        # Record agent age at sexual debut in their memory
        never_sex = non_pp.sexual_debut == 0
        now_active = non_pp.sexually_active == 1
        first_debut = non_pp.filter(now_active * never_sex)
        first_debut.sexual_debut = True
        first_debut.sexual_debut_age = first_debut.age

        active_sex = self.sexually_active == 1
        debuted = self.sexual_debut == 1
        active = self.filter(active_sex * debuted)
        inactive = self.filter(~active_sex * debuted)
        active.months_inactive = 0
        inactive.months_inactive += 1

        inactive_year = self.months_inactive >= 12
        sexually_infrequent = self.filter(inactive_year)

        # print (f'Age: {sexually_infrequent.age}')
        # print (f'Debuted?: {sexually_infrequent.sexual_debut}')
        # print (f'Debut age: {sexually_infrequent.sexual_debut_age}')
        # print (f'Months inactive: {sexually_infrequent.months_inactive}')
        # print (f'On method?: {sexually_infrequent.method}')

        return

    def check_conception(self):
        '''
        Decide if person (female) becomes pregnant at a timestep.
        '''
        all_ppl = self.unfilter()  # For complex array operations
        active = self.filter(self.sexually_active * self.fertile)
        lam = active.filter(active.lam)
        nonlam = active.filter(~active.lam)
        preg_probs = np.zeros(len(all_ppl))  # Use full array

        # Find monthly probability of pregnancy based on fecundity and any use of contraception including LAM - from data
        pars = self.pars  # Shorten
        preg_eval_lam = pars['age_fecundity'][lam.int_age_clip] * lam.personal_fecundity
        preg_eval_nonlam = pars['age_fecundity'][nonlam.int_age_clip] * nonlam.personal_fecundity
        method_eff = np.array(list(pars['methods']['eff'].values()))[nonlam.method]
        lam_eff = pars['LAM_efficacy']

        lam_probs = fpu.annprob2ts((1 - lam_eff) * preg_eval_lam, pars['timestep'])
        nonlam_probs = fpu.annprob2ts((1 - method_eff) * preg_eval_nonlam, pars['timestep'])
        preg_probs[lam.inds] = lam_probs
        preg_probs[nonlam.inds] = nonlam_probs

        # Adjust for decreased likelihood of conception if nulliparous vs already gravid - from PRESTO data
        nullip = active.filter(active.parity == 0)  # Nulliparous
        preg_probs[nullip.inds] *= pars['fecundity_ratio_nullip'][nullip.int_age_clip]

        # Adjust for probability of exposure to pregnancy episode at this timestep based on age and parity - encapsulates background factors - experimental and tunable
        preg_probs *= pars['exposure_factor']
        preg_probs *= pars['exposure_age'][all_ppl.int_age_clip]
        preg_probs *= pars['exposure_parity'][np.minimum(all_ppl.parity, fpd.max_parity)]

        # Use a single binomial trial to check for conception successes this month
        conceived = active.binomial(preg_probs[active.inds], as_filter=True)
        self.step_results['pregnancies'] += len(conceived)  # track all pregnancies
        unintended = conceived.filter(conceived.method != 0)
        self.step_results['unintended_pregs'] += len(unintended)  # track pregnancies due to method failure

        # Check for abortion
        is_abort = conceived.binomial(pars['abortion_prob'])
        abort = conceived.filter(is_abort)
        preg = conceived.filter(~is_abort)

        # Update states
        all_ppl = self.unfilter()
        abort.postpartum = False
        abort.abortion += 1  # Add 1 to number of abortions agent has had
        abort.postpartum_dur = 0
        for i in abort.inds:  # Handle adding dates
            all_ppl.abortion_dates[i].append(all_ppl.age[i])
        self.step_results['abortions'] = len(abort)
        # Make selected agents pregnant
        preg.make_pregnant()
        if self.pars['track_as']:
            pregnant_boolean = np.full(len(self), False)
            pregnant_boolean[np.searchsorted(self.uid, preg.uid)] = True
            pregnant_age_split = self.log_age_split(binned_ages_t=[self.age_by_group], channel='pregnancies',
                                                    numerators=[pregnant_boolean], denominators=None)

            for key in pregnant_age_split:
                self.step_results[key] = pregnant_age_split[key]
        return

    def make_pregnant(self):
        '''
        Update the selected agents to be pregnant
        '''
        pregdur = [self.pars['preg_dur_low'], self.pars['preg_dur_high']]
        self.pregnant = True
        self.gestation = 1  # Start the counter at 1
        self.preg_dur = np.random.randint(pregdur[0], pregdur[1] + 1, size=len(self))  # Duration of this pregnancy
        self.postpartum = False
        self.postpartum_dur = 0
        self.reset_breastfeeding()  # Stop lactating if becoming pregnant
        self.method = 0
        return

    def check_lam(self):
        '''
        Check to see if postpartum agent meets criteria for LAM in this time step
        '''
        max_lam_dur = self.pars['max_lam_dur']
        lam_candidates = self.filter((self.postpartum) * (self.postpartum_dur <= max_lam_dur))
        probs = self.pars['lactational_amenorrhea']['rate'][lam_candidates.postpartum_dur]
        lam_candidates.lam = lam_candidates.binomial(probs)

        not_postpartum = self.postpartum == 0
        over5mo = self.postpartum_dur > max_lam_dur
        not_breastfeeding = self.breastfeed_dur == 0
        not_lam = self.filter(not_postpartum + over5mo + not_breastfeeding)
        not_lam.lam = False

        return

    def update_breastfeeding(self):
        '''
        Track breastfeeding, and update time of breastfeeding for individual pregnancy.
        Agents are randomly assigned a duration value based on a gumbel distribution drawn from the 2018 DHS variable for breastfeeding months. The mean (mu) and the std dev (beta) are both drawn from that distribution in the DHS data.
        '''
        mu, beta = self.pars['breastfeeding_dur_mu'], self.pars['breastfeeding_dur_beta']
        breastfeed_durs = abs(np.random.gumbel(mu, beta, size=len(self)))
        breastfeed_durs = np.ceil(breastfeed_durs)
        breastfeed_finished_inds = self.breastfeed_dur >= breastfeed_durs
        breastfeed_finished = self.filter(breastfeed_finished_inds)
        breastfeed_continue = self.filter(~breastfeed_finished_inds)
        breastfeed_finished.reset_breastfeeding()
        breastfeed_continue.breastfeed_dur += self.pars['timestep']
        return

    def update_postpartum(self):
        '''Track duration of extended postpartum period (0-24 months after birth).  Only enter this function if agent is postpartum'''

        # Stop postpartum episode if reach max length (set to 24 months)
        pp_done = self.filter(self.postpartum_dur >= self.pars['postpartum_dur'])
        pp_done.postpartum = False
        pp_done.postpartum_dur = 0

        # Count the state of the agent for postpartum -- # TOOD: refactor, what is this loop doing?
        pp = self.filter(self.postpartum)
        for key, (pp_low, pp_high) in fpd.postpartum_map.items():
            this_pp_bin = pp.filter((pp.postpartum_dur >= pp_low) * (pp.postpartum_dur < pp_high))
            self.step_results[key] += len(this_pp_bin)
        pp.postpartum_dur += self.pars['timestep']

        return

    def update_pregnancy(self):
        '''Advance pregnancy in time and check for miscarriage'''

        preg = self.filter(self.pregnant)
        preg.gestation += self.pars['timestep']

        # Check for miscarriage at the end of the first trimester
        end_first_tri = preg.filter(preg.gestation == self.pars['end_first_tri'])
        miscarriage_probs = self.pars['miscarriage_rates'][end_first_tri.int_age_clip]
        miscarriage = end_first_tri.binomial(miscarriage_probs, as_filter=True)

        # Reset states and track miscarriages
        all_ppl = self.unfilter()
        miscarriage.pregnant = False
        miscarriage.miscarriage += 1  # Add 1 to number of miscarriages agent has had
        miscarriage.postpartum = False
        miscarriage.gestation = 0  # Reset gestation counter
        for i in miscarriage.inds:  # Handle adding dates
            all_ppl.miscarriage_dates[i].append(all_ppl.age[i])
        self.step_results['miscarriages'] = len(miscarriage)
        return

    def reset_breastfeeding(self):
        '''Stop breastfeeding, calculate total lifetime duration so far, and reset lactation episode to zero'''
        self.lactating = False
        self.breastfeed_dur_total += self.breastfeed_dur
        self.breastfeed_dur = 0
        return

    def check_maternal_mortality(self):
        '''Check for probability of maternal mortality'''
        prob = self.pars['mortality_probs']['maternal'] * self.pars['maternal_mortality_factor']
        is_death = self.binomial(prob)
        death = self.filter(is_death)
        death.alive = False
        self.step_results['maternal_deaths'] += len(death)
        self.step_results['deaths'] += len(death)
        return death

    def check_infant_mortality(self):
        '''Check for probability of infant mortality (death < 1 year of age)'''
        death_prob = (self.pars['mortality_probs']['infant'])
        if len(self) > 0:
            age_inds = sc.findnearest(self.pars['infant_mortality']['ages'], self.age)
            death_prob = death_prob * (self.pars['infant_mortality']['age_probs'][age_inds])
        is_death = self.binomial(death_prob)
        death = self.filter(is_death)
        self.step_results['infant_deaths'] += len(death)
        death.reset_breastfeeding()
        return death

    def check_delivery(self):
        '''Decide if pregnant woman gives birth and explore maternal mortality and child mortality'''

        # Update states
        deliv = self.filter(self.gestation == self.preg_dur)
        if len(deliv):  # check for any deliveries
            deliv.pregnant = False
            deliv.gestation = 0  # Reset gestation counter
            deliv.lactating = True
            deliv.postpartum = True  # Start postpartum state at time of birth
            deliv.breastfeed_dur = 0  # Start at 0, will update before leaving timestep in separate function
            deliv.postpartum_dur = 0

            # Handle stillbirth
            still_prob = self.pars['mortality_probs']['stillbirth']
            rate_ages = self.pars['stillbirth_rate']['ages']
            age_ind = np.searchsorted(rate_ages, deliv.age, side="left")
            prev_idx_is_less = ((age_ind == len(rate_ages)) | (
                    np.fabs(deliv.age - rate_ages[np.maximum(age_ind - 1, 0)]) < np.fabs(
                deliv.age - rate_ages[np.minimum(age_ind, len(rate_ages) - 1)])))
            age_ind[prev_idx_is_less] -= 1  # adjusting for quirks of np.searchsorted
            still_prob = still_prob * (self.pars['stillbirth_rate']['age_probs'][age_ind]) if len(self) > 0 else 0

            is_stillborn = deliv.binomial(still_prob)
            stillborn = deliv.filter(is_stillborn)
            stillborn.stillbirth += 1  # Track how many stillbirths an agent has had
            stillborn.lactating = False  # Set agents of stillbith to not lactate
            self.step_results['stillbirths'] = len(stillborn)

            if self.pars['track_as']:
                stillbirth_boolean = np.full(len(self), False)
                stillbirth_boolean[np.searchsorted(self.uid, stillborn.uid)] = True

                self.step_results['stillbirth_ages'] = self.age_by_group
                self.step_results['as_stillbirths'] = stillbirth_boolean

            # Add dates of live births and stillbirths separately for agent to remember
            all_ppl = self.unfilter()
            live = deliv.filter(~is_stillborn)
            short_interval = 0
            secondary_birth = 0
            for i in live.inds:  # Handle DOBs
                all_ppl.dobs[i].append(all_ppl.age[
                                           i])  # Used for birth spacing only, only add one baby to dob -- CK: can't easily turn this into a Numpy operation
                if len(all_ppl.dobs[i]) == 1:
                    all_ppl.first_birth_age[i] = all_ppl.age[i]
                if (len(all_ppl.dobs[i]) > 1) and all_ppl.age[i] >= self.pars['low_age_short_int'] and all_ppl.age[i] < \
                        self.pars['high_age_short_int']:
                    secondary_birth += 1
                    if ((all_ppl.dobs[i][-1] - all_ppl.dobs[i][-2]) < (self.pars['short_int'] / fpd.mpy)):
                        all_ppl.short_interval_dates[i].append(all_ppl.age[i])
                        all_ppl.short_interval[i] += 1
                        short_interval += 1

            self.step_results['short_intervals'] += short_interval
            self.step_results['secondary_births'] += secondary_birth

            for i in stillborn.inds:  # Handle adding dates
                all_ppl.still_dates[i].append(all_ppl.age[i])

            # Add age of agents at birth with short birth interval
            # for i in live.inds: # Handle DOBs
            # if len(all_ppl.dobs[i]) > 1:
            # for d in range(len(all_ppl.dobs[i]) - 1):
            # if  (all_ppl.dobs[i][d + 1] - all_ppl.dobs[i][d]) < self.pars['short_int']:
            # short_interval_age = all_ppl.dobs[i][d+1].append(all_ppl.age[i][d+1])

            # Handle twins
            is_twin = live.binomial(self.pars['twins_prob'])
            twin = live.filter(is_twin)
            self.step_results['births'] += 2 * len(twin)  # only add births to population if born alive
            twin.parity += 2  # Add 2 because matching DHS "total children ever born (alive) v201"

            # Handle singles
            single = live.filter(~is_twin)
            self.step_results['births'] += len(single)
            single.parity += 1

            # Calculate total births
            self.step_results['total_births'] = len(stillborn) + self.step_results['births']

            live_age = live.age
            for key, (age_low, age_high) in fpd.age_bin_map.items():
                birth_bins = np.sum((live_age >= age_low) * (live_age < age_high))
                self.step_results['birth_bins'][key] += birth_bins

            if self.pars['track_as']:
                total_women_delivering = np.full(len(self), False)
                total_women_delivering[np.searchsorted(self.uid, live.uid)] = True
                self.step_results['mmr_age_by_group'] = self.age_by_group

            # Check mortality
            maternal_deaths = live.check_maternal_mortality()  # Mothers of only live babies eligible to match definition of maternal mortality ratio
            if self.pars['track_as']:
                maternal_deaths_bool = np.full(len(self), False)
                maternal_deaths_bool[np.searchsorted(self.uid, maternal_deaths.uid)] = True

                total_infants_bool = np.full(len(self), False)
                total_infants_bool[np.searchsorted(self.uid, live.uid)] = True

            i_death = live.check_infant_mortality()

            # Save infant deaths and totals into age buckets
            if self.pars['track_as']:
                infant_deaths_bool = np.full(len(self), False)
                infant_deaths_bool[np.searchsorted(self.uid, i_death.uid)] = True
                self.step_results[
                    'imr_age_by_group'] = self.age_by_group  # age groups have to be in same context as imr

                self.step_results[
                    'imr_numerator'] = infant_deaths_bool  # we need to track these over time to be summed by year
                self.step_results['imr_denominator'] = total_infants_bool
                self.step_results['mmr_numerator'] = maternal_deaths_bool
                self.step_results['mmr_denominator'] = total_women_delivering

                live_births_age_split = self.log_age_split(binned_ages_t=[self.age_by_group], channel='births',
                                                           numerators=[total_women_delivering], denominators=None)
                for key in live_births_age_split:
                    self.step_results[key] = live_births_age_split[key]

            # TEMP -- update children, need to refactor
            r = sc.dictobj(**self.step_results)
            new_people = r.births - r.infant_deaths  # Do not add agents who died before age 1 to population
            children_map = sc.ddict(int)
            for i in live.inds:
                children_map[i] += 1
            for i in twin.inds:
                children_map[i] += 1
            for i in i_death.inds:
                children_map[i] -= 1

            assert sum(list(children_map.values())) == new_people
            start_ind = len(all_ppl)
            for mother, n_children in children_map.items():
                end_ind = start_ind + n_children
                children = list(range(start_ind, end_ind))
                all_ppl.children[mother] += children
                start_ind = end_ind

        return

    def update_age(self):
        '''Advance age in the simulation'''
        self.age += self.pars['timestep'] / fpd.mpy  # Age the person for the next timestep
        self.age = np.minimum(self.age, self.pars['max_age'])

        return

    def update_education(self):
        '''Advance education attainment in the simulation, determine if agents have completed their educationm,
        '''

        # Filter people who have not: completed education, dropped out or had their education interrupted
        students = self.filter((self.edu_started & ~self.edu_completed & ~self.edu_dropout & ~self.edu_interrupted))
        # Advance education attainment
        students.edu_attainment += self.pars['timestep'] / fpd.mpy
        # Check who will experience an interruption
        students.interrupt_education()
        # Make some students dropout based on dropout | parity probabilities
        par1 = students.filter(students.parity == 1)
        par1.dropout_education('1')  # Women with parity 1
        par2plus = students.filter(students.parity >= 2)
        par2plus.dropout_education('2+')  # Women with parity 2+

    def graduate(self):
        completed_inds = sc.findinds(self.edu_attainment >= self.edu_objective)
        # NOTE: the two lines below were necessary because edu_completed was not being updating as expected
        tmp = self.edu_completed
        tmp[completed_inds] = True
        self.edu_completed = tmp

    def start_education(self):
        '''
        Begin education
        '''
        new_students = self.filter(~self.edu_started & (self.age >= self.pars["education"]["age_start"]))
        new_students.edu_started = True

    def interrupt_education(self):
        '''
        Interrupt education due to pregnancy. This method hinders education progression if a
        woman is pregnant and towards the end of the first trimester
        '''
        # Hinder education progression if a woman is pregnant and towards the end of the first trimester
        pregnant_students = self.filter(self.pregnant)
        end_first_tri = pregnant_students.filter(pregnant_students.gestation == self.pars['end_first_tri'])
        # Disrupt education
        end_first_tri.edu_interrupted = True

    def resume_education(self):
        '''
        # Basic mechanism to resume education post-pregnancy:
        # If education was interrupted due to pregnancy, resume after 9 months pospartum ()
        #TODO: check if there's any evidence supporting this assumption
        '''
        # Basic mechanism to resume education post-pregnancy:
        # If education was interrupted due to pregnancy, resume after 9 months pospartum
        pospartum_students = self.filter(
            self.postpartum & self.edu_interrupted & ~self.edu_completed & ~self.edu_dropout)
        resume_inds = sc.findinds(pospartum_students.postpartum_dur > 0.5 * self.pars['postpartum_dur'])
        tmp = pospartum_students.edu_interrupted
        tmp[resume_inds] = False
        pospartum_students.edu_interrupted = tmp

    def dropout_education(self, parity):
        dropout_dict = self.pars['education']['edu_dropout_probs'][parity]
        age_cutoffs = np.hstack((dropout_dict['age'], dropout_dict['age'].max() + 1))
        age_inds = np.digitize(self.age, age_cutoffs) - 1
        # Decide who will dropout
        self.edu_dropout = fpu.binomial_arr(dropout_dict['percent'][age_inds])

    def update_age_bin_totals(self):
        '''
        Count how many total live women in each 5-year age bin 10-50, for tabulating ASFR
        '''
        for key, (age_low, age_high) in fpd.age_bin_map.items():
            this_age_bin = self.filter((self.age >= age_low) * (self.age < age_high))
            self.step_results['age_bin_totals'][key] += len(this_age_bin)
        return

    def log_age_split(self, binned_ages_t, channel, numerators, denominators=None):
        counts_dict = {}
        results_dict = {}
        if denominators is not None:  # true when we are calculating rates (like cpr)
            for timestep_index in range(len(binned_ages_t)):
                if len(denominators[timestep_index]) == 0:
                    counts_dict[f"age_true_counts_{timestep_index}"] = {}
                    counts_dict[f"age_false_counts_{timestep_index}"] = {}
                else:
                    binned_ages = binned_ages_t[timestep_index]
                    binned_ages_true = binned_ages[
                        np.logical_and(numerators[timestep_index], denominators[timestep_index])]
                    if len(numerators[timestep_index]) == 0:
                        binned_ages_false = []  # ~[] doesnt make sense
                    else:
                        binned_ages_false = binned_ages[
                            np.logical_and(~numerators[timestep_index], denominators[timestep_index])]

                    counts_dict[f"age_true_counts_{timestep_index}"] = dict(
                        zip(*np.unique(binned_ages_true, return_counts=True)))
                    counts_dict[f"age_false_counts_{timestep_index}"] = dict(
                        zip(*np.unique(binned_ages_false, return_counts=True)))

            age_true_counts = {}
            age_false_counts = {}
            for age_counts_dict_key in counts_dict:
                for index in counts_dict[age_counts_dict_key]:
                    age_true_counts[index] = 0 if index not in age_true_counts else age_true_counts[index]
                    age_false_counts[index] = 0 if index not in age_false_counts else age_false_counts[index]
                    if 'false' in age_counts_dict_key:
                        age_false_counts[index] += counts_dict[age_counts_dict_key][index]
                    else:
                        age_true_counts[index] += counts_dict[age_counts_dict_key][index]

            for index, age_str in enumerate(fpd.age_specific_channel_bins):
                scale = 1
                if channel == "imr":
                    scale = 1000
                elif channel == "mmr":
                    scale = 100000
                if index not in age_true_counts:
                    results_dict[f"{channel}_{age_str}"] = 0
                elif index in age_true_counts and index not in age_false_counts:
                    results_dict[f"{channel}_{age_str}"] = 1.0 * scale
                else:
                    results_dict[f"{channel}_{age_str}"] = (age_true_counts[index] / (
                            age_true_counts[index] + age_false_counts[index])) * scale
        else:  # true when we are calculating counts (like pregnancies)
            for timestep_index in range(len(binned_ages_t)):
                if len(numerators[timestep_index]) == 0:
                    counts_dict[f"age_counts_{timestep_index}"] = {}
                else:
                    binned_ages = binned_ages_t[timestep_index]
                    binned_ages_true = binned_ages[numerators[timestep_index]]
                    counts_dict[f"age_counts_{timestep_index}"] = dict(
                        zip(*np.unique(binned_ages_true, return_counts=True)))
            age_true_counts = {}
            for age_counts_dict_key in counts_dict:
                for index in counts_dict[age_counts_dict_key]:
                    age_true_counts[index] = 0 if index not in age_true_counts else age_true_counts[index]
                    age_true_counts[index] += counts_dict[age_counts_dict_key][index]

            for index, age_str in enumerate(fpd.age_specific_channel_bins):
                if index not in age_true_counts:
                    results_dict[f"{channel}_{age_str}"] = 0
                else:
                    results_dict[f"{channel}_{age_str}"] = age_true_counts[index]
        return results_dict

    def track_mcpr(self):
        '''
        Track for purposes of calculating mCPR at the end of the timestep after all people are updated
        Not including LAM users in mCPR as this model counts all women passively using LAM but
        DHS data records only women who self-report LAM which is much lower.
        Follows the DHS definition of mCPR
        '''
        modern_methods = sc.findinds(list(self.pars['methods']['modern'].values()))
        method_age = (self.pars['method_age'] <= self.age)
        fecund_age = self.age < self.pars['age_limit_fecundity']
        denominator = method_age * fecund_age * self.is_female * (self.alive)
        numerator = np.isin(self.method, modern_methods)
        no_method_mcpr = np.sum((self.method == 0) * denominator)
        on_method_mcpr = np.sum(numerator * denominator)
        self.step_results['no_methods_mcpr'] += no_method_mcpr
        self.step_results['on_methods_mcpr'] += on_method_mcpr

        if self.pars['track_as']:
            as_result_dict = self.log_age_split(binned_ages_t=[self.age_by_group], channel='mcpr',
                                                numerators=[numerator], denominators=[denominator])
            for key in as_result_dict:
                self.step_results[key] = as_result_dict[key]
        return

    def track_cpr(self):
        '''
        Track for purposes of calculating newer ways to conceptualize contraceptive prevalence
        at the end of the timestep after all people are updated
        Includes women using any method of contraception, including LAM
        Denominator of possible users includes all women aged 15-49
        '''
        denominator = ((self.pars['method_age'] <= self.age) * (self.age < self.pars['age_limit_fecundity']) * (
                self.sex == 0) * (self.alive))
        numerator = self.method != 0
        no_method_cpr = np.sum((self.method == 0) * denominator)
        on_method_cpr = np.sum(numerator * denominator)
        self.step_results['no_methods_cpr'] += no_method_cpr
        self.step_results['on_methods_cpr'] += on_method_cpr

        if self.pars['track_as']:
            as_result_dict = self.log_age_split(binned_ages_t=[self.age_by_group], channel='cpr',
                                                numerators=[numerator], denominators=[denominator])
            for key in as_result_dict:
                self.step_results[key] = as_result_dict[key]
        return

    def track_acpr(self):
        '''
        Track for purposes of calculating newer ways to conceptualize contraceptive prevalence
        at the end of the timestep after all people are updated
        Denominator of possible users excludes pregnant women and those not sexually active in the last 4 weeks
        Used to compare new metrics of contraceptive prevalence and eventually unmet need to traditional mCPR definitions
        '''
        denominator = ((self.pars['method_age'] <= self.age) * (self.age < self.pars['age_limit_fecundity']) * (
                self.sex == 0) * (self.pregnant == 0) * (self.sexually_active == 1) * (self.alive))
        numerator = self.method != 0
        no_method_cpr = np.sum((self.method == 0) * denominator)
        on_method_cpr = np.sum(numerator * denominator)
        self.step_results['no_methods_acpr'] += no_method_cpr
        self.step_results['on_methods_acpr'] += on_method_cpr

        if self.pars['track_as']:
            as_result_dict = self.log_age_split(binned_ages_t=[self.age_by_group], channel='acpr',
                                                numerators=[numerator], denominators=[denominator])
            for key in as_result_dict:
                self.step_results[key] = as_result_dict[key]
        return

    def init_step_results(self):
        self.step_results = dict(
            deaths=0,
            births=0,
            stillbirths=0,
            total_births=0,
            short_intervals=0,
            secondary_births=0,
            maternal_deaths=0,
            infant_deaths=0,
            on_methods_mcpr=0,
            no_methods_mcpr=0,
            on_methods_cpr=0,
            no_methods_cpr=0,
            on_methods_acpr=0,
            no_methods_acpr=0,
            as_stillbirths=[],
            imr_numerator=[],
            imr_denominator=[],
            mmr_numerator=[],
            mmr_denominator=[],
            pp0to5=0,
            pp6to11=0,
            pp12to23=0,
            total_women_fecund=0,
            pregnancies=0,
            unintended_pregs=0,
            birthday_fraction=None,
            birth_bins={},
            age_bin_totals={},
            switching_annual={},
            switching_postpartum={},
            imr_age_by_group=[],
            mmr_age_by_group=[],
            stillbirth_ages=[]
        )

        if self.pars['track_as']:
            as_keys = dict(
                as_stillbirths=[],
                imr_numerator=[],
                imr_denominator=[],
                mmr_numerator=[],
                mmr_denominator=[],
                imr_age_by_group=[],
                mmr_age_by_group=[],
                stillbirth_ages=[]
            )
            self.step_results.update(as_keys)

            as_channels = ['acpr', 'cpr', 'mcpr', 'stillbirths', "births", "pregnancies"]
            for age_specific_channel in as_channels:
                for age_range in fpd.age_specific_channel_bins:
                    self.step_results[f"{age_specific_channel}_{age_range}"] = 0

        for key in fpd.age_bin_map.keys():
            self.step_results['birth_bins'][key] = 0
            self.step_results['age_bin_totals'][key] = 0

        m = len(self.pars['methods']['map'])

        def mm_zeros():
            ''' Return an array of m x m zeros '''
            return np.zeros((m, m), dtype=int)

        if self.pars['track_switching']:
            for key in fpd.method_age_map.keys():
                self.step_results['switching_annual'][key] = mm_zeros()
                self.step_results['switching_postpartum'][key] = mm_zeros()

            self.step_results['switching'] = dict(
                annual=mm_zeros(),
                postpartum=mm_zeros(),
            )

        return

    def update(self):
        '''
        Update the person's state for the given timestep.
        t is the time in the simulation in years (ie, 0-60), y is years of simulation (ie, 1960-2010)'''

        self.init_step_results()  # Initialize outputs
        alive_start = self.filter(self.alive)
        alive_start.check_mortality()  # Decide if person dies at this t in the simulation
        alive_check = self.filter(self.alive)  # Reselect live agents after exposure to general mortality

        # Update pregnancy with maternal mortality outcome
        preg = alive_check.filter(alive_check.pregnant)
        preg.check_delivery()  # Deliver with birth outcomes if reached pregnancy duration

        # Reselect for live agents after exposure to maternal mortality
        alive_now = self.filter(self.alive)
        fecund = alive_now.filter((alive_now.sex == 0) * (alive_now.age < alive_now.pars['age_limit_fecundity']))
        nonpreg = fecund.filter(~fecund.pregnant)
        lact = fecund.filter(fecund.lactating)
        if self.pars['restrict_method_use'] == 1:
            methods = nonpreg.filter((nonpreg.age >= nonpreg.fated_debut) * (nonpreg.months_inactive < 12))
        else:
            methods = nonpreg.filter(nonpreg.age >= self.pars['method_age'])

        # Check if has reached their age at first partnership and set partnered attribute to True.
        # TODO: decide whether this is the optimal place to perform this update, and how it may interact with sexual debut age
        alive_now.check_partnership()

        # Update everything else
        preg.update_pregnancy()  # Advance gestation in timestep, handle miscarriage
        nonpreg.check_sexually_active()
        methods.update_methods()
        nonpreg.update_postpartum()  # Updates postpartum counter if postpartum
        lact.update_breastfeeding()
        nonpreg.check_lam()
        nonpreg.check_conception()  # Decide if conceives and initialize gestation counter at 0

        # Update education
        if self.pars['education'] is not None:
            alive_now_f = self.filter(self.is_female)
            alive_now_f.start_education()  # Check if anyone needs to start school
            alive_now_f.update_education()  # Advance attainment, determine who reaches their objective, who dropouts, who has their education interrupted
            alive_now_f.resume_education()  # Determine who goes back to school after an interruption
            alive_now_f.graduate()  # Check if anyone achieves their education goal

        # Update results
        fecund.update_age_bin_totals()
        self.track_mcpr()
        self.track_cpr()
        self.track_acpr()
        age_min = self.age >= 15  # CK: TODO: remove hardcoding
        age_max = self.age < self.pars['age_limit_fecundity']
        self.step_results['total_women_fecund'] = np.sum(self.is_female * age_min * age_max)

        # Age person at end of timestep after tabulating results
        alive_now.update_age()  # Important to keep this here so birth spacing gets recorded accurately

        # Storing ages by method age group
        age_bins = [0] + [max(fpd.age_specific_channel_bins[key]) for key in fpd.age_specific_channel_bins]
        self.age_by_group = np.digitize(self.age, age_bins) - 1

        return self.step_results


