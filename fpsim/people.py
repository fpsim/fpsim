"""
Defines the People class
"""

# %% Imports
import numpy as np  # Needed for a few things not provided by pl
import sciris as sc
from . import utils as fpu
from . import defaults as fpd
from . import base as fpb
from . import empowerment as fpemp
from . import subnational as fpsn

# Specify all externally visible things this file defines
__all__ = ['People']


# %% Define classes

class People(fpb.BasePeople):
    """
    Class for all the people in the simulation.
    """

    def __init__(self, pars, n=None, age=None, sex=None, method=None, **kwargs):

        # Initialization
        super().__init__()

        self.pars = pars  # Set parameters
        if n is None:
            n = int(self.pars['n_agents'])

        # Set default states
        self.states = fpd.person_defaults
        for state_name, state in self.states.items():
            self[state_name] = state.new(n)

        # Overwrite some states with alternative values
        self.uid = np.arange(n)

        # Basic demographics
        _age, _sex = self.get_age_sex(n)
        if self.pars['use_subnational']==False:
            _urban = self.get_urban(n)
        else:
            _urban = fpsn.get_urban_init_vals(self)
        if age is None: age = _age
        if sex is None: sex = _sex

        self.age = self.states['age'].new(n, age)  # Age of the person in years
        self.sex = self.states['sex'].new(n, sex)  # Female (0) or male (1)
        self.urban = self.states['urban'].new(n, _urban)  # Urban (1) or rural (0)

        # Contraceptive use
        if method is not None:
            self.method = method  # Contraceptive method 0-9, see pars['methods']['map'], excludes LAM
        self.barrier = fpu.n_multinomial(self.pars['barriers'][:], n)

        # Parameters on sexual and reproductive history
        self.fertile = fpu.n_binomial(1 - self.pars['primary_infertility'], n)
        # Default initialization for fated_debut; subnational debut initialized in subnational.py otherwise
        if self.pars['use_subnational']==False:
            self.fated_debut = self.pars['debut_age']['ages'][fpu.n_multinomial(self.pars['debut_age']['probs'], n)]
        else:
            self.fated_debut = fpsn.get_debut_init_vals(self)

        # Fecundity variation
        fv = [self.pars['fecundity_var_low'], self.pars['fecundity_var_high']]
        fac = (fv[1] - fv[0]) + fv[0]  # Stretch fecundity by a factor bounded by [f_var[0], f_var[1]]
        self.personal_fecundity = np.random.random(n) * fac


        if self.pars['use_partnership']:
            fpemp.init_partnership_states(self)

        if self.pars['use_empowerment']:
            fpemp.init_empowerment_states(self)

        if self.pars['use_education']:
            fpemp.init_education_states(self)

        if self.pars['use_subnational']:
            fpsn.init_regional_states(self)

        # Store keys
        self._keys = [state.name for state in fpd.person_defaults.values()]

        return

    def get_age_sex(self, n):
        """
        For a sample of n ex nihilo people, return arrays of n ages and sexes
            :param n: number of people
            :return: arrays of length n containing their ages and sexes
        """
        pyramid = self.pars['age_pyramid']
        m_frac = pyramid[:, 1].sum() / pyramid[:, 1:3].sum()

        ages = np.zeros(n)
        sexes = np.random.random(n) < m_frac  # Pick the sex based on the fraction of men vs. women
        f_inds = sc.findinds(sexes == 0)
        m_inds = sc.findinds(sexes == 1)

        age_data_min = pyramid[:, 0]
        age_data_max = np.append(pyramid[1:, 0], self.pars['max_age'])
        age_data_range = age_data_max - age_data_min
        for i, inds in enumerate([m_inds, f_inds]):
            if len(inds):
                age_data_prob = pyramid[:, i + 1]
                age_data_prob = age_data_prob / age_data_prob.sum()  # Ensure it sums to 1
                age_bins = fpu.n_multinomial(age_data_prob, len(inds))  # Choose age bins
                ages[inds] = age_data_min[age_bins] + age_data_range[age_bins] * np.random.random(
                    len(inds))  # Uniformly distribute within this age bin

        return ages, sexes

    def get_urban(self, n):
        """ Get initial distribution of urban """

        urban = np.ones(n, dtype=bool)
        urban_prop = self.pars['urban_prop']
        urban = fpu.n_binomial(urban_prop, n)

        return urban

    def birthday_filter(self):
        """
        Returns a filtered ppl object of people who celebrated their bdays, useful for methods that update
        annualy, but not based on a calendar year, rather every year on an agent's bday."""
        age_diff = self.ceil_age - self.age
        whole_years = ((age_diff < (1 / fpd.mpy)) * (age_diff > 0))
        return self.filter(whole_years)

    def update_method(self):
        """
        Uses a switching matrix from DHS data to decide based on a person's original method their probability of
        changing to a new method and assigns them the new method. This currently allows switching on whole calendar
        years to enter function. Matrix serves as an initiation, discontinuation, continuation, and switching matrix.
        Transition probabilities are for 1 year and only for women who have not given birth within the last 6 months.
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
            match_low_high = fpu.match_ages(self.age, age_low, age_high)
            for m in method_map.values():
                match_m = (orig_methods == m)
                match = match_m * match_low_high
                this_method = self.filter(match)
                if len(this_method):
                    old_method = this_method.method.copy()

                    matrix = annual[key]
                    choices = matrix[m]
                    choices = choices/choices.sum()
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
            match_low_high = fpu.match_ages(self.age, age_low, age_high)
            low_parity = (self.parity < self.pars['high_parity'])
            high_parity = (self.parity >= self.pars['high_parity'])
            match = (self.postpartum * postpartum1 * match_low_high * low_parity)
            match_high_parity = (self.postpartum * postpartum1 * match_low_high * high_parity)
            this_method = self.filter(match)
            if len(this_method):
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
            match_low_high = fpu.match_ages(self.age, age_low, age_high)
            match_postpartum_age = self.postpartum * postpartum6 * match_low_high
            for m in methods_map.values():
                match_m = (orig_methods == m)
                match = match_m * match_postpartum_age
                this_method = self.filter(match)
                if len(this_method):
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
            postpartum = self.postpartum * (self.postpartum_dur <= 6)
            pp = self.filter(postpartum)
            non_pp = self.filter(~postpartum)

            if len(pp):
                pp.update_method_pp()  # Update method for postpartum women

            age_diff = non_pp.ceil_age - non_pp.age
            whole_years = ((age_diff < (1 / fpd.mpy)) * (age_diff > 0))
            birthdays = non_pp.filter(whole_years)
            if len(birthdays):
                birthdays.update_method()

        return

    def check_mortality(self):
        """ Decide if person dies at a timestep """

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
        """
        Decide if an agent has reached their age at first partnership. Age-based data from DHS.
        """

        is_not_partnered = self.partnered == 0
        reached_partnership_age = self.age >= self.partnership_age
        first_timers = self.filter(is_not_partnered * reached_partnership_age)
        first_timers.partnered = True

    def check_sexually_active(self):
        """
        Decide if agent is sexually active based either on month postpartum or age if
        not postpartum.  Postpartum and general age-based data from DHS.
        """
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
        """
        Decide if person (female) becomes pregnant at a timestep.
        """
        all_ppl = self.unfilter()  # For complex array operations
        active = self.filter(self.sexually_active * self.fertile)
        lam = active.filter(active.lam)
        nonlam = active.filter(~active.lam)
        preg_probs = np.zeros(len(all_ppl))  # Use full array

        # Find monthly probability of pregnancy based on fecundity and use of contraception including LAM - from data
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

        # Adjust for probability of exposure to pregnancy episode at this timestep based on age and parity.
        # This encapsulates background factors and is experimental and tunable.
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
        """
        Update the selected agents to be pregnant. This also sets their method to no contraception.
        """
        pregdur = [self.pars['preg_dur_low'], self.pars['preg_dur_high']]
        self.pregnant = True
        self.gestation = 1  # Start the counter at 1
        self.preg_dur = np.random.randint(pregdur[0], pregdur[1] + 1, size=len(self))  # Duration of this pregnancy
        self.postpartum = False
        self.postpartum_dur = 0
        self.reset_breastfeeding()  # Stop lactating if becoming pregnant
        self.method = 0  # Not using contraception during pregnancy
        return

    def check_lam(self):
        """
        Check to see if postpartum agent meets criteria for LAM in this time step
        """
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
        """
        Track breastfeeding, and update time of breastfeeding for individual pregnancy.
        Agents are randomly assigned a duration value based on a gumbel distribution drawn from the 2018 DHS variable for breastfeeding months. The mean (mu) and the std dev (beta) are both drawn from that distribution in the DHS data.
        """
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
        """
        Track duration of extended postpartum period (0-24 months after birth).
        Only enter this function if agent is postpartum.
        """

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
        """ Advance pregnancy in time and check for miscarriage """

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
        """
        Stop breastfeeding, calculate total lifetime duration so far, and reset lactation episode to zero
        """
        self.lactating = False
        self.breastfeed_dur_total += self.breastfeed_dur
        self.breastfeed_dur = 0
        return

    def check_maternal_mortality(self):
        """
        Check for probability of maternal mortality
        """
        prob = self.pars['mortality_probs']['maternal'] * self.pars['maternal_mortality_factor']
        is_death = self.binomial(prob)
        death = self.filter(is_death)
        death.alive = False
        self.step_results['maternal_deaths'] += len(death)
        self.step_results['deaths'] += len(death)
        return death

    def check_infant_mortality(self):
        """
        Check for probability of infant mortality (death < 1 year of age)
        """
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
        """
        Decide if pregnant woman gives birth and explore maternal mortality and child mortality
        """

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
                match_low_high = fpu.match_ages(live_age, age_low, age_high)
                birth_bins = np.sum(match_low_high)
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
        """
        Advance age in the simulation
        """
        self.age += self.pars['timestep'] / fpd.mpy  # Age the person for the next timestep
        self.age = np.minimum(self.age, self.pars['max_age'])

        return

    def update_age_bin_totals(self):
        """
        Count how many total live women in each 5-year age bin 10-50, for tabulating ASFR
        """
        for key, (age_low, age_high) in fpd.age_bin_map.items():
            this_age_bin = self.filter((fpu.match_ages(self.age, age_low, age_high)))
            self.step_results['age_bin_totals'][key] += len(this_age_bin)
        return

    def log_age_split(self, binned_ages_t, channel, numerators, denominators=None):
        """
        Method called if age-specific results are being tracked. Separates results by age.
        """
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
        """
        Track for purposes of calculating mCPR at the end of the timestep after all people are updated
        Not including LAM users in mCPR as this model counts all women passively using LAM but
        DHS data records only women who self-report LAM which is much lower.
        Follows the DHS definition of mCPR
        """
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
        """
        Track for purposes of calculating newer ways to conceptualize contraceptive prevalence
        at the end of the timestep after all people are updated
        Includes women using any method of contraception, including LAM
        Denominator of possible users includes all women aged 15-49
        """
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
        """
        Track for purposes of calculating newer ways to conceptualize contraceptive prevalence
        at the end of the timestep after all people are updated
        Denominator of possible users excludes pregnant women and those not sexually active in the last 4 weeks
        Used to compare new metrics of contraceptive prevalence and eventually unmet need to traditional mCPR definitions
        """
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
        """
        Update the person's state for the given timestep.
        t is the time in the simulation in years (ie, 0-60), y is years of simulation (ie, 1960-2010)
        """

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
        if self.pars['use_education']:
            alive_now_f = self.filter(self.is_female)
            fpemp.update_education(alive_now_f)

        # Update empowerment on bdays (unless there's a different mechanism to update)
        if self.pars['use_empowerment']:
            birthdays = self.filter(self.is_female).birthday_filter()
            if len(birthdays):
               fpemp.update_empowerment(birthdays)

        # Update results
        fecund.update_age_bin_totals()
        self.track_mcpr()
        self.track_cpr()
        self.track_acpr()
        age_min = self.age >= fpd.min_age
        age_max = self.age < self.pars['age_limit_fecundity']

        self.step_results['total_women_fecund'] = np.sum(self.is_female * age_min * age_max)

        # Age person at end of timestep after tabulating results
        alive_now.update_age()  # Important to keep this here so birth spacing gets recorded accurately

        # Storing ages by method age group
        age_bins = [0] + [max(fpd.age_specific_channel_bins[key]) for key in fpd.age_specific_channel_bins]
        self.age_by_group = np.digitize(self.age, age_bins) - 1

        return self.step_results
