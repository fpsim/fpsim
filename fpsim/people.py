"""
Defines the People class
"""

# %% Imports
import numpy as np  # Needed for a few things not provided by pl
import sciris as sc
from . import utils as fpu
from . import defaults as fpd
from . import base as fpb
from . import demographics as fpdmg
from . import subnational as fpsn

# Specify all externally visible things this file defines
__all__ = ['People']


# %% Define classes

class People(fpb.BasePeople):
    """
    Class for all the people in the simulation.
    """

    def __init__(self, pars, n=None, age=None, sex=None,
                 empowerment_module=None, education_module=None, **kwargs):

        # Initialization
        super().__init__(**kwargs)

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
        if not self.pars['use_subnational']:
            _urban = self.get_urban(n)
        else:
            _urban = fpsn.get_urban_init_vals(self)
        if age is None: age = _age
        if sex is None: sex = _sex

        self.age = self.states['age'].new(n, age)  # Age of the person in years
        self.sex = self.states['sex'].new(n, sex)  # Female (0) or male (1)
        self.urban = self.states['urban'].new(n, _urban)  # Urban (1) or rural (0)

        # Parameters on sexual and reproductive history
        self.fertile = fpu.n_binomial(1 - self.pars['primary_infertility'], n)

        # Fertility intent
        has_intent = "fertility_intent"
        self.fertility_intent   = self.states[has_intent].new(n, fpd.person_defaults[has_intent].val)
        self.categorical_intent = self.states["categorical_intent"].new(n, "no")
        # Update distribution of fertility intent with location-specific values if it is present in self.pars
        self.update_fertility_intent(n)

        # Intent to use contraception
        has_intent = "intent_to_use"
        self.intent_to_use = self.states[has_intent].new(n, fpd.person_defaults[has_intent].val)
        # Update distribution of fertility intent if it is present in self.pars
        self.update_intent_to_use(n)

        self.wealthquintile = self.states["wealthquintile"].new(n, fpd.person_defaults["wealthquintile"].val)
        self.update_wealthquintile(n)

        # Default initialization for fated_debut; subnational debut initialized in subnational.py otherwise
        if not self.pars['use_subnational']:
            self.fated_debut = self.pars['debut_age']['ages'][fpu.n_multinomial(self.pars['debut_age']['probs'], n)]
        else:
            self.fated_debut = fpsn.get_debut_init_vals(self)

        # Fecundity variation
        fv = [self.pars['fecundity_var_low'], self.pars['fecundity_var_high']]
        fac = (fv[1] - fv[0]) + fv[0]  # Stretch fecundity by a factor bounded by [f_var[0], f_var[1]]
        self.personal_fecundity = np.random.random(n) * fac

        # Initialise ti_contra based on age and fated debut
        self.update_time_to_choose()

        # Empowerment and education
        self.empowerment_module = empowerment_module
        self.education_module = education_module
        if self.empowerment_module is not None:
            self.empowerment_module.initialize(self.filter(self.is_female))

        if self.education_module is not None:
            self.education_module.initialize(self)

        # Partnership
        if self.pars['use_partnership']:
            fpdmg.init_partnership_states(self)

        # Handle circular buffer to keep track of historical data
        self.longitude = sc.objdict()
        self.initialize_circular_buffer()

        # Once all the other metric are initialized, determine initial contraceptive use
        self.contraception_module = None  # Set below
        self.barrier = fpu.n_multinomial(self.pars['barriers'][:], n)

        # Store keys
        self._keys = [s.name for s in self.states.values()]

        if self.pars['use_subnational']:
            fpsn.init_regional_states(self)
            fpsn.init_regional_states(self)

        return

    def initialize_circular_buffer(self):
        # Initialize circular buffers to track longitudinal data
        longitude_keys = fpd.longitude_keys
        # NOTE: by default the history array/circular buffer is initialised with constant
        # values. We could potentially initialise the buffer
        # with the data from a previous simulation.

        for key in longitude_keys:
            current = getattr(self, key)  # Current value of this attribute
            self.longitude[key] = np.full((self.n, self.tiperyear), current[0])
        return

    @property
    def dt(self):
        return self.pars['timestep'] / fpd.mpy

    @property
    def tiperyear(self):
        return self.pars['tiperyear']

    @property
    def yei(self):
        """
        The index of year-ending as of the same date expressed in ti
        or 12-months ago as of the same date.
        """
        return (self.ti + 1) % self.tiperyear

    def get_longitudinal_state(self, state_name):
        """
        Extract values of one of the longitudinal state/attributes (aka states with history)

        Arguments:
            state_name (str): the name of the state or attribute that we are extracting

        Returns:
            state_vals (np.arr):  array of the ppl.term values from one year prior to current timestep
        """
        # Calculate correct index for data 1 year prior
        if len(self):
            state_vals = self['longitude'][state_name][self.inds, self.yei]
        else:
            state_vals = np.empty((0,))
        return state_vals

    def get_urban(self, n):
        """ Get initial distribution of urban """
        urban_prop = self.pars['urban_prop']
        urban = fpu.n_binomial(urban_prop, n)
        return urban

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

    def update_fertility_intent(self, n):
        if self.pars['fertility_intent'] is None:
            return
        self.update_fertility_intent_by_age()
        return

    def update_intent_to_use(self, n):
        if self.pars['intent_to_use'] is None:
            return
        self.update_intent_to_use_by_age()
        return

    def update_wealthquintile(self, n):
        if self.pars['wealth_quintile'] is None:
            return
        wq_probs = self.pars['wealth_quintile']['percent']
        vals = np.random.choice(len(wq_probs), size=n, p=wq_probs)+1
        self.wealthquintile = vals
        return

    def update_time_to_choose(self):
        """
        Initialise the counter to determine when girls/women will have to first choose a method.
        """
        inds = sc.findinds((self.sex == 0) * (self.age < self.pars['age_limit_fecundity']))
        time_to_debut = (self.fated_debut[inds]-self.age[inds])/self.dt
        self.ti_contra[inds] = np.maximum(time_to_debut, 0)
        # Validation
        time_to_set_contra = self.ti_contra[inds] == 0
        if not np.array_equal(((self.age[inds] - self.fated_debut[inds]) > -self.dt), time_to_set_contra):
            errormsg = 'Should be choosing contraception for everyone past fated debut age.'
            raise ValueError(errormsg)
        return

    def decide_contraception(self, ti=None, year=None, contraception_module=None):
        """
        Decide who will start using contraception, when, which contraception method and the
        duration on that method. This method is called by the simulation to initialise the
        people object at the beginning of the simulation and new people born during the simulation.

        #TODO: rename to something that indicates this method is used for initialisation
        """
        fecund = self.filter((self.sex == 0) * (self.age < self.pars['age_limit_fecundity']))
        # NOTE: PSL: This line effectively "initialises" whether a woman is sexually active or not.
        # Because of the current initialisation flow, it's not possible to initialise the
        # sexually_active state in the init constructor.
        fecund.check_sexually_active()
        fecund.update_time_to_choose()

        # Check whether have reached the time to choose
        time_to_set_contra = fecund.ti_contra == 0
        contra_choosers = fecund.filter(time_to_set_contra)

        if contraception_module is not None:
            self.contraception_module = contraception_module
            contra_choosers.on_contra = contraception_module.get_contra_users(contra_choosers, year=year, ti=ti, tiperyear=self.tiperyear)
            oc = contra_choosers.filter(contra_choosers.on_contra)
            oc.method = contraception_module.init_method_dist(oc)
            oc.ever_used_contra = 1
            method_dur = contraception_module.set_dur_method(contra_choosers)
            contra_choosers.ti_contra = ti + method_dur
        return

    def update_fertility_intent_by_age(self):
        """
        In the absence of other factors, fertilty intent changes as a function of age
        each year on a woman’s birthday
        """
        intent_pars = self.pars['fertility_intent']

        f_inds = sc.findinds(self.is_female)
        f_ages = self.age[f_inds]
        age_inds = fpu.digitize_ages_1yr(f_ages)
        for age in intent_pars.keys():
            aged_x_inds = f_inds[age_inds == age]
            fi_cats = list(intent_pars[age].keys())  # all ages have the same intent categories
            probs = np.array(list(intent_pars[age].values()))
            ci = np.random.choice(fi_cats, aged_x_inds.size, p=probs)
            self.categorical_intent[aged_x_inds] = ci

        self.fertility_intent[sc.findinds(self.categorical_intent == "yes")] = True
        self.fertility_intent[sc.findinds((self.categorical_intent == "no") |
                                          (self.categorical_intent == "cannot"))] = False
        return

    def update_intent_to_use_by_age(self):
        """
        In the absence of other factors, intent to use contraception
        can change as a function of age each year on a woman’s birthday.

        This function is also used to initialise the State intent_to_use
        """
        intent_pars = self.pars['intent_to_use']

        f_inds = sc.findinds(self.is_female)
        f_ages = self.age[f_inds]
        age_inds = fpu.digitize_ages_1yr(f_ages)

        for age in intent_pars.keys():
            f_aged_x_inds = f_inds[age_inds == age]  # indices of women of a given age
            prob = intent_pars[age][1]  # Get the probability of having intent
            self.intent_to_use[f_aged_x_inds] = fpu.n_binomial(prob, len(f_aged_x_inds))

        return

    def update_method(self, year=None, ti=None):
        """ Inputs: filtered people, only includes those for whom it's time to update """
        cm = self.contraception_module
        if year is None: year = self.y
        if ti is None: ti = self.ti
        if cm is not None:

            # If people are 1 or 6m postpartum, we use different parameters for updating their contraceptive decisions
            is_pp1 = (self.postpartum_dur == 1)
            is_pp6 = (self.postpartum_dur == 6) & ~self.on_contra  # They may have decided to use contraception after 1m
            pp0 = self.filter(~(is_pp1 | is_pp6))
            pp1 = self.filter(is_pp1)
            pp6 = self.filter(is_pp6)

            # Update choices for people who aren't postpartum
            if len(pp0):

                # If force_choose is True, then all non-users will be made to pick a method
                if cm.pars['force_choose']:
                    must_use = pp0.filter(~pp0.on_contra)
                    choosers = pp0.filter(pp0.on_contra)

                    if len(must_use):
                        must_use.on_contra = True
                        pp0.step_results['contra_access'] += len(must_use)
                        must_use.method = cm.choose_method(must_use)
                        must_use.ever_used_contra = 1
                        pp0.step_results['new_users'] += np.count_nonzero(must_use.method)

                else:
                    choosers = pp0

                # Get previous users and see whether they will switch methods or stop using
                if len(choosers):

                    choosers.on_contra = cm.get_contra_users(choosers, year=year, ti=ti, tiperyear=self.pars['tiperyear'])
                    choosers.ever_used_contra = choosers.ever_used_contra | choosers.on_contra

                    # Divide people into those that keep using contraception vs those that stop
                    switching_contra = choosers.filter(choosers.on_contra)
                    stopping_contra = choosers.filter(~choosers.on_contra)
                    pp0.step_results['contra_access'] += len(switching_contra)

                    # For those who keep using, choose their next method
                    if len(switching_contra):
                        switching_contra.method = cm.choose_method(switching_contra)
                        choosers.step_results['new_users'] += np.count_nonzero(switching_contra.method)

                    # For those who stop using, set method to zero
                    if len(stopping_contra):
                        stopping_contra.method = 0

                # Validate
                n_methods = len(pp0.contraception_module.methods)
                invalid_vals = (pp0.method >= n_methods) * (pp0.method < 0)
                if invalid_vals.any():
                    errormsg = f'Invalid method set: ti={pp0.ti}, inds={invalid_vals.nonzero()[-1]}'
                    raise ValueError(errormsg)

            # Now update choices for postpartum people. Logic here is simpler because none of these
            # people should be using contraception currently. We first check that's the case, then
            # have them choose their contraception options.
            ppdict = {'pp1': pp1, 'pp6': pp6}
            for event, pp in ppdict.items():
                if len(pp):
                    if pp.on_contra.any():
                        errormsg = 'Postpartum women should not currently be using contraception.'
                        raise ValueError(errormsg)
                    pp.on_contra = cm.get_contra_users(pp, year=year, event=event, ti=ti, tiperyear=self.pars['tiperyear'])
                    on_contra = pp.filter(pp.on_contra)
                    off_contra = pp.filter(~pp.on_contra)
                    pp.step_results['contra_access'] += len(on_contra)

                    # Set method for those who use contraception
                    if len(on_contra):
                        on_contra.method = cm.choose_method(on_contra, event=event)
                        on_contra.ever_used_contra = 1

                    if len(off_contra):
                        off_contra.method = 0
                        if event == 'pp1':  # For women 1m postpartum, choose again when they are 6 months pp
                            off_contra.ti_contra = ti + 5

            # Set duration of use for everyone, and reset the time they'll next update
            durs_fixed = (self.postpartum_dur == 1) & (self.method == 0)
            update_durs = self.filter(~durs_fixed)
            update_durs.ti_contra = ti + cm.set_dur_method(update_durs)

        return

    def decide_death_outcome(self):
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

    def start_partnership(self):
        """
        Decide if an agent has reached their age at first partnership. Age-based data from DHS.
        """

        is_not_partnered = self.partnered == 0
        reached_partnership_age = self.age >= self.partnership_age
        first_timers = self.filter(is_not_partnered * reached_partnership_age)
        first_timers.partnered = True
        return

    def check_sexually_active(self):
        """
        Decide if agent is sexually active based either time-on-postpartum month
        or their age if not postpartum.

        Agents can revert to active or not active each timestep. Postpartum and
        general age-based data from DHS.
        """
        # Set postpartum probabilities
        match_low = self.postpartum_dur >= 0
        match_high = self.postpartum_dur <= self.pars['postpartum_dur']
        pp_match = self.postpartum * match_low * match_high
        non_pp_match = ((self.age >= self.fated_debut) * (~pp_match))
        pp = self.filter(pp_match)
        non_pp = self.filter(non_pp_match)

        # Adjust for postpartum women's birth spacing preferences
        if len(pp):
            pref = self.pars['spacing_pref']  # Shorten since used a lot
            spacing_bins = pp.postpartum_dur / pref['interval']  # Main calculation: divide the duration by the interval
            spacing_bins = np.array(np.minimum(spacing_bins, pref['n_bins']), dtype=int)  # Bound by longest bin
            probs_pp = self.pars['sexual_activity_pp']['percent_active'][pp.postpartum_dur]
            # Adjust the probability: check the overall probability with print(pref['preference'][spacing_bins].mean())
            probs_pp *= pref['preference'][spacing_bins]
            pp.sexually_active = fpu.binomial_arr(probs_pp)

        # Set non-postpartum probabilities
        if len(non_pp):
            probs_non_pp = self.pars['sexual_activity'][non_pp.int_age]
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

        # Get each woman's degree of protection against conception based on her contraception or LAM
        eff_array = np.array([m.efficacy for m in self.contraception_module.methods.values()])
        method_eff = eff_array[nonlam.method]
        lam_eff = pars['LAM_efficacy']

        # Change to a monthly probability and set pregnancy probabilities
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
        self.step_results['method_failures'] += len(unintended)  # unintended pregnancies due to method failure

        # Check for abortion
        is_abort = conceived.binomial(pars['abortion_prob'])
        abort = conceived.filter(is_abort)
        preg = conceived.filter(~is_abort)

        # Update states
        n_aborts = len(abort)
        self.step_results['abortions'] = n_aborts
        if n_aborts:
            all_ppl = self.unfilter()
            for cum_aborts in np.unique(abort.abortion):
                all_ppl.abortion_ages[abort.inds, cum_aborts] = abort.age
            abort.postpartum = False
            abort.abortion += 1  # Add 1 to number of abortions agent has had
            abort.postpartum_dur = 0

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
        self.on_contra = False  # Not using contraception during pregnancy
        self.method = 0  # Method zero due to non-use
        return

    def check_lam(self):
        """
        Check to see if postpartum agent meets criteria for
        Lactation amenorrhea method (LAM) LAM in this time step
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
        Agents are randomly assigned a duration value based on a gumbel distribution drawn
        from the 2018 DHS variable for breastfeeding months.
        The mean (mu) and the std dev (beta) are both drawn from that distribution in the DHS data.
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
        postpart = self.filter(self.postpartum)
        for key, (pp_low, pp_high) in fpd.postpartum_map.items():
            this_pp_bin = postpart.filter((postpart.postpartum_dur >= pp_low) * (postpart.postpartum_dur < pp_high))
            self.step_results[key] += len(this_pp_bin)
        postpart.postpartum_dur += self.pars['timestep']

        return

    def progress_pregnancy(self):
        """ Advance pregnancy in time and check for miscarriage """

        preg = self.filter(self.pregnant)
        preg.gestation += self.pars['timestep']

        # Check for miscarriage at the end of the first trimester
        end_first_tri = preg.filter(preg.gestation == self.pars['end_first_tri'])
        miscarriage_probs = self.pars['miscarriage_rates'][end_first_tri.int_age_clip]
        miscarriage = end_first_tri.binomial(miscarriage_probs, as_filter=True)

        # Reset states and track miscarriages
        n_miscarriages = len(miscarriage)
        self.step_results['miscarriages'] = n_miscarriages

        if n_miscarriages:
            all_ppl = self.unfilter()
            for cum_miscarriages in np.unique(miscarriage.miscarriage):
                all_ppl.miscarriage_ages[miscarriage.inds, cum_miscarriages] = miscarriage.age
            miscarriage.pregnant = False
            miscarriage.miscarriage += 1  # Add 1 to number of miscarriages agent has had
            miscarriage.postpartum = False
            miscarriage.gestation = 0  # Reset gestation counter
            miscarriage.ti_contra = self.ti+1  # Update contraceptive choices

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
        death.ti_contra = self.ti + 1  # Trigger update to contraceptive choices following infant death
        return death

    def process_delivery(self):
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
            deliv.ti_contra = self.ti + 1  # Trigger a call to re-evaluate whether to use contraception when 1month pp

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

            # Record ages of agents when live births / stillbirths occur
            all_ppl = self.unfilter()
            live = deliv.filter(~is_stillborn)
            for parity in np.unique(live.parity):
                inds = live.inds[live.parity == parity]
                all_ppl.birth_ages[inds, parity] = all_ppl.age[inds]
                if parity == 0: all_ppl.first_birth_age[inds] = all_ppl.age[inds]
            for parity in np.unique(stillborn.parity):
                inds = stillborn.inds[stillborn.parity == parity]
                all_ppl.stillborn_ages[inds, parity] = all_ppl.age[inds]

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
        modern_methods_num = [idx for idx, m in enumerate(self.contraception_module.methods.values()) if m.modern]
        method_age = (self.pars['method_age'] <= self.age)
        fecund_age = self.age < self.pars['age_limit_fecundity']
        denominator = method_age * fecund_age * self.is_female * (self.alive)
        numerator = np.isin(self.method, modern_methods_num)
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

    def birthday_filter(self, int_age=None):
        """
        Returns a filtered ppl object of people who celebrated their bdays, useful for methods that update
        annualy, but not based on a calendar year, rather every year on an agent's bday."""
        if int_age is None:
            int_age = self.int_age

        age_diff = int_age - self.age
        had_bday = (age_diff <= (self.pars['timestep'] / fpd.mpy))
        return self.filter(had_bday)

    def reset_step_results(self):
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
            contra_access=0,
            new_users=0,
            ever_used_contra=0,
            urban_women=0,
            as_stillbirths=[],
            imr_numerator=[],
            imr_denominator=[],
            mmr_numerator=[],
            mmr_denominator=[],
            pp0to5=0,
            pp6to11=0,
            pp12to23=0,
            parity0to1=0,
            parity2to3=0,
            parity4to5=0,
            parity6plus=0,
            wq1=0,
            wq2=0,
            wq3=0,
            wq4=0,
            wq5=0,
            total_women_fecund=0,
            pregnancies=0,
            method_failures=0,
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

        return

    def update_history_buffer(self):
        """
        Updates longitudinal params in people object
        """

        # Calculate column index in which to store current vals
        index = self.ti % self.tiperyear

        # Store the current params in people.longitude object
        for key in self.longitude.keys():
            self.longitude[key][:, index] = getattr(self, key)

        return

    def step(self):
        """
        Perform all updates to people within a single timestep
        """
        self.reset_step_results()  # Allocate an 'empty' dictionary for the outputs of this time step

        alive_start = self.filter(self.alive)
        alive_start.decide_death_outcome()     # Decide if person dies at this t in the simulation
        alive_check = self.filter(self.alive)  # Reselect live agents after exposure to general mortality

        # Update pregnancy with maternal mortality outcome
        preg = alive_check.filter(alive_check.pregnant)
        preg.process_delivery()  # Deliver with birth outcomes if reached pregnancy duration

        # Reselect for live agents after exposure to maternal mortality
        alive_now = self.filter(self.alive)
        fecund = alive_now.filter((alive_now.sex == 0) * (alive_now.age < alive_now.pars['age_limit_fecundity']))

        nonpreg = fecund.filter(~fecund.pregnant)
        lact = fecund.filter(fecund.lactating)

        # Update empowerment states, and empowerment-related states
        alive_now_f = self.filter(self.is_female & self.alive)

        if self.empowerment_module is not None: alive_now_f.step_empowerment()
        if self.education_module is not None: alive_now_f.step_education()

        # Figure out who to update methods for
        ready = nonpreg.filter(nonpreg.ti_contra <= self.ti)

        # Check if has reached their age at first partnership and set partnered attribute to True.
        # TODO: decide whether this is the optimal place to perform this update, and how it may interact with sexual debut age
        alive_now.start_partnership()

        # Complete all updates. Note that these happen in a particular order!
        preg.progress_pregnancy()  # Advance gestation in timestep, handle miscarriage
        nonpreg.check_sexually_active()

        # Update methods for those who are elgible
        if len(ready):
            ready.update_method()

        methods_ok = np.array_equal(self.on_contra.nonzero()[-1], self.method.nonzero()[-1])
        if not methods_ok:
            errormsg = 'Agents not using contraception are not the same as agents who are using None method'
            raise ValueError(errormsg)

        nonpreg.update_postpartum()  # Updates postpartum counter if postpartum
        lact.update_breastfeeding()
        nonpreg.check_lam()
        nonpreg.check_conception()  # Decide if conceives and initialize gestation counter at 0

        # Populate the ciruclar buffers with data from Peoples states that
        # are needed for methods/classes that use historical data (eg, need previous year's data)
        self.update_history_buffer()

        # Update results
        fecund.update_age_bin_totals()

        # Add check for ti contra
        if (self.ti_contra < 0).any():
            errormsg = f'Invalid values for ti_contra at timestep {self.ti}'
            raise ValueError(errormsg)

        return

    def step_empowerment(self):
        eligible = self.filter(self.is_dhs_age)
        # Women who just turned 15 get assigned a value based on empowerment probs
        bday_15 = eligible.birthday_filter(int_age=int(fpd.min_age))
        if len(bday_15):
            self.empowerment_module.update_empwr_states(bday_15)
        # Update states on her bday, based on coefficients
        bday = eligible.birthday_filter()
        if len(bday):
            # The empowerment module will update the empowerment states and intent to use
            self.empowerment_module.update(bday)
            # Update fertility intent on her bday, together with empowerment updates
            bday.update_fertility_intent_by_age()
        return

    def step_education(self):
        self.education_module.update(self)
        return

    def step_age(self):
        """
        Advance people's age at at end of timestep after tabulating results
        and update the age_by_group, based on the new age distribution to
        quantify results in the next time step.
        """
        alive_now = self.filter(self.alive)
        # Age person at end of timestep after tabulating results
        alive_now.update_age()  # Important to keep this here so birth spacing gets recorded accurately

        # Storing ages by method age group
        age_bins = [0] + [max(fpd.age_specific_channel_bins[key]) for key in
                          fpd.age_specific_channel_bins]
        self.age_by_group = np.digitize(self.age, age_bins) - 1
        return

    def get_step_results(self):
        """Calculate and return the results for this specific time step"""
        self.track_mcpr()
        self.track_cpr()
        self.track_acpr()
        age_min = self.age >= fpd.min_age
        age_max = self.age < self.pars['age_limit_fecundity']

        self.step_results['total_women_fecund'] = np.sum(self.is_female * age_min * age_max)
        self.step_results['urban_women'] = np.sum(self.urban * self.is_female) / np.sum(self.is_female) * 100
        self.step_results['ever_used_contra'] = np.sum(self.ever_used_contra * self.is_female) / np.sum(self.is_female) * 100
        self.step_results['parity0to1'] = np.sum((self.parity <= 1) & self.is_female) / np.sum(self.is_female) * 100
        self.step_results['parity2to3'] = np.sum((self.parity >= 2) & (self.parity <= 3) & self.is_female) / np.sum(self.is_female) * 100
        self.step_results['parity4to5'] = np.sum((self.parity >= 4) & (self.parity <= 5) & self.is_female) / np.sum(self.is_female) * 100
        self.step_results['parity6plus'] = np.sum((self.parity >= 6) & self.is_female) / np.sum(self.is_female) * 100

        self._step_results_wq()
        self._step_results_intent()
        self._step_results_empower()

        return self.step_results

    def _step_results_wq(self):
        """" Calculate step results on wealthquintile """
        for i in range(1, 6):
            self.step_results[f'wq{i}'] = (np.sum((self.wealthquintile == i) & self.is_female) / np.sum(self.is_female) * 100)
        return

    def _step_results_empower(self):
        self.step_results['paid_employment'] = (np.sum(self.paid_employment & self.is_female & self.alive) / (self.n_female))*100
        self.step_results['decision_wages'] = (np.sum(self.decision_wages & self.is_female & self.alive) / (self.n_female))*100
        self.step_results['decide_spending_partner'] = (np.sum(self.decide_spending_partner & self.is_female & self.alive) / (self.n_female))*100
        self.step_results['buy_decision_major'] = (np.sum(self.buy_decision_major & self.is_female & self.alive) / (self.n_female))*100
        self.step_results['buy_decision_daily'] = (np.sum(self.buy_decision_daily & self.is_female & self.alive) / (self.n_female))*100
        self.step_results['buy_decision_clothes'] = (np.sum(self.buy_decision_clothes & self.is_female & self.alive) / (self.n_female))*100
        self.step_results['decision_health'] = (np.sum(self.decision_health & self.is_female & self.alive) / (self.n_female))*100
        self.step_results['has_savings'] = (np.sum(self.has_savings & self.is_female & self.alive) / (self.n_female))*100
        self.step_results['has_fin_knowl'] = (np.sum(self.has_fin_knowl & self.is_female & self.alive) / (self.n_female))*100
        self.step_results['has_fin_goals'] = (np.sum(self.has_fin_goals & self.is_female & self.alive) / (self.n_female))*100
        #self.step_results['financial_autonomy'] = (np.sum(self.financial_autonomy & self.is_female & self.alive) / (self.n_female)) # incorrect type, need to fiure it out
        #self.step_results['decision_making'] = (np.sum(self.decision_making & self.is_female & self.alive) / (self.n_female))
        return

    def _step_results_intent(self):
        """ Calculate percentage of living women who have intent to use contraception and intent to become pregnant in the next 12 months"""
        self.step_results['perc_contra_intent'] = (np.sum(self.alive & self.is_female & self.intent_to_use) / self.n_female) * 100
        self.step_results['perc_fertil_intent'] = (np.sum(self.alive & self.is_female & self.fertility_intent) / self.n_female) * 100
        return
