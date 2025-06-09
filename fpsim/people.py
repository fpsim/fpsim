"""
Defines the People class
"""

# %% Imports
import numpy as np  # Needed for a few things not provided by pl
import pylab as pl
import sciris as sc
from . import utils as fpu
from . import defaults as fpd
from . import demographics as fpdmg
from . import education as fped
from . import methods as fpm
import starsim as ss

# Specify all externally visible things this file defines
__all__ = ['People']


# %% Define classes

class People(ss.People):
    """
    Class for all the people in the simulation.

    Age pyramid is a 2d array with columns: age, male count, female count
    """

    def __init__(self, n_agents=None, age_pyramid=None, contraception_module=None, empowerment_module=None, education_module=None, **kwargs):

        # Allow defaults to be dynamically set
        self.person_defaults = sc.dcp(fpd.person_defaults)

        ages = age_pyramid[:, 0]
        age_counts = age_pyramid[:, 1] + age_pyramid[:, 2]
        age_data = np.array([ages, age_counts]).T

        f_frac = age_pyramid[:, 2].sum() / age_pyramid[:, 1:3].sum()
        # Initialization
        super().__init__(n_agents, age_data, extra_states=self.person_defaults, **kwargs)
        self.female.default.set(p=f_frac)
        # Empowerment and education
        self.empowerment_module = empowerment_module
        self.education_module = education_module
        self.contraception_module = contraception_module

        self.binom = ss.bernoulli(p=0.5)

        return

    def init_vals(self, uids=None):
        super().init_vals()

        sim = self.sim
        pars = sim.pars
        fp_pars = sim.fp_pars

        if uids is None:
            uids = self.alive.uids

        _urban = self.get_urban(len(uids))

        # init the various modules
        self.contraception_module = self.contraception_module or sc.dcp(fpm.StandardChoice(location=fp_pars['location']))
        self.education_module = self.education_module or sc.dcp(fped.Education(location=fp_pars['location']))

        self.urban[uids] = _urban  # Urban (1) or rural (0)

        # Parameters on sexual and reproductive history
        self.fertile[uids] = fpu.n_binomial(1 - fp_pars['primary_infertility'], len(uids)) # todo replace with ss dist

        # Fertility intent
        # Update distribution of fertility intent with location-specific values if it is present in self.pars
        self.update_fertility_intent(uids)

        # Intent to use contraception
        # Update distribution of fertility intent if it is present in self.pars
        self.update_intent_to_use(uids)

        self.update_wealthquintile(uids)

        # Default initialization for fated_debut; subnational debut initialized in subnational.py otherwise
        self.fated_debut[uids] = fp_pars['debut_age']['ages'][fpu.n_multinomial(fp_pars['debut_age']['probs'], len(uids))]

        # Fecundity variation
        fv = [fp_pars['fecundity_var_low'], fp_pars['fecundity_var_high']]
        fac = (fv[1] - fv[0]) + fv[0]  # Stretch fecundity by a factor bounded by [f_var[0], f_var[1]]
        self.personal_fecundity[uids] = np.random.random(len(uids)) * fac # todo replace

        # Initialise ti_contra based on age and fated debut
        self.update_time_to_choose(uids)

        female_uids = uids[self.female[uids]]

        # Initialize empowerment and education mods.
        if self.empowerment_module is not None:
            self.empowerment_module.initialize(female_uids)

        if self.education_module is not None:
            self.education_module.initialize(self, uids)

        # Partnership
        if fp_pars['use_partnership']:
            fpdmg.init_partnership_states(uids)

        # Handle circular buffer to keep track of historical data
        self.longitude = sc.objdict()
        # self.initialize_circular_buffer() # todo move to analyzer

        # Once all the other metric are initialized, determine initial contraceptive use
        self.barrier[uids] = fpu.n_multinomial(fp_pars['barriers'][:], len(uids))

        # Store keys
        self._keys = [s.name for s in self.states.values()]

        self.init_contraception(uids)  # Initialize contraceptive methods. v3 will refactor this to other modules
        return

    # todo move to analyzer?
    # def initialize_circular_buffer(self):
    #     # Initialize circular buffers to track longitudinal data
    #     longitude_keys = fpd.longitude_keys
    #     # NOTE: by default the history array/circular buffer is initialised with constant
    #     # values. We could potentially initialise the buffer
    #     # with the data from a previous simulation.
    #
    #     for key in longitude_keys:
    #         current = getattr(self, key)  # Current value of this attribute
    #         self.longitude[key] = np.full((self.sim.pars.n_agents, int(self.sim.fp_pars['tiperyear'])), current[0])
    #     return

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
        n_agents = n
        urban_prop = self.sim.fp_pars['urban_prop']
        urban = fpu.n_binomial(urban_prop, n_agents) # todo replace with ss dist
        return urban

    def get_age_sex(self, n):
        """
        For a sample of n ex nihilo people, return arrays of n ages and sexes
            :param n: number of people
            :return: arrays of length n containing their ages and sexes
        """
        pyramid = self.sim.fp_pars['age_pyramid']
        m_frac = pyramid[:, 1].sum() / pyramid[:, 1:3].sum()

        ages = np.zeros(n)
        sexes = np.random.random(n) < m_frac  # Pick the sex based on the fraction of men vs. women
        f_inds = sc.findinds(sexes == 0)
        m_inds = sc.findinds(sexes == 1)

        age_data_min = pyramid[:, 0]
        age_data_max = np.append(pyramid[1:, 0], self.sim.fp_pars['max_age'])
        age_data_range = age_data_max - age_data_min
        for i, inds in enumerate([m_inds, f_inds]):
            if len(inds):
                age_data_prob = pyramid[:, i + 1]
                age_data_prob = age_data_prob / age_data_prob.sum()  # Ensure it sums to 1
                age_bins = fpu.n_multinomial(age_data_prob, len(inds))  # Choose age bins
                ages[inds] = age_data_min[age_bins] + age_data_range[age_bins] * np.random.random(
                    len(inds))  # Uniformly distribute within this age bin

        return ages, sexes

    def update_fertility_intent(self, uids):
        if self.sim.fp_pars['fertility_intent'] is None:
            return
        self.update_fertility_intent_by_age(uids)
        return

    def update_intent_to_use(self, uids):
        if self.sim.fp_pars['intent_to_use'] is None:
            return
        self.update_intent_to_use_by_age(uids)
        return

    def update_wealthquintile(self, uids):
        if self.sim.fp_pars['wealth_quintile'] is None:
            return
        wq_probs = self.sim.fp_pars['wealth_quintile']['percent']
        vals = np.random.choice(len(wq_probs), size=len(uids), p=wq_probs)+1 # todo replace with ss dist
        self.wealthquintile[uids] = vals
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

    def init_contraception(self, uids):
        if self.contraception_module is not None:
            self.decide_contraception(uids=uids)
        return

    def decide_contraception(self, uids=None, ti=None, year=None, contraception_module=None):
        """
        Decide who will start using contraception, when, which contraception method and the
        duration on that method. This method is called by the simulation to initialise the
        people object at the beginning of the simulation and new people born during the simulation.

        #TODO: rename to something that indicates this method is used for initialisation
        #Todo remove excess input params
        """
        if uids is None:
            uids = self.alive.uids

        if ti is None:
            ti = self.sim.ti

        if year is None:
            year = self.sim.y

        fecund = (self.female[uids]==True) & (self.age[uids] < self.sim.fp_pars['age_limit_fecundity'])
        fecund_uids = uids[fecund]
        # NOTE: PSL: This line effectively "initialises" whether a woman is sexually active or not.
        # Because of the current initialisation flow, it's not possible to initialise the
        # sexually_active state in the init constructor.
        self.check_sexually_active(fecund_uids)
        self.update_time_to_choose(fecund_uids)

        # Check whether have reached the time to choose
        time_to_set_contra_uids = fecund_uids[(self.ti_contra[fecund_uids] == 0)]

        if contraception_module is not None:
            self.contraception_module = contraception_module

        if self.contraception_module is not None:
            self.on_contra[time_to_set_contra_uids] = self.contraception_module.get_contra_users(self, time_to_set_contra_uids, year=year, ti=ti, tiperyear=self.sim.fp_pars['tiperyear'])
            oc_uids = time_to_set_contra_uids[(self.on_contra[time_to_set_contra_uids] == True)]
            self.method[oc_uids] = self.contraception_module.init_method_dist(self, oc_uids)
            self.ever_used_contra[oc_uids] = 1
            method_dur = self.contraception_module.set_dur_method(self, time_to_set_contra_uids)
            self.ti_contra[time_to_set_contra_uids] = ti + method_dur

        # Change the intent of women who have started to use a contraception method
        self.intent_to_use[self.on_contra] = False
        return

    def update_fertility_intent_by_age(self, uids=None):
        """
        In the absence of other factors, fertilty intent changes as a function of age
        each year on a woman’s birthday
        """
        intent_pars = self.sim.fp_pars['fertility_intent']
        if uids is None:
            uids = self.alive.uids

        f_uids = uids[(self.female[uids])]
        f_ages = self.age[f_uids]
        age_inds = fpu.digitize_ages_1yr(f_ages)
        for age in intent_pars.keys():
            aged_x_uids = f_uids[age_inds == age]
            fi_cats = list(intent_pars[age].keys())  # all ages have the same intent categories
            probs = np.array(list(intent_pars[age].values()))
            ci = np.random.choice(fi_cats, aged_x_uids.size, p=probs)
            self.categorical_intent[aged_x_uids] = ci

        self.fertility_intent[(self.categorical_intent == "yes")] = True
        self.fertility_intent[((self.categorical_intent == "no") |
                                          (self.categorical_intent == "cannot"))] = False
        return

    def update_intent_to_use_by_age(self, uids=None):
        """
        In the absence of other factors, intent to use contraception
        can change as a function of age each year on a woman’s birthday.

        This function is also used to initialise the State intent_to_use
        """
        intent_pars = self.sim.fp_pars['intent_to_use']
        if uids is None:
            uids = self.alive.uids

        f_uids = uids[(self.female)]
        f_ages = self.age[f_uids]
        age_inds = fpu.digitize_ages_1yr(f_ages)

        for age in intent_pars.keys():
            f_aged_x_uids = f_uids[age_inds == age]  # indices of women of a given age
            prob = intent_pars[age][1]  # Get the probability of having intent
            self.intent_to_use[f_aged_x_uids] = fpu.n_binomial(prob, len(f_aged_x_uids))
        return

    def update_method(self, uids, year=None, ti=None):
        """ Inputs: filtered people, only includes those for whom it's time to update """
        sim = self.sim
        cm = self.contraception_module
        if year is None: year = sim.y
        if ti is None: ti = sim.ti
        if cm is not None:

            # If people are 1 or 6m postpartum, we use different parameters for updating their contraceptive decisions
            is_pp1 = (self.postpartum_dur[uids] == 1)
            is_pp6 = (self.postpartum_dur[uids] == 6) & ~self.on_contra[uids]  # They may have decided to use contraception after 1m
            pp0 = uids[~(is_pp1 | is_pp6)]
            pp1 = uids[is_pp1]
            pp6 = uids[is_pp6]

            # Update choices for people who aren't postpartum
            if len(pp0):

                # If force_choose is True, then all non-users will be made to pick a method
                if cm.pars['force_choose']:
                    must_use = pp0[~self.on_contra[pp0]]
                    choosers = pp0[self.oncontra[pp0]]

                    if len(must_use):
                        self.on_contra[must_use] = True
                        sim.results['contra_access'][sim.ti] += len(must_use)
                        self.method[must_use] = cm.choose_method(must_use)
                        self.ever_used_contra[must_use] = 1
                        sim.results['new_users'][sim.ti] += np.count_nonzero(self.method[must_use])

                else:
                    choosers = pp0

                # Get previous users and see whether they will switch methods or stop using
                if len(choosers):

                    self.on_contra[choosers] = cm.get_contra_users(self, choosers, year=year, ti=ti, tiperyear=sim.fp_pars['tiperyear'])
                    self.ever_used_contra[choosers] = self.ever_used_contra[choosers] | self.on_contra[choosers]

                    # Divide people into those that keep using contraception vs those that stop
                    continuing_contra = choosers[self.on_contra[choosers]]
                    stopping_contra = choosers[~self.on_contra[choosers]]
                    sim.results['contra_access'][sim.ti] += len(continuing_contra)

                    # For those who keep using, choose their next method
                    if len(continuing_contra):
                        self.method[continuing_contra] = cm.choose_method(self, continuing_contra)
                        sim.results['new_users'][sim.ti] += np.count_nonzero(self.method[continuing_contra])

                    # For those who stop using, set method to zero
                    if len(stopping_contra):
                        self.method[stopping_contra] = 0

                # Validate
                n_methods = len(self.contraception_module.methods)
                invalid_vals = (self.method[pp0] >= n_methods) * (self.method[pp0] < 0)
                if invalid_vals.any():
                    errormsg = f'Invalid method set: ti={pp0.ti}, inds={invalid_vals.nonzero()[-1]}'
                    raise ValueError(errormsg)

            # Now update choices for postpartum people. Logic here is simpler because none of these
            # people should be using contraception currently. We first check that's the case, then
            # have them choose their contraception options.
            ppdict = {'pp1': pp1, 'pp6': pp6}
            for event, pp in ppdict.items():
                if len(pp):
                    if self.on_contra[pp].any():
                        errormsg = 'Postpartum women should not currently be using contraception.'
                        raise ValueError(errormsg)
                    self.on_contra[pp] = cm.get_contra_users(self, pp, year=year, event=event, ti=ti, tiperyear=sim.fp_pars['tiperyear'])
                    on_contra = pp[self.on_contra[pp]]
                    off_contra = pp[~self.on_contra[pp]]
                    sim.results['contra_access'][sim.ti] += len(on_contra)

                    # Set method for those who use contraception
                    if len(on_contra):
                        self.method[on_contra] = cm.choose_method(self, on_contra, event=event)
                        self.ever_used_contra[on_contra] = 1

                    if len(off_contra):
                        self.method[off_contra] = 0
                        if event == 'pp1':  # For women 1m postpartum, choose again when they are 6 months pp
                            self.ti_contra[off_contra] = ti + 5

            # Set duration of use for everyone, and reset the time they'll next update
            durs_fixed = (self.postpartum_dur[uids] == 1) & (self.method[uids] == 0)
            update_durs = uids[~durs_fixed]
            dur_methods = cm.set_dur_method(self, update_durs)

            # Check validity
            if (dur_methods < 0).any():
                raise ValueError('Negative duration of method use')

            self.ti_contra[update_durs] = ti + dur_methods

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

        # TODO does setting the probability twice affect cRNG safety?
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

    def start_partnership(self, uids):
        """
        Decide if an agent has reached their age at first partnership. Age-based data from DHS.
        """

        is_not_partnered = self.partnered[uids] == 0
        reached_partnership_age = self.age[uids] >= self.partnership_age[uids]
        first_timers = uids[is_not_partnered & reached_partnership_age]
        self.partnered[first_timers] = True
        return

    def check_sexually_active(self, uids=None):
        """
        Decide if agent is sexually active based either time-on-postpartum month
        or their age if not postpartum.

        Agents can revert to active or not active each timestep. Postpartum and
        general age-based data from DHS.
        """

        if uids is None:
            uids = self.alive.uids

        # Set postpartum probabilities
        match_low = (self.postpartum_dur[uids] >= 0)
        match_high = (self.postpartum_dur[uids] <= self.sim.fp_pars['postpartum_dur'])
        match = (self.postpartum[uids]) & match_low & match_high
        pp = uids[match]
        non_pp = uids[(self.age[uids] >= self.fated_debut[uids]) & ~match]

        # Adjust for postpartum women's birth spacing preferences
        if len(pp):
            pref = self.sim.fp_pars['spacing_pref']  # Shorten since used a lot
            spacing_bins = self.postpartum_dur[pp] / pref['interval']  # Main calculation: divide the duration by the interval
            spacing_bins = np.array(np.minimum(spacing_bins, pref['n_bins']), dtype=int)  # Bound by longest bin
            probs_pp = self.sim.fp_pars['sexual_activity_pp']['percent_active'][(self.postpartum_dur[pp]).astype(int)]
            # Adjust the probability: check the overall probability with print(pref['preference'][spacing_bins].mean())
            probs_pp *= pref['preference'][spacing_bins]
            self.binom.set(p=probs_pp)
            sexually_active, sexually_inactive = self.binom.filter(pp, both=True)
            self.sexually_active[sexually_inactive] = False
            self.sexually_active[sexually_active] = True


        # Set non-postpartum probabilities
        if len(non_pp):
            probs_non_pp = self.sim.fp_pars['sexual_activity'][self.int_age(non_pp)]
            self.sexually_active[non_pp] = fpu.binomial_arr(probs_non_pp)

            # Set debut to True if sexually active for the first time
            # Record agent age at sexual debut in their memory
            never_sex = self.sexual_debut[non_pp] == 0
            now_active = self.sexually_active[non_pp] == 1
            first_debut = non_pp[now_active & never_sex]
            self.sexual_debut[first_debut] = True
            self.sexual_debut_age[first_debut] = self.age[first_debut]

        active_sex = (self.sexually_active[uids] == 1)
        debuted = (self.sexual_debut[uids] == 1)
        active = uids[(active_sex & debuted)]
        inactive = uids[(~active_sex & debuted)]
        self.months_inactive[active] = 0
        self.months_inactive[inactive] += 1

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
        eff_array = np.array([m.efficacy for m in pars['contraception_module'].methods.values()])
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
                self.abortion_ages[abort_uid][abortion_age_index] = self.age[abort_uid]
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
        Agents are randomly assigned a duration value based on a gumbel distribution drawn
        from the 2018 DHS variable for breastfeeding months.
        The mean (mu) and the std dev (beta) are both drawn from that distribution in the DHS data.
        """
        mu, beta = self.sim.fp_pars['breastfeeding_dur_mu'], self.sim.fp_pars['breastfeeding_dur_beta']
        breastfeed_durs = abs(np.random.gumbel(mu, beta, size=len(uids))) # todo replace with ss dist
        breastfeed_durs = np.ceil(breastfeed_durs)
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
                self.miscarriage_ages[miscarriage_uid][miscarriage_age_index] = self.age[miscarriage_uid]
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
                for uid in single_uids:
                    self.birth_ages[ss.uids(uid)][int(parity)] = self.age[ss.uids(uid)]
                if parity == 0: self.first_birth_age[single_uids] = self.age[single_uids]
            for parity in np.unique(self.parity[twin]):
                twin_uids = twin[self.parity[twin] == parity]
                for uid in twin_uids:
                    self.birth_ages[uid][int(parity)] = self.age[uid]
                    self.birth_ages[uid][int(parity) + 1] = self.age[uid]
                if parity == 0: self.first_birth_age[twin_uids] = self.age[twin_uids]
            for parity in np.unique(self.parity[stillborn]):
                uids = stillborn[self.parity[stillborn] == parity]
                for uid in uids:
                    self.stillborn_ages[uid][int(parity)] = self.age[uid]

            self.parity[single] += 1
            self.parity[twin] += 2  # Add 2 because matching DHS "total children ever born (alive) v201"

            # Calculate short intervals
            prev_birth_single = single[self.parity[single] > 1]
            prev_birth_twins = twin[self.parity[twin] > 2]
            if len(prev_birth_single):
                pidx = (self.parity[prev_birth_single] - 1).astype(int)
                all_ints = [self.birth_ages[r][pidx] - self.birth_ages[r][pidx-1] for r in prev_birth_single]
                latest_ints = np.array([r[~np.isnan(r)][-1] for r in all_ints])
                short_ints = np.count_nonzero(latest_ints < (fp_pars['short_int']/fpd.mpy))
                sim.results['short_intervals'][ti] += short_ints
            if len(prev_birth_twins):
                pidx = (self.parity[prev_birth_twins] - 2).astype(int)
                all_ints = [self.birth_ages[r][pidx] - self.birth_ages[r][pidx-1] for r in prev_birth_twins]
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

    def track_mcpr(self):
        """
        Track for purposes of calculating mCPR at the end of the timestep after all people are updated
        Not including LAM users in mCPR as this model counts all women passively using LAM but
        DHS data records only women who self-report LAM which is much lower.
        Follows the DHS definition of mCPR
        """
        modern_methods_num = [idx for idx, m in enumerate(self.contraception_module.methods.values()) if m.modern]
        method_age = (self.sim.fp_pars['method_age'] <= self.age)
        fecund_age = self.age < self.sim.fp_pars['age_limit_fecundity']
        denominator = method_age * fecund_age * self.female * (self.alive)
        numerator = np.isin(self.method, modern_methods_num)
        no_method_mcpr = np.sum((self.method == 0) * denominator)
        on_method_mcpr = np.sum(numerator * denominator)
        self.sim.results['no_methods_mcpr'][self.sim.ti] += no_method_mcpr
        self.sim.results['on_methods_mcpr'][self.sim.ti] += on_method_mcpr

        return

    def track_cpr(self):
        """
        Track for purposes of calculating newer ways to conceptualize contraceptive prevalence
        at the end of the timestep after all people are updated
        Includes women using any method of contraception, including LAM
        Denominator of possible users includes all women aged 15-49
        """
        denominator = ((self.sim.fp_pars['method_age'] <= self.age) * (self.age < self.sim.fp_pars['age_limit_fecundity']) * (
                self.female == True) * (self.alive))
        numerator = self.method != 0
        no_method_cpr = np.sum((self.method == 0) * denominator)
        on_method_cpr = np.sum(numerator * denominator)
        self.sim.results['no_methods_cpr'][self.sim.ti] += no_method_cpr
        self.sim.results['on_methods_cpr'][self.sim.ti] += on_method_cpr

        return

    def track_acpr(self):
        """
        Track for purposes of calculating newer ways to conceptualize contraceptive prevalence
        at the end of the timestep after all people are updated
        Denominator of possible users excludes pregnant women and those not sexually active in the last 4 weeks
        Used to compare new metrics of contraceptive prevalence and eventually unmet need to traditional mCPR definitions
        """
        denominator = ((self.sim.fp_pars['method_age'] <= self.age) * (self.age < self.sim.fp_pars['age_limit_fecundity']) * (
                self.female == True) * (self.pregnant == 0) * (self.sexually_active == 1) * (self.alive))
        numerator = self.method != 0
        no_method_cpr = np.sum((self.method == 0) * denominator)
        on_method_cpr = np.sum(numerator * denominator)
        self.sim.results['no_methods_acpr'][self.sim.ti] += no_method_cpr
        self.sim.results['on_methods_acpr'][self.sim.ti] += on_method_cpr

        return

    def birthday_filter(self, uids=None):
        """
        Returns a filtered ppl object of people who celebrated their bdays, useful for methods that update
        annualy, but not based on a calendar year, rather every year on an agent's bday."""
        if uids is None:
            uids = self.alive.uids
        age_diff = self.age[uids] - self.int_age(uids)
        had_bday = uids[(age_diff <= self.sim.t.dt_year)]
        return had_bday


    # todo move to longitudinal analyzer
    # def update_history_buffer(self):
    #     """
    #     Updates longitudinal params in people object
    #     """
    #
    #     # Calculate column index in which to store current vals
    #     index = int(self.sim.ti % self.sim.fp_pars['tiperyear'])
    #
    #     # Store the current params in people.longitude object
    #     for key in self.longitude.keys():
    #         self.longitude[key][:, index] = getattr(self, key)
    #
    #     return

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

        # Update empowerment states, and empowerment-related states
        if self.empowerment_module is not None: self.step_empowerment(self.female.uids)
        if self.education_module is not None: self.step_education(self.female.uids)

        # Figure out who to update methods for
        ready = nonpreg[self.ti_contra[nonpreg] <= self.sim.ti]

        # Check who has reached their age at first partnership and set partnered attribute to True.
        self.start_partnership(self.female.uids)

        # Complete all updates. Note that these happen in a particular order!
        self.progress_pregnancy(self.pregnant.uids)  # Advance gestation in timestep, handle miscarriage

        # Update mothers
        if self.sim.fp_pars['track_children']:
            self.update_mothers()

        self.check_sexually_active(nonpreg)

        # Update methods for those who are eligible
        if len(ready):
            self.update_method(ready)
            self.sim.results['switchers'][self.sim.ti] = len(ready)  # Track how many people switch methods (incl on/off)

        # Make sure that women who are on contraception do not have intent to use contraception
        self.intent_to_use[self.on_contra==True] = False

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

    def step_empowerment(self, uids):
        """
        NOTE: by default this will not be used, but it will be used for analyses run from the kenya_empowerment repo
        """
        eligible_uids = (self.is_dhs_age[uids]).uids
        # Women who just turned 15 get assigned a value based on empowerment probs
        bday_15_uids = eligible_uids[
            ((self.age[eligible_uids] > int(fpd.min_age)) &
             (self.age[eligible_uids] <= int(fpd.min_age) +
              (self.sim.fp_pars['timestep'] / fpd.mpy)))]
        if len(bday_15_uids):
            self.empowerment_module.update_empwr_states(bday_15_uids)
        # Update states on her bday, based on coefficients
        bday = self.birthday_filter(eligible_uids)
        if len(bday):
            # The empowerment module will update the empowerment states and intent to use
            self.empowerment_module.update(bday)
            # Update fertility intent on her bday, together with empowerment updates
            self.update_fertility_intent_by_age(bday)
        return

    def step_education(self, uids):
        self.education_module.update(self, uids)
        return


    def update_results(self):
        """Calculate and return the results for this specific time step"""
        super().update_results()
        sim = self.sim
        res = sim.results
        ti = sim.ti
        self.track_mcpr()
        self.track_cpr()
        self.track_acpr()
        age_min = self.age >= fpd.min_age
        age_max = self.age < sim.fp_pars['age_limit_fecundity']

        res['total_women_fecund'][ti] = np.sum(self.female * age_min * age_max)
        res['urban_women'][ti] = np.sum(self.urban * self.female) / np.sum(self.female) * 100
        res['ever_used_contra'][ti] = np.sum(self.ever_used_contra * self.female) / np.sum(self.female) * 100
        res['parity0to1'][ti] = np.sum((self.parity <= 1) & self.female) / np.sum(self.female) * 100
        res['parity2to3'][ti] = np.sum((self.parity >= 2) & (self.parity <= 3) & self.female) / np.sum(self.female) * 100
        res['parity4to5'][ti] = np.sum((self.parity >= 4) & (self.parity <= 5) & self.female) / np.sum(self.female) * 100
        res['parity6plus'][ti] = np.sum((self.parity >= 6) & self.female) / np.sum(self.female) * 100

        # Update wealth and education
        self._step_results_wq()
        self._step_results_edu()

        # Update intent and empowerment if empowerment module is present
        if self.empowerment_module is not None:
            self._step_results_intent()
            self._step_results_empower()

        percent0to5 = (res.pp0to5[ti] / res.total_women_fecund[ti]) * 100
        percent6to11 = (res.pp6to11[ti] / res.total_women_fecund[ti]) * 100
        percent12to23 = (res.pp12to23[ti] / res.total_women_fecund[ti]) * 100
        nonpostpartum = ((res.total_women_fecund[ti] - res.pp0to5[ti] - res.pp6to11[ti] - res.pp12to23[ti]) / res.total_women_fecund[ti]) * 100

        # Store results
        res['mcpr'][ti] = sc.safedivide(res.on_methods_mcpr[ti], (res.no_methods_mcpr[ti] + res.on_methods_mcpr[ti]))
        res['cpr'][ti] = sc.safedivide(res.on_methods_cpr[ti], (res.no_methods_cpr[ti] + res.on_methods_cpr[ti]))
        res['acpr'][ti] = sc.safedivide(res.on_methods_acpr[ti], (res.no_methods_acpr[ti] + res.on_methods_acpr[ti]))
        res['pp0to5'][ti] = percent0to5
        res['pp6to11'][ti] = percent6to11
        res['pp12to23'][ti] = percent12to23
        res['nonpostpartum'][ti] = nonpostpartum



        time_months = int(np.round(ti * sim.t.dt_year * fpd.mpy))  # time since the beginning of the sim, expresse in months

        # Start calculating annual metrics after we have at least 1 year of data. These are annual metrics, and so are not the same dimensions as a standard results object.
        if (time_months >= fpd.mpy) and (time_months % fpd.mpy) == 0:
                index = int(time_months / fpd.mpy) - 1
                res['tfr_years'][index] = (sim.t.yearvec[ti]-1.0)  # Subtract one year as we're calculating the statistics between 1st jan 'year-1' and 1st jan 'year'. Results correspond to 'year-1'.
                stop_index = ti
                start_index = int(stop_index - sim.fp_pars['tiperyear'])
                for res_name, new_res_name in fpd.to_annualize.items():
                    res_over_year = self.annualize_results(res_name, start_index, stop_index)
                    annual_res_name = f'{new_res_name}_over_year'
                    res[annual_res_name][index] = (res_over_year)

                computed_method_usage = self.compute_method_usage()
                for i in range(len(computed_method_usage)):
                    res['method_usage'][i][index] = (computed_method_usage[i])
                res['pop_size'][index] = (res['n_alive'][ti])
                res['mcpr_by_year'][index] = (res['mcpr'][ti])
                res['cpr_by_year'][index] = (res['cpr'][ti])

                # Calculate annual ratios
                self.calculate_annual_ratios()

                tfr = 0
                for key in fpd.age_bin_map.keys():
                    age_bin_births_year = np.sum(res['total_births_' + key][start_index:stop_index])
                    age_bin_total_women_year = res['total_women_' + key][stop_index]
                    age_bin_births_per_woman = sc.safedivide(age_bin_births_year, age_bin_total_women_year)
                    res[f'asfr'][key][index] = (age_bin_births_per_woman * 1000)
                    res[f'tfr_{key}'][index] = (age_bin_births_per_woman * 1000)
                    tfr += age_bin_births_per_woman
                res['tfr_rates'][index] = (tfr * 5)  # CK: TODO: why *5? # SB: I think this corresponds to size of age bins?

        return

    def get_annual_index(self):
        time_months = int(np.round(
            self.sim.ti * self.sim.t.dt_year * fpd.mpy))  # time since the beginning of the sim, expresse in months

        return int(time_months / fpd.mpy) - 1

    def annualize_results(self, key, start_index, stop_index):
        return np.sum(self.sim.results[key][start_index:stop_index])


    def calculate_annual_ratios(self):
        index = self.get_annual_index()
        live_births_over_year = self.sim.results['live_births_over_year'][index]

        maternal_mortality_ratio = sc.safedivide(self.sim.results['maternal_deaths_over_year'][index], live_births_over_year) * 100000
        self.sim.results['mmr'][index] = (maternal_mortality_ratio)

        infant_mortality_rate = sc.safedivide(self.sim.results['infant_deaths_over_year'][index], live_births_over_year) * 1000
        self.sim.results['imr'][index] = (infant_mortality_rate)

        self.sim.results['proportion_short_interval_by_year'][index] = (sc.safedivide(self.sim.results['short_intervals_over_year'][index], self.sim.results['secondary_births_over_year'][index]))
        return

    def _step_results_wq(self):
        """" Calculate step results on wealthquintile """
        for i in range(1, 6):
            self.sim.results[f'wq{i}'][self.sim.ti] = (np.sum((self.wealthquintile == i) & self.female) / np.sum(self.female) * 100)
        return

    @staticmethod
    def cond_prob(a, b):
        """ Calculate conditional probability. This should be moved somewhere else. """
        return np.sum(a & b) / np.sum(b)

    def _step_results_edu(self):
        denom = self.female & self.alive & (self.age >= fpd.min_age) & (self.age < fpd.max_age)
        self.sim.results['edu_objective'][self.sim.ti] = np.mean(self.edu_objective[denom])
        self.sim.results['edu_attainment'][self.sim.ti] = np.mean(self.edu_attainment[denom])

    def _step_results_empower(self):
        res = self.sim.results
        ti = self.sim.ti
        res['paid_employment'][ti] = (np.sum(self.paid_employment & self.female & self.alive  & (self.age>=fpd.min_age) & (self.age<fpd.max_age))/ np.sum(self.female & self.alive  & (self.age>=fpd.min_age) & (self.age<fpd.max_age)))*100
        res['decision_wages'][ti] = (np.sum(self.decision_wages & self.female & self.alive & (self.age>=fpd.min_age) & (self.age<fpd.max_age)) / np.sum(self.female & self.alive  & (self.age>=fpd.min_age) & (self.age<fpd.max_age)))*100
        res['decide_spending_partner'][ti] = (np.sum(self.decide_spending_partner & self.female & self.alive) / (self.n_female))*100
        res['buy_decision_major'][ti] = (np.sum(self.buy_decision_major & self.female & self.alive) / (self.n_female))*100
        res['buy_decision_daily'][ti] = (np.sum(self.buy_decision_daily & self.female & self.alive) / (self.n_female))*100
        res['buy_decision_clothes'][ti] = (np.sum(self.buy_decision_clothes & self.female & self.alive) / (self.n_female))*100
        res['decision_health'][ti] = (np.sum(self.decision_health & self.female & self.alive) / (self.n_female))*100
        res['has_savings'][ti] = (np.sum(self.has_savings & self.female & self.alive) / (self.n_female))*100
        res['has_fin_knowl'][ti] = (np.sum(self.has_fin_knowl & self.female & self.alive) / (self.n_female))*100
        res['has_fin_goals'][ti] = (np.sum(self.has_fin_goals & self.female & self.alive) / (self.n_female))*100

        return


    def _step_results_intent(self):
        """ Calculate percentage of living women who have intent to use contraception and intent to become pregnant in the next 12 months"""
        self.sim.results['perc_contra_intent'][self.sim.ti] = (np.sum(self.alive & self.female & self.intent_to_use) / self.n_female) * 100
        self.sim.results['perc_fertil_intent'][self.sim.ti] = (np.sum(self.alive & self.female & self.fertility_intent) / self.n_female) * 100
        return


    def int_age(self, uids=None):
        ''' Return ages as an integer '''
        if uids is None:
            return np.array(self.age, dtype=np.int64)
        return np.array(self.age[uids], dtype=np.int64)


    def int_age_clip(self, uids=None):
        ''' Return ages as integers, clipped to maximum allowable age for pregnancy '''
        if uids is None:
            return np.minimum(self.int_age(), fpd.max_age_preg)
        return np.minimum(self.int_age(uids), fpd.max_age_preg)

    def update_post(self):
        """ Final updates at the very end of the timestep """
        sim = self.sim
        if sim.pars.use_aging:
            self.age[self.alive.uids] += sim.t.dt_year
            # there is a max age for some of the stats, so if we exceed that, reset it
            self.age[self.alive.uids] = np.minimum(self.age[self.alive.uids], self.sim.fp_pars['max_age'])
        return


    def compute_method_usage(self):
        """
        Computes method mix proportions from a sim object

        Returns:
            list of lists where list[years_after_start][method_index] == proportion of
            fecundity aged women using that method on that year
        """

        min_age = fpd.min_age
        max_age = self.sim.fp_pars['age_limit_fecundity']

        # filtering for women with appropriate characteristics
        bool_list_uids = (self.alive & (self.female) & (self.age >= min_age) * (self.age <= max_age)).uids
        filtered_methods = self.method[bool_list_uids]

        unique, counts = np.unique(filtered_methods, return_counts=True)
        count_dict = dict(zip(unique, counts))

        result = [0] * (len(self.sim.fp_pars['contraception_module'].methods))
        for method in count_dict:
            result[int(method)] = count_dict[int(method)] / len(filtered_methods)

        return result


    def plot(self, fig_args=None, hist_args=None):
        ''' Plot histograms of each quantity '''

        fig_args  = sc.mergedicts(fig_args)
        hist_args = sc.mergedicts(dict(bins=50), hist_args)
        keys = [key for key in self.__dict__.keys() if isinstance(self.__dict__[key], ss.FloatArr)]
        nkeys = len(keys)
        rows,cols = sc.get_rows_cols(nkeys)

        fig = pl.figure(**fig_args)

        for k,key in enumerate(keys):
            pl.subplot(rows,cols,k+1)
            try:
                data = np.array(self[key], dtype=float)
                mean = data.mean()
                label = f'mean: {mean}'
                pl.hist(data, label=label, **hist_args)
                pl.title(key)
                pl.legend()
            except:
                pl.title(f'Could not plot {key}')

        return fig