'''
Defines the Sim class, the core class of the FP model (FPsim).
'''

#%% Imports
import numpy as np # Needed for a few things not provided by pl
import pylab as pl
import seaborn as sns
import sciris as sc
import pandas as pd
from . import defaults as fpd
from . import utils as fpu
from . import base as fpb


# Specify all externally visible things this file defines
__all__ = ['People', 'Sim', 'MultiSim', 'parallel']


#%% Define classes
def arr(n=None, val=0):
    ''' Shortcut for defining an empty array with the correct value and data type '''
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
    '''
    Class for all the people in the simulation.
    '''
    def __init__(self, pars, n=None, **kwargs):

        # Initialization
        super().__init__()
        self.pars = pars # Set parameters
        d = sc.mergedicts(fpd.person_defaults, kwargs) # d = defaults
        if n is None:
            n = int(self.pars['n_agents'])

        # Basic states
        init_states = dir(self)
        self.uid      = arr(n, np.arange(n))
        self.age      = arr(n, np.float64(d['age'])) # Age of the person (in years)
        self.sex      = arr(n, d['sex']) # Female (0) or male (1)
        self.parity   = arr(n, d['parity']) # Number of children
        self.method   = arr(n, d['method'])  # Contraceptive method 0-9, see pars['methods']['map'], excludes LAM as method
        self.barrier  = arr(n, d['barrier'])  # Reason for non-use
        self.alive    = arr(n, d['alive'])
        self.pregnant = arr(n, d['pregnant'])
        self.fertile  = arr(n, d['fertile'])  # assigned likelihood of remaining childfree throughout reproductive years

        # Sexual and reproductive history
        self.sexually_active  = arr(n, d['sexually_active'])
        self.sexual_debut     = arr(n, d['sexual_debut'])
        self.sexual_debut_age = arr(n, np.float64(d['sexual_debut_age'])) # Age at first sexual debut in years, If not debuted, -1
        self.fated_debut      = arr(n, np.float64(d['debut_age']))
        self.first_birth_age  = arr(n, np.float64(d['first_birth_age'])) # Age at first birth.  If no births, -1
        self.lactating        = arr(n, d['lactating'])
        self.gestation        = arr(n, d['gestation'])
        self.preg_dur         = arr(n, d['preg_dur'])
        self.stillbirth       = arr(n, d['stillbirth']) # Number of stillbirths
        self.miscarriage      = arr(n, d['miscarriage']) # Number of miscarriages
        self.abortion         = arr(n, d['abortion']) # Number of abortions
        self.postpartum       = arr(n, d['postpartum'])
        self.mothers          = arr(n, d['mothers'])

        self.postpartum_dur       = arr(n, d['postpartum_dur']) # Tracks # months postpartum
        self.lam                  = arr(n, d['lam']) # Separately tracks lactational amenorrhea, can be using both LAM and another method
        self.breastfeed_dur       = arr(n, d['breastfeed_dur'])
        self.breastfeed_dur_total = arr(n, d['breastfeed_dur_total'])

        self.children          = arr(n, []) # Indices of children -- list of lists
        self.dobs              = arr(n, []) # Dates of births -- list of lists
        self.still_dates       = arr(n, []) # Dates of stillbirths -- list of lists
        self.miscarriage_dates = arr(n, []) # Dates of miscarriages -- list of lists
        self.abortion_dates    = arr(n, []) # Dates of abortions -- list of lists

        # Fecundity variation
        fv = [self.pars['fecundity_var_low'], self.pars['fecundity_var_high']]
        self.personal_fecundity = arr(n, np.random.random(n)*(fv[1]-fv[0])+fv[0]) # Stretch fecundity by a factor bounded by [f_var[0], f_var[1]]
        self.remainder_months = arr(n, d['remainder_months'])

        # Store keys
        final_states = dir(self)
        self._keys = [s for s in final_states if s not in init_states]

        return


    def update_method(self):
        '''
        Uses a switching matrix from DHS data to decide based on a person's original method their probability of changing to a
        new method and assigns them the new method. Currently allows switching on whole calendar years to enter function.
        Matrix serves as an initiation, discontinuation, continuation, and switching matrix. Transition probabilities are for 1 year and
        only for women who have not given birth within the last 6 months.
        '''
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
        for key,(age_low, age_high) in fpd.method_age_map.items():
            match_low  = (self.age >= age_low) # CK: TODO: refactor into single method
            match_high = (self.age <  age_high)
            match_low_high = match_low * match_high
            for m in method_map.values():
                match_m = (orig_methods == m)
                match = match_m * match_low_high
                this_method = self.filter(match)
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
            self.step_results_switching['annual'] += switching_events # CK: TODO: remove this extra result and combine with step_results
            for key in fpd.method_age_map.keys():
                self.step_results['switching_annual'][key] += switching_events_ages[key]

        return


    def update_method_pp(self):
        '''
        Utilizes data from birth to allow agent to initiate a method postpartum coming from birth by
        3 months postpartum and then initiate, continue, or discontinue a method by 6 months postpartum.
        Next opportunity to switch methods will be on whole calendar years, whenever that falls.
        '''
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
        for key,(age_low, age_high) in fpd.method_age_map.items():
            match_low  = (self.age >= age_low)
            match_high = (self.age <  age_high)
            match_postpartum_age = self.postpartum * postpartum6 * match_low * match_high
            for m in methods_map.values():
                match_m    = (orig_methods == m)
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
        '''If eligible (age 15-49 and not pregnant), choose new method or stay with current one'''

        if not (self.i % self.pars['method_timestep']): # Allow skipping timesteps
            postpartum = (self.postpartum) * (self.postpartum_dur <= 6)
            pp = self.filter(postpartum)
            non_pp = self.filter(~postpartum)

            pp.update_method_pp() # Update method for

            age_diff = non_pp.ceil_age - non_pp.age
            whole_years = ((age_diff < (1/fpd.mpy)) * (age_diff > 0))
            birthdays = non_pp.filter(whole_years)
            birthdays.update_method()

        return


    def check_mortality(self):
        '''Decide if person dies at a timestep'''

        timestep  = self.pars['timestep']
        trend_val = self.pars['mortality_probs']['gen_trend']
        age_mort  = self.pars['age_mortality']
        f_spline = age_mort['f_spline'] * trend_val
        m_spline = age_mort['m_spline'] * trend_val
        over_one = self.filter(self.age >= 1)
        female   = over_one.filter(over_one.is_female)
        male     = over_one.filter(over_one.is_male)
        f_ages = female.int_age
        m_ages = male.int_age

        f_mort_prob = fpu.annprob2ts(f_spline[f_ages], timestep)
        m_mort_prob = fpu.annprob2ts(m_spline[m_ages], timestep)

        f_died = female.binomial(f_mort_prob, as_filter=True)
        m_died = male.binomial(m_mort_prob, as_filter=True)
        for died in [f_died, m_died]:
            died.alive           = False,
            died.pregnant        = False,
            died.gestation       = False,
            died.sexually_active = False,
            died.lactating       = False,
            died.postpartum      = False,
            died.lam             = False,
            died.breastfeed_dur  = 0,
            self.step_results['deaths'] += len(died)

        return


    def check_sexually_active(self):
        '''
        Decide if agent is sexually active based either on month postpartum or age if
        not postpartum.  Postpartum and general age-based data from DHS.
        '''
        # Set postpartum probabilities
        match_low  = self.postpartum_dur >= 0
        match_high = self.postpartum_dur <= self.pars['postpartum_dur']
        pp_match = self.postpartum * match_low * match_high
        non_pp_match = ((self.age >= self.fated_debut) * (~pp_match))
        pp = self.filter(pp_match)
        non_pp = self.filter(non_pp_match)

        # Adjust for postpartum women's birth spacing preferences
        pref = self.pars['spacing_pref'] # Shorten since used a lot
        spacing_bins = pp.postpartum_dur / pref['interval'] # Main calculation -- divide the duration by the interval
        spacing_bins = np.array(np.minimum(spacing_bins, pref['n_bins']), dtype=int) # Convert to an integer and bound by longest bin
        probs_pp = self.pars['sexual_activity_pp']['percent_active'][pp.postpartum_dur]
        probs_pp *= pref['preference'][spacing_bins] # Actually adjust the probability -- check the overall probability with print(pref['preference'][spacing_bins].mean())

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

        return


    def check_conception(self):
        '''
        Decide if person (female) becomes pregnant at a timestep.
        '''
        all_ppl    = self.unfilter() # For complex array operations
        active     = self.filter(self.sexually_active * self.fertile)
        lam        = active.filter(active.lam)
        nonlam     = active.filter(~active.lam)
        preg_probs = np.zeros(len(all_ppl)) # Use full array

        # Find monthly probability of pregnancy based on fecundity and any use of contraception including LAM - from data
        preg_eval_lam    = self.pars['age_fecundity'][lam.int_age_clip] * lam.personal_fecundity
        preg_eval_nonlam = self.pars['age_fecundity'][nonlam.int_age_clip] * nonlam.personal_fecundity
        method_eff       = self.pars['method_efficacy'][nonlam.method]
        lam_eff          = self.pars['LAM_efficacy']

        lam_probs    = fpu.annprob2ts((1-lam_eff)*preg_eval_lam,       self.pars['timestep'])
        nonlam_probs = fpu.annprob2ts((1-method_eff)*preg_eval_nonlam, self.pars['timestep'])
        preg_probs[lam.inds]    = lam_probs
        preg_probs[nonlam.inds] = nonlam_probs

        # Adjust for decreased likelihood of conception if nulliparous vs already gravid - from PRESTO data
        nullip = active.filter(active.parity == 0) # Nulliparous
        preg_probs[nullip.inds] *= self.pars['fecundity_ratio_nullip'][nullip.int_age_clip]

        # Adjust for probability of exposure to pregnancy episode at this timestep based on age and parity - encapsulates background factors - experimental and tunable
        preg_probs *= self.pars['exposure_factor']
        preg_probs *= self.pars['exposure_age'][all_ppl.int_age_clip]
        preg_probs *= self.pars['exposure_parity'][np.minimum(all_ppl.parity, fpd.max_parity)]

        # Use a single binomial trial to check for conception successes this month
        conceived = active.binomial(preg_probs[active.inds], as_filter=True)
        unintended = conceived.filter(conceived.method != 0)
        self.step_results['unintended_pregs'] += len(unintended)

        # Check for abortion
        is_abort = conceived.binomial(self.pars['abortion_prob'])
        abort = conceived.filter(is_abort)
        preg = conceived.filter(~is_abort)

        # Update states
        all_ppl = self.unfilter()
        abort.postpartum = False
        abort.abortion += 1 # Add 1 to number of abortions agent has had
        abort.postpartum_dur = 0
        for i in abort.inds: # Handle adding dates
            all_ppl.abortion_dates[i].append(all_ppl.age[i])

        # Make selected agents pregnant
        preg.make_pregnant()

        return


    def make_pregnant(self):
        '''
        Update the selected agents to be pregnant
        '''
        pregdur = [self.pars['preg_dur_low'], self.pars['preg_dur_high']]
        self.pregnant = True
        self.gestation = 1  # Start the counter at 1
        self.preg_dur = np.random.randint(pregdur[0], pregdur[1]+1, size=len(self))  # Duration of this pregnancy
        self.postpartum = False
        self.postpartum_dur = 0
        self.reset_breastfeeding() # Stop lactating if becoming pregnant
        self.method = 0
        return


    def check_lam(self):
        '''
        Check to see if postpartum agent meets criteria for LAM in this time step
        '''
        max_lam_dur = 5 # TODO: remove hard-coding, make a parameter
        lam_candidates = self.filter((self.postpartum) * (self.postpartum_dur <= max_lam_dur))
        probs = self.pars['lactational_amenorrhea']['rate'][lam_candidates.postpartum_dur]
        lam_candidates.lam = lam_candidates.binomial(probs)

        not_postpartum    = self.postpartum == 0
        over5mo           = self.postpartum_dur > max_lam_dur
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
        for key,(pp_low, pp_high) in fpd.postpartum_map.items():
            this_pp_bin = pp.filter((pp.postpartum_dur >= pp_low) * (pp.postpartum_dur <  pp_high))
            self.step_results[key] += len(this_pp_bin)
        pp.postpartum_dur += self.pars['timestep']

        return


    def update_pregnancy(self):
        '''Advance pregnancy in time and check for miscarriage'''

        preg = self.filter(self.pregnant)
        preg.gestation += self.pars['timestep']

        # Check for miscarriage at the end of the first trimester
        end_first_tri     = preg.filter(preg.gestation == self.pars['end_first_tri'])
        miscarriage_probs = self.pars['miscarriage_rates'][end_first_tri.int_age_clip]
        miscarriage  = end_first_tri.binomial(miscarriage_probs, as_filter=True)

        # Reset states and track miscarriages
        all_ppl = self.unfilter()
        miscarriage.pregnant   = False
        miscarriage.miscarriage += 1 # Add 1 to number of miscarriages agent has had
        miscarriage.postpartum = False
        miscarriage.gestation  = 0  # Reset gestation counter
        for i in miscarriage.inds: # Handle adding dates
            all_ppl.miscarriage_dates[i].append(all_ppl.age[i])
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

        deliv.pregnant = False
        deliv.gestation = 0  # Reset gestation counter
        deliv.lactating = True
        deliv.postpartum = True # Start postpartum state at time of birth
        deliv.breastfeed_dur = 0  # Start at 0, will update before leaving timestep in separate function
        deliv.postpartum_dur = 0

        # Handle stillbirth
        is_stillborn = deliv.binomial(self.pars['mortality_probs']['stillbirth'])
        stillborn = deliv.filter(is_stillborn)
        stillborn.stillbirth += 1  # Track how many stillbirths an agent has had
        stillborn.lactating = False   # Set agents of stillbith to not lactate
        self.step_results['stillbirths'] = len(stillborn)

        # Add dates of live births and stillbirths separately for agent to remember
        all_ppl = self.unfilter()
        live = deliv.filter(~is_stillborn)
        for i in live.inds: # Handle DOBs
            all_ppl.dobs[i].append(all_ppl.age[i])  # Used for birth spacing only, only add one baby to dob -- CK: can't easily turn this into a Numpy operation
            if len(all_ppl.dobs[i]) == 1:
                all_ppl.first_birth_age[i] = all_ppl.age[i]
        for i in stillborn.inds: # Handle adding dates
            all_ppl.still_dates[i].append(all_ppl.age[i])

        # Handle twins
        is_twin = live.binomial(self.pars['twins_prob'])
        twin = live.filter(is_twin)
        self.step_results['births'] += 2*len(twin) # only add births to population if born alive
        twin.parity += 2 # Add 2 because matching DHS "total children ever born (alive) v201"

        # Handle singles
        single = live.filter(~is_twin)
        self.step_results['births'] += len(single)
        single.parity += 1

        #Calculate total births
        self.step_results['total_births'] = len(stillborn) + self.step_results['births']

        live_age = live.age
        for key, (age_low, age_high) in fpd.age_bin_map.items():
            birth_bins = np.sum((live_age >= age_low) * (live_age < age_high))
            self.step_results['birth_bins'][key] += birth_bins

        # Check mortality
        live.check_maternal_mortality() # Mothers of only live babies eligible to match definition of maternal mortality ratio
        i_death = live.check_infant_mortality()

        # TEMP -- update children, need to refactor
        r = fpu.dict2obj(self.step_results)
        new_people = r.births - r.infant_deaths # Do not add agents who died before age 1 to population
        children_map = sc.ddict(int)
        for i in live.inds:
            children_map[i] += 1
        for i in twin.inds:
            children_map[i] += 1
        for i in i_death.inds:
            children_map[i] -= 1

        assert sum(list(children_map.values())) == new_people
        start_ind = len(all_ppl)
        for mother,n_children in children_map.items():
            end_ind = start_ind+n_children
            children = list(range(start_ind, end_ind))
            all_ppl.children[mother] += children
            start_ind = end_ind

        return


    def update_age(self):
        '''Advance age in the simulation'''
        self.age += self.pars['timestep'] / fpd.mpy  # Age the person for the next timestep
        self.age = np.minimum(self.age, self.pars['max_age'])
        return


    def update_age_bin_totals(self):
        '''
        Count how many total live women in each 5-year age bin 10-50, for tabulating ASFR
        '''
        for key, (age_low, age_high) in fpd.age_bin_map.items():
            this_age_bin = self.filter((self.age >= age_low) * (self.age < age_high))
            self.step_results['age_bin_totals'][key] += len(this_age_bin)
        return


    def track_mcpr(self):
        '''
        Track for purposes of calculating mCPR at the end of the timestep after all people are updated
        Not including LAM users in mCPR as this model counts all women passively using LAM but
        DHS data records only women who self-report LAM which is much lower.
        Follows the DHS definition of mCPR
        '''
        modern_methods = [1, 2, 3, 4, 5, 7, 9]
        denominator = (self.pars['method_age'] <= self.age) * (self.age < self.pars['age_limit_fecundity']) * \
                      (self.sex == 0) * (self.alive)
        no_method_mcpr = np.sum((self.method == 0) * denominator)
        on_method_mcpr = np.sum((np.isin(self.method, modern_methods)) * denominator)
        self.step_results['no_methods_mcpr'] += no_method_mcpr
        self.step_results['on_methods_mcpr'] += on_method_mcpr
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
        no_method_cpr = np.sum((self.method == 0) * denominator)
        on_method_cpr = np.sum((self.method != 0) * denominator)
        self.step_results['no_methods_cpr'] += no_method_cpr
        self.step_results['on_methods_cpr'] += on_method_cpr
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
        no_method_cpr = np.sum((self.method == 0) * denominator)
        on_method_cpr = np.sum((self.method != 0) * denominator)
        self.step_results['no_methods_acpr'] += no_method_cpr
        self.step_results['on_methods_acpr'] += on_method_cpr
        return


    def init_step_results(self):
        self.step_results = dict(
            deaths          = 0,
            births          = 0,
            stillbirths     = 0,
            total_births    = 0,
            maternal_deaths = 0,
            infant_deaths   = 0,
            on_methods_mcpr = 0,
            no_methods_mcpr = 0,
            on_methods_cpr  = 0,
            no_methods_cpr  = 0,
            on_methods_acpr = 0,
            no_methods_acpr = 0,
            pp0to5          = 0,
            pp6to11         = 0,
            pp12to23        = 0,
            total_women_fecund = 0,
            unintended_pregs = 0,
            birthday_fraction = None,
            birth_bins        = {},
            age_bin_totals    = {},
            switching_annual     = {},
            switching_postpartum = {}
        )

        for key in fpd.age_bin_map.keys():
            self.step_results['birth_bins'][key] = 0
            self.step_results['age_bin_totals'][key] = 0

        m = len(self.pars['methods']['map'])

        def mm_zeros():
            ''' Return an array of m x m zeros '''
            return np.zeros((m, m), dtype=int)

        if self.pars['track_switching']:
            for key in fpd.method_age_map.keys():
                self.step_results['switching_annual'][key]    = mm_zeros()
                self.step_results['switching_postpartum'][key] = mm_zeros()

            self.step_results['switching'] = dict(
                annual     = mm_zeros(),
                postpartum = mm_zeros(),
            )

        return


    def update(self):
        '''
        Update the person's state for the given timestep.
        t is the time in the simulation in years (ie, 0-60), y is years of simulation (ie, 1960-2010)'''

        self.init_step_results()   # Initialize outputs
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
        lact    = fecund.filter(fecund.lactating)
        methods = nonpreg.filter(nonpreg.age >= self.pars['method_age'])

        # Update everything else
        preg.update_pregnancy()  # Advance gestation in timestep, handle miscarriage
        nonpreg.check_sexually_active()
        methods.update_methods()
        nonpreg.check_lam()
        nonpreg.update_postpartum() # Updates postpartum counter if postpartum
        lact.update_breastfeeding()
        nonpreg.check_conception()  # Decide if conceives and initialize gestation counter at 0

        # Update results
        fecund.update_age_bin_totals()
        #fecund.check_mcpr() TODO - build method to check mcpr at end of step, will be simpler than below
        #fecund.update_total_fecund_women()  TODO- build method to track all live women 15-49 for TFR, below not working

        # Update results
        self.track_mcpr()
        self.track_cpr()
        self.track_acpr()
        self.step_results['total_women_fecund'] = np.sum((self.sex == 0) * (15 <= self.age) * (self.age < self.pars['age_limit_fecundity'])) # CK: TODO: remove hardcoding

        # Age person at end of timestep after tabulating results
        alive_now.update_age()  # Important to keep this here so birth spacing gets recorded accurately

        return self.step_results



class Sim(fpb.BaseSim):
    '''
    The Sim class handles the running of the simulation
    '''

    def __init__(self, pars=None, location=None, label=None, mother_ids=False, **kwargs):
        if pars is None:
            pars = fpd.pars(location)

        # Check parameters
        mismatches = [key for key in kwargs.keys() if key not in fpd.par_keys]
        if len(mismatches):
            errormsg = f'Key(s) {mismatches} not found; available keys are {fpd.par_keys}'
            raise sc.KeyNotFoundError(errormsg)
        super().__init__(pars, location=location, **kwargs) # Initialize and set the parameters as attributes

        self.initialized = False
        self.already_run = False
        self.test_mode   = False
        self.label       = label
        self.mother_ids  = mother_ids
        fpu.set_metadata(self) # Set version, date, and git info
        return


    def initialize(self, force=False):
        if force or not self.initialized:
            fpu.set_seed(self['seed'])
            self.init_results()
            self.init_people()
        return self


    def init_results(self):
        resultscols = ['t', 'pop_size_months', 'births', 'deaths', 'stillbirths', 'total_births', 'maternal_deaths', 'infant_deaths',
                       'cum_maternal_deaths', 'cum_infant_deaths', 'on_methods_mcpr', 'no_methods_mcpr', 'on_methods_cpr', 'no_methods_cpr', 'on_methods_acpr',
                       'no_methods_acpr', 'mcpr', 'cpr', 'acpr', 'pp0to5', 'pp6to11', 'pp12to23', 'nonpostpartum', 'total_women_fecund', 'unintended_pregs', 'birthday_fraction',
                       'total_births_10-14', 'total_births_15-19', 'total_births_20-24', 'total_births_25-29', 'total_births_30-34', 'total_births_35-39', 'total_births_40-44',
                       'total_births_45-49', 'total_women_10-14', 'total_women_15-19', 'total_women_20-24', 'total_women_25-29', 'total_women_30-34', 'total_women_35-39',
                       'total_women_40-44', 'total_women_45-49']
        self.results = {}
        for key in resultscols:
            self.results[key] = np.zeros(int(self.npts))
        self.results['tfr_years'] = [] # CK: TODO: refactor into loop with keys
        self.results['tfr_rates'] = []
        self.results['pop_size'] = []
        self.results['mcpr_by_year'] = []
        self.results['cpr_by_year']  = []
        self.results['method_failures_over_year'] = []
        self.results['infant_deaths_over_year'] = []
        self.results['total_births_over_year'] = []
        self.results['live_births_over_year'] = []
        self.results['maternal_deaths_over_year'] = []
        self.results['mmr'] = []
        self.results['imr'] = []
        self.results['birthday_fraction'] = []
        self.results['asfr'] = {}

        for key in fpd.age_bin_map.keys():
            self.results['asfr'][key] = []

        if self['track_switching']:
            m = len(self['methods']['map'])
            keys = [
                'switching_events_annual',
                'switching_events_postpartum',
                'switching_events_<18',
                'switching_events_18-20',
                'switching_events_21-25',
                'switching_events_>25',
                'switching_events_pp_<18',
                'switching_events_pp_18-20',
                'switching_events_pp_21-25',
                'switching_events_pp_>25',
            ]
            for key in keys:
                self.results[key] = {} # CK: TODO: refactor
                for p in range(self.npts):
                    self.results[key][p] = np.zeros((m, m), dtype=int)

        return


    def get_age_sex(self, n):
        ''' For an ex nihilo person, figure out if they are male and female, and how old '''
        pyramid = self['age_pyramid']
        self.m_frac = pyramid[:,1].sum() / pyramid[:,1:3].sum()

        ages = np.zeros(n)
        sexes = np.random.random(n) < self.m_frac  # Pick the sex based on the fraction of men vs. women
        f_inds = sc.findinds(sexes == 0)
        m_inds = sc.findinds(sexes == 1)

        age_data_min   = pyramid[:,0]
        age_data_max   = np.append(pyramid[1:,0], self['max_age'])
        age_data_range = age_data_max - age_data_min
        for i,inds in enumerate([m_inds, f_inds]):
            if len(inds):
                age_data_prob  = pyramid[:,i+1]
                age_data_prob  = age_data_prob/age_data_prob.sum() # Ensure it sums to 1
                age_bins       = fpu.n_multinomial(age_data_prob, len(inds)) # Choose age bins
                ages[inds]     = age_data_min[age_bins] + age_data_range[age_bins]*np.random.random(len(inds)) # Uniformly distribute within this age bin


        return ages, sexes


    def make_people(self, n=1, age=None, sex=None, method=None, debut_age=None):
        ''' Set up each person '''
        _age, _sex = self.get_age_sex(n)
        if age    is None: age    = _age
        if sex    is None: sex    = _sex
        if method is None: method = np.zeros(n, dtype=np.int64)
        barrier = fpu.n_multinomial(self['barriers'][:], n)
        debut_age = self['debut_age']['ages'][fpu.n_multinomial(self['debut_age']['probs'], n)]
        fertile = fpu.n_binomial(1 - self['primary_infertility'], n)
        data = dict(age=age, sex=sex, method=method, barrier=barrier, debut_age=debut_age, fertile=fertile)
        return data


    def init_people(self, output=False, **kwargs):
        ''' Create the people '''
        p = sc.objdict(self.make_people(n=int(self['n_agents'])))
        self.people = People(pars=self.pars, age=p.age, sex=p.sex, method=p.method, barrier=p.barrier, debut_age=p.debut_age, fertile=p.fertile)
        return


    def update_methods(self):
        '''
        Update all contraceptive method matrices to have probabilities that follow a trend closest to the
        year the sim is on based on mCPR in that year
        '''

        methods = self['methods'] # Shorten methods
        methods['adjusted'] = sc.dcp(methods['raw']) # Avoids needing to copy this within loops later

        # Compute the trend in MCPR
        trend_years = methods['mcpr_years']
        trend_vals  = methods['mcpr_rates']
        ind      = sc.findnearest(trend_years, self.y) # The year of data closest to the sim year
        norm_ind = sc.findnearest(trend_years, self['mcpr_norm_year']) # The year we're using to normalize

        nearest_val = trend_vals[ind] # Nearest MCPR value from the data
        norm_val    = trend_vals[norm_ind] # Normalization value
        if self.y > max(trend_years): # We're after the last year of data: extrapolate
            eps = 1e-3 # Epsilon for lowest allowed MCPR value (to avoid divide by zero errors)
            nearest_year = trend_years[ind]
            year_diff  = self.y - nearest_year
            correction = self['mcpr_growth_rate']*year_diff # Project the change in MCPR
            extrapolated_val = nearest_val*(1 + correction) # Multiply the current value by the projection
            trend_val  = np.clip(extrapolated_val, eps, self['mcpr_max']) # Ensure it stays within bounds
        else: # Otherwise, just use the nearest data point
            trend_val = nearest_val
        norm_trend_val  = trend_val/norm_val # Normalize so the correction factor is 1 at the normalization year

        # Update annual (non-postpartum) population and postpartum switching matrices for current year mCPR - stratified by age
        for switchkey in ['annual', 'pp1to6']:
            for matrix in methods['adjusted'][switchkey].values():
                matrix[0, 0] /= norm_trend_val  # Takes into account mCPR during year of sim
                for i in range(len(matrix)):
                    denom = matrix[i,:].sum()
                    if denom > 0:
                        matrix[i,:] = matrix[i, :] / denom  # Normalize so probabilities add to 1

        # Update postpartum initiation matrices for current year mCPR - stratified by age
        for matrix in methods['adjusted']['pp0to1'].values():
            matrix[0] /= norm_trend_val  # Takes into account mCPR during year of sim
            matrix /= matrix.sum()

        return


    def update_mortality(self):
        ''' Update infant and maternal mortality for the sim's current year.  Update general mortality trend
        as this uses a spline interpolation instead of an array'''

        mapping = {
            'age_mortality':      'gen_trend',
            'infant_mortality':   'infant',
            'maternal_mortality': 'maternal',
            'stillbirth_rate':    'stillbirth',
        }

        self['mortality_probs'] = {}
        for key1,key2 in mapping.items():
            ind = sc.findnearest(self[key1]['year'], self.y)
            val = self[key1]['probs'][ind]
            self['mortality_probs'][key2] = val

        return


    def update_mothers(self):
        '''Add link between newly added individuals and their mothers'''
        all_ppl = self.people.unfilter()
        for mother_index, postpartum in enumerate(all_ppl.postpartum):
            if postpartum and all_ppl.postpartum_dur[mother_index] < 2:
                for child in all_ppl.children[mother_index]:
                    all_ppl.mothers[child] = mother_index
        return


    def apply_interventions(self):
        ''' Apply each intervention in the model '''
        from . import interventions as fpi # To avoid circular import
        for i,intervention in enumerate(sc.tolist(self['interventions'])):
            if isinstance(intervention, fpi.Intervention):
                if not intervention.initialized: # pragma: no cover
                    intervention.initialize(self)
                intervention.apply(self) # If it's an intervention, call the apply() method
            elif callable(intervention):
                intervention(self) # If it's a function, call it directly
            else: # pragma: no cover
                errormsg = f'Intervention {i} ({intervention}) is neither callable nor an Intervention object: it is {type(intervention)}'
                raise TypeError(errormsg)
        return


    def apply_analyzers(self):
        ''' Apply each analyzer in the model '''
        from . import analyzers as fpa # To avoid circular import
        for i,analyzer in enumerate(sc.tolist(self['analyzers'])):
            if isinstance(analyzer, fpa.Analyzer):
                if not analyzer.initialized: # pragma: no cover
                    analyzer.initialize(self)
                analyzer.apply(self) # If it's an intervention, call the apply() method
            elif callable(analyzer):
                analyzer(self) # If it's a function, call it directly
            else: # pragma: no cover
                errormsg = f'Analyzer {i} ({analyzer}) is neither callable nor an Analyzer object: it is {type(analyzer)}'
                raise TypeError(errormsg)
        return


    def run(self, verbose=None):
        ''' Run the simulation '''

        # Initialize -- reset settings and results
        T = sc.timer()
        if verbose is None:
            verbose = self['verbose']
        self.initialize()
        if self.already_run:
            errormsg = 'Cannot re-run an already run sim; please recreate or copy prior to a run'
            raise RuntimeError(errormsg)

        # Main simulation loop

        for i in range(self.npts):  # Range over number of timesteps in simulation (ie, 0 to 261 steps)
            self.i = i # Timestep
            self.t = self.ind2year(i)  # t is time elapsed in years given how many timesteps have passed (ie, 25.75 years)
            self.y = self.ind2calendar(i)  # y is calendar year of timestep (ie, 1975.75)

            # Print progress
            elapsed = T.toc(output=True)
            if verbose:
                simlabel = f'"{self.label}": ' if self.label else ''
                string = f'  Running {simlabel}{self.y:0.0f} of {self["end_year"]} ({i:2.0f}/{self.npts}) ({elapsed:0.2f} s) '
                if verbose >= 2:
                    sc.heading(string)
                elif verbose>0:
                    if not (self.t % int(1.0/verbose)):
                        sc.progressbar(self.t+1, self.npts, label=string, length=20, newline=True)

            # Update method matrices for year of sim to trend over years
            self.update_methods()

            # Update mortality probabilities for year of sim
            self.update_mortality()

            # Apply interventions
            self.apply_interventions()

            # Update the people
            self.people.i = self.i
            self.people.t = self.t
            step_results = self.people.update()
            r = fpu.dict2obj(step_results)

            # Start calculating results
            new_people = r.births - r.infant_deaths # Do not add agents who died before age 1 to population

            # Births
            data = self.make_people(n=new_people, age=np.zeros(new_people))

            people = People(pars=self.pars, n=new_people, **data)
            self.people += people

            # Update mothers
            if self.mother_ids:
                self.update_mothers()

            # Results
            percent0to5   = (r.pp0to5 / r.total_women_fecund) * 100
            percent6to11  = (r.pp6to11 / r.total_women_fecund) * 100
            percent12to23 = (r.pp12to23 / r.total_women_fecund) * 100
            nonpostpartum = ((r.total_women_fecund - r.pp0to5 - r.pp6to11 - r.pp12to23)/r.total_women_fecund) * 100

            # Store results
            if self['scaled_pop']:
                scale = self['scaled_pop']/self['n_agents']
            else:
                scale = 1
            self.results['t'][i]               = self.tvec[i]
            self.results['pop_size_months'][i] = self.n*scale
            self.results['births'][i]          = r.births*scale
            self.results['deaths'][i]          = r.deaths*scale
            self.results['stillbirths'][i]     = r.stillbirths*scale
            self.results['total_births'][i]    = r.total_births*scale
            self.results['maternal_deaths'][i] = r.maternal_deaths*scale
            self.results['infant_deaths'][i]   = r.infant_deaths*scale
            self.results['on_methods_mcpr'][i] = r.on_methods_mcpr
            self.results['no_methods_mcpr'][i] = r.no_methods_mcpr
            self.results['on_methods_cpr'][i]  = r.on_methods_cpr
            self.results['no_methods_cpr'][i]  = r.no_methods_cpr
            self.results['on_methods_acpr'][i] = r.on_methods_acpr
            self.results['no_methods_acpr'][i] = r.no_methods_acpr
            self.results['mcpr'][i]            = r.on_methods_mcpr/(r.no_methods_mcpr + r.on_methods_mcpr)
            self.results['cpr'][i]             = r.on_methods_cpr/(r.no_methods_cpr + r.on_methods_cpr)
            self.results['acpr'][i]            = r.on_methods_acpr/(r.no_methods_acpr + r.on_methods_acpr)
            self.results['pp0to5'][i]          = percent0to5
            self.results['pp6to11'][i]         = percent6to11
            self.results['pp12to23'][i]           = percent12to23
            self.results['nonpostpartum'][i]      = nonpostpartum
            self.results['total_women_fecund'][i] = r.total_women_fecund*scale
            self.results['unintended_pregs'][i]   = r.unintended_pregs*scale

            for agekey in fpd.age_bin_map.keys():
                births_key = f'total_births_{agekey}'
                women_key = f'total_women_{agekey}'
                self.results[births_key][i] = r.birth_bins[agekey]*scale # Store results of total births per age bin for ASFR
                self.results[women_key][i]  = r.age_bin_totals[agekey]*scale # Store results of total fecund women per age bin for ASFR

            # Store results of number of switching events in each age group
            if self['track_switching']:
                switch_events = step_results.pop('switching')
                self.results['switching_events_<18'][i]        = scale**scale*r.switching_annual['<18']
                self.results['switching_events_18-20'][i]      = scale*r.switching_annual['18-20']
                self.results['switching_events_21-25'][i]      = scale*r.switching_annual['21-25']
                self.results['switching_events_>25'][i]        = scale*r.switching_annual['>25']
                self.results['switching_events_pp_<18'][i]     = scale*r.switching_postpartum['<18']
                self.results['switching_events_pp_18-20'][i]   = scale*r.switching_postpartum['18-20']
                self.results['switching_events_pp_21-25'][i]   = scale*r.switching_postpartum['21-25']
                self.results['switching_events_pp_>25'][i]     = scale*r.switching_postpartum['>25']
                self.results['switching_events_annual'][i]     = scale*switch_events['annual']
                self.results['switching_events_postpartum'][i] = scale*switch_events['postpartum']

            # Calculate metrics over the last year in the model and save whole years and stats to an array
            if i % fpd.mpy == 0:
                self.results['tfr_years'].append(self.y)
                start_index = (int(self.t)-1)*fpd.mpy
                stop_index = int(self.t)*fpd.mpy
                unintended_pregs_over_year = scale*np.sum(self.results['unintended_pregs'][start_index:stop_index]) # Grabs sum of unintended pregnancies due to method failures over the last 12 months of calendar year
                infant_deaths_over_year    = scale*np.sum(self.results['infant_deaths'][start_index:stop_index])
                total_births_over_year     = scale*np.sum(self.results['total_births'][start_index:stop_index])
                live_births_over_year      = scale*np.sum(self.results['births'][start_index:stop_index])
                maternal_deaths_over_year  = scale*np.sum(self.results['maternal_deaths'][start_index:stop_index])
                self.results['pop_size'].append(scale*self.n) # CK: TODO: replace with arrays
                self.results['mcpr_by_year'].append(self.results['mcpr'][i])
                self.results['cpr_by_year'].append(self.results['cpr'][i])
                self.results['method_failures_over_year'].append(unintended_pregs_over_year)
                self.results['infant_deaths_over_year'].append(infant_deaths_over_year)
                self.results['total_births_over_year'].append(total_births_over_year)
                self.results['live_births_over_year'].append(live_births_over_year)
                self.results['maternal_deaths_over_year'].append(maternal_deaths_over_year)
                if maternal_deaths_over_year == 0:
                    self.results['mmr'].append(0)
                else:
                    maternal_mortality_ratio = maternal_deaths_over_year / live_births_over_year * 100000
                    self.results['mmr'].append(maternal_mortality_ratio)
                if infant_deaths_over_year == 0:
                    self.results['imr'].append(infant_deaths_over_year)
                else:
                    infant_mortality_rate = infant_deaths_over_year / live_births_over_year * 1000
                    self.results['imr'].append(infant_mortality_rate)

                tfr = 0
                for key in fpd.age_bin_map.keys():
                    age_bin_births_year = np.sum(self.results['total_births_'+key][start_index:stop_index])
                    age_bin_total_women_year = self.results['total_women_'+key][stop_index]
                    age_bin_births_per_woman = sc.safedivide(age_bin_births_year, age_bin_total_women_year)
                    self.results['asfr'][key].append(age_bin_births_per_woman*1000)
                    tfr += age_bin_births_per_woman # CK: TODO: check if this is right

                self.results['tfr_rates'].append(tfr*5) # CK: TODO: why *5?

            if self.test_mode:
                self.log_daily_totals()

        if self.test_mode:
            self.save_daily_totals()

        if not self.mother_ids:
            delattr(self.people, "mothers")

        # Apply analyzers
        self.apply_analyzers()

        # Convert all results to Numpy arrays
        for key,arr in self.results.items():
            if isinstance(arr, list):
                self.results[key] = np.array(arr) # Convert any lists to arrays

        # Calculate cumulative totals
        self.results['cum_maternal_deaths_by_year'] = np.cumsum(self.results['maternal_deaths_over_year'])
        self.results['cum_infant_deaths_by_year']   = np.cumsum(self.results['infant_deaths_over_year'])
        self.results['cum_live_births_by_year']     = np.cumsum(self.results['live_births_over_year'])

        # Convert to an objdict for easier access
        self.results = sc.objdict(self.results)

        if verbose:
            print(f'Final population size: {self.n}.')
            elapsed = T.toc(output=True)
            print(f'Run finished for "{self.label}" after {elapsed:0.1f} s')

        self.already_run = True

        return self


    def store_postpartum(self):

        '''Stores snapshot of who is currently pregnant, their parity, and various
        postpartum states in final step of model for use in calibration'''

        min_age = 12.5
        max_age = self['age_limit_fecundity']

        ppl = self.people
        rows = []
        for i in range(len(ppl)):
            if ppl.alive[i] and ppl.sex[i] == 0 and min_age <= ppl.age[i] < max_age:
                row = {'Age': None, 'PP0to5': None, 'PP6to11': None, 'PP12to23': None, 'NonPP': None, 'Pregnant': None, 'Parity': None}
                row['Age'] = int(round(ppl.age[i]))
                row['NonPP'] = 1 if not ppl.postpartum[i] else 0
                if ppl.postpartum[i]:
                    pp_dur = ppl.postpartum_dur[i]
                    row['PP0to5'] = 1 if 0 <= pp_dur < 6 else 0
                    row['PP6to11'] = 1 if 6 <= pp_dur < 12 else 0
                    row['PP12to23'] = 1 if 12 <= pp_dur <= 24 else 0
                row['Pregnant'] = 1 if ppl.pregnant[i] else 0
                row['Parity'] = ppl.parity[i]
                rows.append(row)

        pp = pd.DataFrame(rows, index = None, columns = ['Age', 'PP0to5', 'PP6to11', 'PP12to23', 'NonPP', 'Pregnant', 'Parity'])
        pp.fillna(0, inplace=True)
        return pp


    def to_df(self):
        '''
        Export all sim results to a dataframe
        '''
        raw_res = sc.odict(defaultdict=list)
        for reskey in self.results.keys():
            res = self.results[reskey]
            if sc.isarray(res) and len(res) == self.npts:
                raw_res[reskey] += res.tolist()
        df = pd.DataFrame(raw_res)
        self.df = df
        return df


    def plot(self, do_save=None, do_show=True, fig_args=None, plot_args=None, axis_args=None, fill_args=None,
             label=None, new_fig=True):
        '''
        Plot the results -- can supply arguments for both the figure and the plots.

        Args:
            dosave    (bool): Whether or not to save the figure. If a string, save to that filename.
            doshow    (bool): Whether to show the plots at the end
            figargs   (dict): Passed to pl.figure()
            plot_args (dict): Passed to pl.plot()
            axis_args (dict): Passed to pl.subplots_adjust()
            fill_args (dict): Passed to pl.fill_between())
            label     (str):  Label to override default
            new_fig   (bool): whether to create a new figure (true unless part of a multisim)
        '''

        fig_args  = sc.mergedicts(dict(figsize=(16,10)), fig_args)
        plot_args = sc.mergedicts(dict(lw=2, alpha=0.7), plot_args)
        axis_args = sc.mergedicts(dict(left=0.1, bottom=0.05, right=0.9, top=0.97, wspace=0.2, hspace=0.25), axis_args)
        fill_args = sc.mergedicts(dict(alpha=0.2), fill_args)

        fig = pl.figure(**fig_args) if new_fig else pl.gcf()
        pl.subplots_adjust(**axis_args)

        res = self.results # Shorten since heavily used

        x = res['tfr_years'] # Likewise

        # Plot everything
        to_plot = sc.odict({
            'mCPR':               sc.odict({'mcpr_by_year': 'Modern contraceptive prevalence rate (%)'}),
            'Maternal mortality ratio': sc.odict({'mmr': 'Maternal mortality ratio'}),
            'Infant mortality rate': sc.odict({'imr': 'Infant mortality rate'}),
            'Cumulative live births': sc.odict({'cum_live_births_by_year': 'Live births'}),
            'Cumulative maternal deaths': sc.odict({'cum_maternal_deaths_by_year': 'Maternal deaths'}),
            'Cumulative infant deaths': sc.odict({'cum_infant_deaths_by_year':  'Infant deaths'}),
            })
        for p,title,keylabels in to_plot.enumitems():
            ax = pl.subplot(2,3,p+1)
            for i,key,reslabel in keylabels.enumitems():
                this_res = res[key]
                is_dist = hasattr(this_res, 'best')
                if is_dist:
                    y, low, high = this_res.best, this_res.low, this_res.high
                else:
                    y, low, high = this_res, None, None

                if key == 'mcpr_by_year':
                    y *= 100
                    if is_dist:
                        low *= 100
                        high *= 100

                # Handle label
                if label is not None:
                    plotlabel = label
                else:
                    if new_fig: # It's a new figure, use the result label
                        plotlabel = reslabel
                    else: # Replace with sim label to avoid duplicate labels
                        plotlabel = self.label

                # Actually plot
                ax.plot(x, y, label=plotlabel, **plot_args)
                if is_dist:
                    if 'c' in plot_args:
                        fill_args['facecolor'] = plot_args['c']
                    ax.fill_between(x, low, high, **fill_args)

            # Handle annotations
            fpu.fixaxis(useSI=fpd.useSI, set_lim=new_fig) # If it's not a new fig, don't set the lim
            if key == 'mcpr_by_year':
                pl.ylabel('Percentage')
            elif key == 'mmr':
                pl.ylabel('Deaths per 100,000 live births')
            elif key == 'imr':
                pl.ylabel('Deaths per 1,000 live births')
            else:
                pl.ylabel('Count')
            pl.xlabel('Year')
            pl.title(title, fontweight='bold')

        # Ensure the figure actually renders or saves
        if do_save:
            if isinstance(do_save, str):
                filename = do_save # It's a string, assume it's a filename
            else:
                filename = 'fpsim.png' # Just give it a default name
            pl.savefig(filename)
        if do_show:
            pl.show() # Only show if we're not saving

        return fig


    def plot_cpr(self, do_save=None, do_show=True, fig_args=None, plot_args=None, axis_args=None, fill_args=None,
             label=None, new_fig=True):
        '''
        Plot the results -- can supply arguments for both the figure and the plots.

        Args:
            dosave    (bool): Whether or not to save the figure. If a string, save to that filename.
            doshow    (bool): Whether to show the plots at the end
            figargs   (dict): Passed to pl.figure()
            plot_args (dict): Passed to pl.plot()
            axis_args (dict): Passed to pl.subplots_adjust()
            fill_args (dict): Passed to pl.fill_between())
            label     (str):  Label to override default
            new_fig   (bool): whether to create a new figure (true unless part of a multisim)
        '''

        fig_args  = sc.mergedicts(dict(figsize=(20,8)), fig_args)
        plot_args = sc.mergedicts(dict(lw=2, alpha=0.7), plot_args)
        axis_args = sc.mergedicts(dict(left=0.1, bottom=0.05, right=0.9, top=0.97, wspace=0.2, hspace=0.25), axis_args)
        fill_args = sc.mergedicts(dict(alpha=0.2), fill_args)

        fig = pl.figure(**fig_args) if new_fig else pl.gcf()
        pl.subplots_adjust(**axis_args)

        res = self.results # Shorten since heavily used

        x = res['t'] # Likewise

        # Plot everything
        to_plot = sc.odict({
            'mCPR \n (modern method users \namong all women 15-49)':               sc.odict({'mcpr': 'Modern contraceptive prevalence rate'}),
            'CPR \n (all method users \namong all women 15-49)': sc.odict({'cpr': 'Contraceptive prevalence rate'}),
            'ACPR \n (all method users \namong nonpregnant, sexually active women 15-49)': sc.odict({'acpr': 'Alternative contraceptive prevalence rate'}),
            })

        ax = None
        for p,title,keylabels in to_plot.enumitems():
            ax = pl.subplot(1, 3, p+1, sharey=ax)
            for i,key,reslabel in keylabels.enumitems():
                this_res = res[key]
                is_dist = hasattr(this_res, 'best')
                if is_dist:
                    y, low, high = this_res.best, this_res.low, this_res.high
                else:
                    y, low, high = this_res, None, None

                y *= 100
                if is_dist:
                    low *= 100
                    high *= 100

                # Handle label
                if label is not None:
                    plotlabel = label
                else:
                    if new_fig: # It's a new figure, use the result label
                        plotlabel = reslabel
                    else: # Replace with sim label to avoid duplicate labels
                        plotlabel = self.label

                ax.plot(x, y, label=plotlabel, **plot_args)
                if is_dist:
                    if 'c' in plot_args:
                        fill_args['facecolor'] = plot_args['c']
                    ax.fill_between(x, low, high, **fill_args)

            pl.ylabel('Percentage')
            pl.xlabel('Year')
            pl.title(title, fontweight='bold')
            fpu.fixaxis(useSI=fpd.useSI, set_lim=new_fig) # If it's not a new fig, don't set the lim

        # Ensure the figure actually renders or saves
        sc.figlayout()
        if do_save:
            if isinstance(do_save, str):
                filename = do_save # It's a string, assume it's a filename
            else:
                filename = 'fpsim.png' # Just give it a default name
            pl.savefig(filename)
        if do_show:
            pl.show() # Only show if we're not saving

        return fig


    def plot_age_first_birth(self, do_show=False, do_save=True, output_file="first_birth_age.png"):
        to_plot = [age for age in self.people.first_birth_age if age is not None]

        sns.set(rc={'figure.figsize':(7,5)})
        pl.title("Age at first birth")
        sns.boxplot(x=to_plot, orient='v', notch=True)
        if do_show:
            pl.show()
        if do_save:
            print(f"Saved age at first birth plot at {output_file}")
            pl.savefig(output_file)


    def plot_people(self):
        ''' Use imshow() to show all individuals as rows, with time as columns, one pixel per timestep per person '''
        # The test_mode might help with this
        raise NotImplementedError

    # Used in verbose model
    def log_daily_totals(self):
        pass

    def save_daily_totals(self):
        pass


class MultiSim(sc.prettyobj):
    '''
    The MultiSim class handles the running of multiple simulations
    '''

    def __init__(self, sims=None, base_sim=None, label=None, n=None, **kwargs):

        # Handle inputs
        if base_sim is None:
            if isinstance(sims, Sim):
                base_sim = sims
                sims = None
            elif isinstance(sims, list):
                base_sim = sims[0]
            else:
                errormsg = f'If base_sim is not supplied, sims must be either a single sim (treated as base_sim) or a list of sims, not {type(sims)}'
                raise TypeError(errormsg)

        # Set properties
        self.sims      = sims
        self.base_sim  = base_sim
        self.label     = base_sim.label if (label is None and base_sim is not None) else label
        self.run_args  = sc.mergedicts(kwargs)
        self.results   = None
        self.which     = None # Whether the multisim is to be reduced, combined, etc.
        self.already_run = False
        fpu.set_metadata(self) # Set version, date, and git info

        return


    def __len__(self):
        try:
            return len(self.sims)
        except:
            return 0


    def run(self, compute_stats=True, **kwargs):
        self.sims = multi_run(self.sims, **kwargs)
        if compute_stats:
            self.compute_stats()
        self.already_run = True
        return self


    def compute_stats(self, return_raw=False, quantiles=None, use_mean=False, bounds=None):
        ''' Compute statistics across multiple sims '''

        if use_mean:
            if bounds is None:
                bounds = 1
        else:
            if quantiles is None:
                quantiles = {'low':0.1, 'high':0.9}
            if not isinstance(quantiles, dict):
                try:
                    quantiles = {'low':float(quantiles[0]), 'high':float(quantiles[1])}
                except Exception as E:
                    errormsg = f'Could not figure out how to convert {quantiles} into a quantiles object: must be a dict with keys low, high or a 2-element array ({str(E)})'
                    raise ValueError(errormsg)

        base_sim = sc.dcp(self.sims[0])
        raw = sc.objdict()
        results = sc.objdict()
        axis = 1
        start_end = np.array([sim.tvec[[0, -1]] for sim in self.sims])
        if len(np.unique(start_end)) != 2:
            errormsg = f'Cannot compute stats for sims: start and end values do not match:\n{start_end}'
            raise ValueError(errormsg)

        reskeys = list(base_sim.results.keys())
        for key in ['t', 'tfr_years']: # Don't compute high/low for these
            results[key] = base_sim.results[key]
            reskeys.remove(key)
        for reskey in reskeys:
            if isinstance(base_sim.results[reskey], dict):
                if return_raw:
                    for s, sim in enumerate(self.sims):
                        raw[reskey][s] = base_sim.results[reskey]
            else:
                results[reskey] = sc.objdict()
                npts = len(base_sim.results[reskey])
                raw[reskey] = np.zeros((npts, len(self.sims)))
                for s,sim in enumerate(self.sims):
                    raw[reskey][:, s] = sim.results[reskey] # Stack into an array for processing

                if use_mean:
                    r_mean = np.mean(raw[reskey], axis=axis)
                    r_std = np.std(raw[reskey], axis=axis)
                    results[reskey].best = r_mean
                    results[reskey].low = r_mean - bounds * r_std
                    results[reskey].high = r_mean + bounds * r_std
                else:
                    results[reskey].best = np.quantile(raw[reskey], q=0.5, axis=axis)
                    results[reskey].low = np.quantile(raw[reskey], q=quantiles['low'], axis=axis)
                    results[reskey].high = np.quantile(raw[reskey], q=quantiles['high'], axis=axis)

        self.results = results
        self.base_sim.results = results # Store here too, to enable plotting

        if return_raw:
            return raw
        else:
            return


    @staticmethod
    def merge(*args, base=False):
        '''
        Convenience method for merging two MultiSim objects.

        Args:
            args (MultiSim): the MultiSims to merge (either a list, or separate)
            base (bool): if True, make a new list of sims from the multisim's two base sims; otherwise, merge the multisim's lists of sims

        Returns:
            msim (MultiSim): a new MultiSim object

        **Examples**:

            mm1 = fp.MultiSim.merge(msim1, msim2, base=True)
            mm2 = fp.MultiSim.merge([m1, m2, m3, m4], base=False)
        '''

        # Handle arguments
        if len(args) == 1 and isinstance(args[0], list):
            args = args[0] # A single list of MultiSims has been provided

        # Create the multisim from the base sim of the first argument
        msim = MultiSim(base_sim=sc.dcp(args[0].base_sim), sims=[], label=args[0].label)
        msim.sims = []
        msim.chunks = [] # This is used to enable automatic splitting later

        # Handle different options for combining
        if base: # Only keep the base sims
            for i,ms in enumerate(args):
                sim = sc.dcp(ms.base_sim)
                sim.label = ms.label
                msim.sims.append(sim)
                msim.chunks.append([[i]])
        else: # Keep all the sims
            for ms in args:
                len_before = len(msim.sims)
                msim.sims += list(sc.dcp(ms.sims))
                len_after= len(msim.sims)
                msim.chunks.append(list(range(len_before, len_after)))

        return msim


    def split(self, inds=None, chunks=None):
        '''
        Convenience method for splitting one MultiSim into several. You can specify
        either individual indices of simulations to extract, via inds, or consecutive
        chunks of indices, via chunks. If this function is called on a merged MultiSim,
        the chunks can be retrieved automatically and no arguments are necessary.

        Args:
            inds (list): a list of lists of indices, with each list turned into a MultiSim
            chunks (int or list): if an int, split the MultiSim into that many chunks; if a list return chunks of that many sims

        Returns:
            A list of MultiSim objects

        **Examples**::

            m1 = fp.MultiSim(fp.Sim(label='sim1'))
            m2 = fp.MultiSim(fp.Sim(label='sim2'))
            m3 = fp.MultiSim.merge(m1, m2)
            m3.run()
            m1b, m2b = m3.split()

            msim = fp.MultiSim(fp.Sim(), n_runs=6)
            msim.run()
            m1, m2 = msim.split(inds=[[0,2,4], [1,3,5]])
            mlist1 = msim.split(chunks=[2,4]) # Equivalent to inds=[[0,1], [2,3,4,5]]
            mlist2 = msim.split(chunks=2) # Equivalent to inds=[[0,1,2], [3,4,5]]
        '''

        # Process indices and chunks
        if inds is None: # Indices not supplied
            if chunks is None: # Chunks not supplied
                if hasattr(self, 'chunks'): # Created from a merged MultiSim
                    inds = self.chunks
                else: # No indices or chunks and not created from a merge
                    errormsg = 'If a MultiSim has not been created via merge(), you must supply either inds or chunks to split it'
                    raise ValueError(errormsg)
            else: # Chunks supplied, but not inds
                inds = [] # Initialize
                sim_inds = np.arange(len(self)) # Indices for the simulations
                if sc.isiterable(chunks): # e.g. chunks = [2,4]
                    chunk_inds = np.cumsum(chunks)[:-1]
                    inds = np.split(sim_inds, chunk_inds)
                else: # e.g. chunks = 3
                    inds = np.split(sim_inds, chunks) # This will fail if the length is wrong

        # Do the conversion
        mlist = []
        for indlist in inds:
            sims = sc.dcp([self.sims[i] for i in indlist])
            msim = MultiSim(sims=sims)
            mlist.append(msim)

        return mlist


    def remerge(self, base=True, **kwargs):
        '''
        Split a sim, compute stats, and re-merge.

        Args:
            base (bool): whether to use the base sim (otherwise, has no effect)
            kwargs (dict): passed to msim.split()

        Note: returns a new MultiSim object (if that concerns you).
        '''
        ms = self.split(**kwargs)
        for m in ms:
            m.compute_stats() # Recompute the statistics on each separate MultiSim
        out = MultiSim.merge(*ms, base=base) # Now re-merge, this time using the base_sim
        return out


    def to_df(self):
        '''
        Export all individual sim results to a dataframe
        '''
        raw_res = sc.odict(defaultdict=list)
        for s,sim in enumerate(self.sims):
            for reskey in sim.results.keys():
                res = sim.results[reskey]
                if sc.isarray(res) and len(res) == sim.npts:
                    raw_res[reskey] += res.tolist()
            raw_res['sim'] += [s]*sim.npts
            raw_res['sim_label'] += [sim.label]*sim.npts
        df = pd.DataFrame(raw_res)
        self.df = df
        return df


    def plot(self, do_show=True, plot_sims=True, fig_args=None, plot_args=None, plot_cpr=False, **kwargs):
        '''
        Plot the MultiSim
        '''
        fig_args = sc.mergedicts(dict(figsize=(16,10)), fig_args)

        if plot_sims:
            fig = pl.figure(**fig_args)
            do_show = kwargs.pop('do_show', True)
            labels = sc.autolist()
            labellist = sc.autolist() # TODO: shouldn't need this
            for sim in self.sims: # Loop over and find unique labels
                if sim.label not in labels:
                    labels += sim.label
                    labellist += sim.label
                    label = sim.label
                else:
                    labellist += ''
                n_unique = len(np.unique(labels)) # How many unique sims there are
            colors = sc.gridcolors(n_unique)
            colors = {k:c for k,c in zip(labels, colors)}
            for s,sim in enumerate(self.sims): # Note: produces duplicate legend entries
                label = labellist[s]
                n_unique = len(labels) # How many unique sims there are
                color = colors[sim.label]
                alpha = max(0.2, 1/np.sqrt(n_unique))
                sim_plot_args = sc.mergedicts(dict(alpha=alpha, c=color), plot_args)
                kw = dict(new_fig=False, do_show=False, label=label, plot_args=sim_plot_args)
                if plot_cpr:
                    sim.plot_cpr(**kw, **kwargs)
                else:
                    sim.plot(**kw, **kwargs)
            if do_show:
                pl.show()
            return fig
        else:
            return self.base_sim.plot(do_show=do_show, fig_args=fig_args, plot_args=plot_args, **kwargs)


    def plot_cpr(self, *args, **kwargs):
        ''' Plot the contraceptive prevalence rate '''
        return self.plot(*args, **kwargs, plot_cpr=True)


    def plot_method_mix(self, n_sims=10, do_show=False, do_save=True, filepath="method_mix.png"):
        """
        Plots the average method mix for n_sims runs

        Args:
            n_sims   (int): The number of sims you want to run to calculate average mix and standard deviation.
            do_show (bool): Whether or not the user wants to show the output plot.
            do_save (bool): Whether or not the user wants to save the plot to filepath.
            filepath (str): The name of the path to output the plot.
        """
        method_table = {"sim" : [], "sim_index": [], "proportion": [], "method": []}

        # Run each sim n_sims times, get save proportion and let barplot calculate averages
        for sim in self.sims:
            print(f"Processing sim: {sim.label}")
            sim_run_list = [0] * n_sims
            for sim_index in range(n_sims):
                new_sim = sc.dcp(sim) # CK: TODO: should not need to be copied
                new_sim.pars['seed'] = sim_index
                sim_run_list[sim_index] = new_sim

            multi = MultiSim(sims=sim_run_list)
            multi.run() # CK: TODO: should not need to be run

            for sim_index in range(n_sims):
                people = multi.sims[sim_index].people
                unique, counts = np.unique(people.method, return_counts=True)
                count_dict = dict(zip(unique, counts))

                for method in count_dict:
                    if method != 0:
                        method_table["proportion"].append(count_dict[method] / len(people.method))
                        method_table["sim_index"].append(sim_index)
                        method_table["method"].append(method)
                        method_table["sim"].append(sim.label)

        # Plotting
        df = pd.DataFrame(method_table) # Makes it a bit easier to subset for bar charts

        # We want names for the methods
        methods_map = self.sims[0].pars['methods']['map']
        inv_methods_map = {value: key for key, value in methods_map.items()}
        df['method'] = df['method'].map(inv_methods_map)

        # plotting and saving
        sns.set(rc={'figure.figsize':(12,8.27)})
        sns.barplot(data=df, x="proportion", y="method", estimator=np.mean, hue="sim", ci="sd", order=['Implants', 'Injectables', 'Pill', 'IUDs', 'Other traditional', 'Condoms', "BTL", 'Other modern', 'Withdrawal'])
        pl.title(f"Mean method mix over {n_sims} sims")

        if do_save:
            pl.savefig(filepath)
        if do_show:
            pl.show()

    def plot_age_first_birth(self, do_show=False, do_save=True, output_file='age_first_birth_multi.png'):
        length = sum([len([num for num in sim.people.first_birth_age if num is not None]) for sim in self.sims])
        print(f"Length of total is: {length}")
        data_dict = {"age": [0] * length, "sim": [0] * length}
        i = 0
        for sim in self.sims:
            for value in [num for num in sim.people.first_birth_age if num is not None]:
                data_dict['age'][i] = value
                data_dict['sim'][i] = sim.label
                i = i + 1

        data = pd.DataFrame(data_dict)
        pl.title("Age at first birth")
        sns.boxplot(data=data, y='age', x='sim', orient='v', notch=True)
        if do_show:
            pl.show()
        if do_save:
            print(f"Saved age at first birth plot at {output_file}")
            pl.savefig(output_file)


def single_run(sim):
    ''' Helper function for multi_run(); rarely used on its own '''
    sim.run()
    return sim


def multi_run(sims, **kwargs):
    ''' Run multiple sims in parallel; usually used via the MultiSim class, not directly '''
    sims = sc.parallelize(single_run, iterarg=sims, **kwargs)
    return sims


def parallel(*args, **kwargs):
    '''
    A shortcut to ``fp.MultiSim()``, allowing the quick running of multiple simulations
    at once.

    Args:
        args (list): The simulations to run
        kwargs (dict): passed to multi_run()

    Returns:
        A run MultiSim object.

    **Examples**::

        s1 = fp.Sim(exposure_factor=0.5, label='Low')
        s2 = fp.Sim(exposure_factor=2.0, label='High')
        fp.parallel(s1, s2).plot()
        msim = fp.parallel(s1, s2)
    '''
    sims = sc.mergelists(*args)
    return MultiSim(sims=sims).run(**kwargs)
