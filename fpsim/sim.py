'''
Defines the Sim class, the core class of the FP model (FPsim).
'''

#%% Imports
import math #needed to round up with math.ceil
import numpy as np # Needed for a few things not provided by pl
import pylab as pl
import sciris as sc
import pandas as pd
import pyarrow.feather as feather
from . import defaults as fpd
from . import utils as fpu
from . import base as fpb
from . import interventions as fpi
from collections import defaultdict


# Specify all externally visible things this file defines
__all__ = ['People', 'Sim', 'MultiSim']


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
            n = int(self.pars['n'])

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

        # #Socio-demographic
        # self.wealth   = arr(n, d['wealth'])
        # self.cluster  = arr(n, d['cluster'])
        # self.urban    = arr(n, d['urban'])

        # Sexual and reproductive history
        self.sexually_active = arr(n, d['sexually_active'])
        self.lactating       = arr(n, d['lactating'])
        self.gestation       = arr(n, d['gestation'])
        self.preg_dur        = arr(n, d['preg_dur'])
        self.stillbirth      = arr(n, d['stillbirth'])
        self.postpartum      = arr(n, d['postpartum'])
        self.postpartum_dur  = arr(n, d['postpartum_dur']) # Tracks # months postpartum
        self.lam             = arr(n, d['lam']) # Separately tracks lactational amenorrhea, can be using both LAM and another method
        self.children        = arr(n, []) # Indices of children -- list of lists
        self.dobs            = arr(n, []) # Dates of births -- list of lists
        self.still_dates     = arr(n, []) # Dates of stillbirths -- list of lists
        self.breastfeed_dur  = arr(n, d['breastfeed_dur'])
        self.breastfeed_dur_total = arr(n, d['breastfeed_dur_total'])

        # Fecundity variation
        fv = [self.pars['fecundity_variation_low'], self.pars['fecundity_variation_high']]
        self.personal_fecundity = arr(n, np.random.random(n)*(fv[1]-fv[0])+fv[0]) # Stretch fecundity by a factor bounded by [f_var[0], f_var[1]]
        self.remainder_months = arr(n, d['remainder_months'])

        # Store keys
        final_states = dir(self)
        self._keys = [s for s in final_states if s not in init_states]

        return


    def get_method(self):
        '''
        Uses a switching matrix from DHS data to decide based on a person's original method their probability of changing to a
        new method and assigns them the new method. Currently allows switching on whole calendar years to enter function.
        Matrix serves as an initiation, discontinuation, continuation, and switching matrix. Transition probabilities are for 1 year and
        only for women who have not given birth within the last 6 months.
        '''
        methods = self.pars['methods']
        orig_methods = self.method

        # Method switching depends both on agent age and also on their current method, so we need to loop over both
        for m in methods['map'].values():
            for key,(age_low, age_high) in fpd.method_age_mapping.items():
                match_m    = (orig_methods == m)
                match_low  = (self.age >= age_low)
                match_high = (self.age <  age_high)
                match = match_m * match_low * match_high
                this_method = self.filter(match)

                matrix = self.pars['methods'][key]
                choices = matrix[m]
                choices = choices/choices.sum()
                new_methods = fpu.n_multinomial(choices, len(this_method))
                this_method.method = np.array(new_methods, dtype=np.int64)

        return


    def get_method_postpartum(self):
        '''Utilizes data from birth to allow agent to initiate a method postpartum coming from birth by
         3 months postpartum and then initiate, continue, or discontinue a method by 6 months postpartum.
        Next opportunity to switch methods will be on whole calendar years, whenever that falls.
        '''
        # TODO- Probabilities need to be adjusted for postpartum women on the next annual draw in "get_method" since they may be less than one year

        # Probability of initiating a postpartum method at 0-3 months postpartum
        # Transitional probabilities are for the first 3 month time period after delivery from DHS data

        pp_methods = self.pars['methods_postpartum']
        pp_switch  = self.pars['methods_postpartum_switch']
        orig_methods = self.method

        postpartum1 = (self.postpartum_dur == 0)
        postpartum6 = (self.postpartum_dur == 6)

        # In first time step after delivery, choice is by age but not previous method (since just gave birth)
        # All women are coming from birth and on no method to start, either will stay on no method or initiate a method
        for key, (age_low, age_high) in fpd.method_age_mapping.items():
            match_low = (self.age >= age_low)
            match_high = (self.age < age_high)
            match = (self.postpartum * postpartum1 * match_low * match_high * (self.parity < self.pars['high_parity']))
            match_high_parity = (self.postpartum * postpartum1 * match_low * match_high * (
                        self.parity >= self.pars['high_parity']))
            this_method = self.filter(match)
            this_method_high_parity = self.filter(match_high_parity)

            choices = pp_methods[key]
            choices_high_parity = sc.dcp(choices)
            choices_high_parity[0] *= self.pars['high_parity_nonuse_correction']
            choices_high_parity = choices_high_parity / choices_high_parity.sum()
            new_methods = fpu.n_multinomial(choices, len(this_method))
            new_methods_high_parity = fpu.n_multinomial(choices_high_parity, len(this_method_high_parity))
            this_method.method = np.array(new_methods, dtype=np.int64)
            this_method_high_parity.method = np.array(new_methods_high_parity, dtype=np.int64)

        # At 6 months, choice is by previous method and by age
        # Allow initiation, switching, or discontinuing with matrix at 6 months postpartum
        # Transitional probabilities are for 5 months, 1-6 months after delivery from DHS data
        for m in pp_methods['map'].values():
            for key,(age_low, age_high) in fpd.method_age_mapping.items():
                match_m    = (orig_methods == m)
                match_low  = (self.age >= age_low)
                match_high = (self.age <  age_high)
                match = match_m * self.postpartum * postpartum6 * match_low * match_high
                this_method = self.filter(match)

                matrix = pp_switch[key]
                choices = matrix[m]
                new_methods = fpu.n_multinomial(choices, len(this_method))
                this_method.method = np.array(new_methods, dtype=np.int64)

        return


    def check_mortality(self):
        '''Decide if person dies at a timestep'''

        timestep = self.pars['timestep']
        trend_val = self.pars['mortality_probs']['gen_trend']
        age_mort = self.pars['age_mortality']
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
            died.alive = False
            self.step_results['deaths'] += len(died)

        return


    def check_sexually_active(self):
        '''
        Decide if agent is sexually active based either on month postpartum or age if
        not postpartum.  Postpartum and general age-based data from DHS.
        '''
        # Set postpartum probabilities
        match_low  = self.postpartum_dur >= 0
        match_high = self.postpartum_dur <= self.pars['postpartum_length']
        match = self.postpartum * match_low * match_high
        pp = self.filter(match)
        non_pp = self.filter(~match)

        # Adjust for postpartum women's birth spacing preferences
        pref = self.pars['pref_spacing'] # Shorten since used a lot
        spacing_bins = pp.postpartum_dur / pref['interval'] # Main calculation -- divide the duration by the interval
        spacing_bins = np.array(np.minimum(spacing_bins, pref['n_bins']), dtype=int) # Convert to an integer and bound by longest bin
        probs_pp = self.pars['sexual_activity_postpartum']['percent_active'][pp.postpartum_dur]
        probs_pp *= pref['preference'][spacing_bins] # Actually adjust the probability -- check the overall probability with print(pref['preference'][spacing_bins].mean())

        # Set non-postpartum probabilities
        probs_non_pp = self.pars['sexual_activity'][non_pp.int_age]

        # Evaluate likelihood in this time step of being sexually active
        # Can revert to active or not active each timestep
        pp.sexually_active = fpu.binomial_arr(probs_pp)
        non_pp.sexually_active = fpu.binomial_arr(probs_non_pp)

        return


    def check_conception(self):
        '''
        Decide if person (female) becomes pregnant at a timestep.
        '''
        all_ppl = self.unfilter() # For complex array operations
        active = self.filter(self.sexually_active)
        lam         = active.filter(active.lam)
        nonlam      = active.filter(~active.lam)
        preg_probs = np.zeros(len(all_ppl)) # Use full array

        # Find monthly probability of pregnancy based on fecundity and any use of contraception including LAM - from data
        preg_eval_lam     = self.pars['age_fecundity'][lam.int_age_clip] * lam.personal_fecundity
        preg_eval_nonlam  = self.pars['age_fecundity'][nonlam.int_age_clip] * nonlam.personal_fecundity
        method_eff  = self.pars['method_efficacy'][nonlam.method]
        lam_eff     = self.pars['LAM_efficacy']

        lam_probs    = fpu.annprob2ts((1-lam_eff)*preg_eval_lam,       self.pars['timestep'])
        nonlam_probs = fpu.annprob2ts((1-method_eff)*preg_eval_nonlam, self.pars['timestep'])
        preg_probs[lam.inds]    = lam_probs
        preg_probs[nonlam.inds] = nonlam_probs

        # Adjust for decreased likelihood of conception if nulliparous vs already gravid - from data
        nullip = active.filter(active.parity == 0) # Nulliparous
        preg_probs[nullip.inds] *= self.pars['fecundity_ratio_nullip'][nullip.int_age_clip]

        # Adjust for probability of exposure to pregnancy episode at this timestep based on age and parity - encapsulates background factors - experimental and tunable
        preg_probs *= self.pars['exposure_correction']
        preg_probs *= self.pars['exposure_correction_age'][all_ppl.int_age_clip]
        preg_probs *= self.pars['exposure_correction_parity'][np.minimum(all_ppl.parity, fpd.max_parity)]

        # Use a single binomial trial to check for conception successes this month
        conceived = active.binomial(preg_probs[active.inds], as_filter=True)
        unintended = conceived.filter(conceived.method != 0)
        self.step_results['unintended_pregs'] += len(unintended)

        # Check for abortion
        is_abort = conceived.binomial(self.pars['abortion_prob'])
        abort = conceived.filter(is_abort)
        preg = conceived.filter(~is_abort)

        # Update states
        abort.postpartum = False
        abort.postpartum_dur = 0

        preg.pregnant = True
        preg.gestation = 1  # Start the counter at 1
        pregdur = [self.pars['preg_dur_low'], self.pars['preg_dur_high']]
        preg.preg_dur = np.random.randint(pregdur[0], pregdur[1]+1, size=len(preg))  # Duration of this pregnancy
        preg.postpartum = False
        preg.postpartum_dur = 0
        preg.reset_breastfeeding() # Stop lactating if becoming pregnant
        preg.method = 0

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
        bfdur = [self.pars['breastfeeding_dur_low'], self.pars['breastfeeding_dur_high']]
        bf_mu, bf_beta = 10.66828+9, 7.2585
        breastfeed_durs = abs(np.random.gumbel(bf_mu, bf_beta, size = len(self)))
        breastfeed_durs = [math.ceil(number) for number in breastfeed_durs]
        breastfeed_finished_inds = self.breastfeed_dur >= breastfeed_durs
        breastfeed_finished = self.filter(breastfeed_finished_inds)
        breastfeed_continue = self.filter(~breastfeed_finished_inds)
        breastfeed_finished.reset_breastfeeding()
        breastfeed_continue.breastfeed_dur += self.pars['timestep']
        return


    def update_postpartum(self):
        '''Track duration of extended postpartum period (0-24 months after birth).  Only enter this function if agent is postpartum'''

        # Stop postpartum episode if reach max length (set to 24 months)
        pp_done = self.filter(self.postpartum_dur >= self.pars['postpartum_length'])
        pp_done.postpartum = False
        pp_done.postpartum_dur = 0

        # Count the state of the agent for postpartum -- # TOOD: refactor, what is this loop doing?
        pp = self.filter(self.postpartum)
        for key,(pp_low, pp_high) in fpd.postpartum_mapping.items():
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

        # Reset states
        miscarriage.pregnant   = False
        miscarriage.postpartum = False
        miscarriage.gestation  = 0  # Reset gestation counter
        return


    def reset_breastfeeding(self):
        '''Stop breastfeeding, calculate total lifetime duration so far, and reset lactation episode to zero'''
        self.lactating = False
        self.breastfeed_dur_total += self.breastfeed_dur
        self.breastfeed_dur = 0
        return


    def maternal_mortality(self):
        '''Check for probability of maternal mortality'''
        prob = self.pars['mortality_probs']['maternal'] * self.pars['maternal_mortality_multiplier']
        is_death = self.binomial(prob)
        death = self.filter(is_death)
        death.alive = False
        self.step_results['maternal_deaths'] += len(death)
        self.step_results['deaths'] += len(death)
        return death


    def infant_mortality(self):
        '''Check for probability of infant mortality (death < 1 year of age)'''
        is_death = self.binomial(self.pars['mortality_probs']['infant'])
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

        for key, (age_low, age_high) in fpd.age_bin_mapping.items():
            this_age_bin = live.filter((live.age >= age_low) * (live.age < age_high))
            self.step_results['birth_bins'][key] += len(this_age_bin)

        # Check mortality
        live.maternal_mortality() # Mothers of only live babies eligible to match definition of maternal mortality ratio
        i_death = live.infant_mortality()

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
            # print(mother, children)
            all_ppl.children[mother] += children
            start_ind = end_ind

        return


    def age_person(self):
        '''Advance age in the simulation'''
        self.age += self.pars['timestep'] / fpd.mpy  # Age the person for the next timestep
        self.age = np.minimum(self.age, self.pars['max_age'])
        return


    def update_age_bin_totals(self):
        '''
        Count how many total live women in each 5-year age bin 10-50, for tabulating ASFR
        '''
        for key, (age_low, age_high) in fpd.age_bin_mapping.items():
            this_age_bin = self.filter((self.age >= age_low) * (self.age < age_high))
            self.step_results['age_bin_totals'][key] += len(this_age_bin)
        return


    def update_contraception(self):
        '''If eligible (age 15-49 and not pregnant), choose new method or stay with current one'''

        postpartum = (self.postpartum) * (self.postpartum_dur <= 6)
        pp = self.filter(postpartum)
        non_pp = self.filter(~postpartum)

        pp.get_method_postpartum()

        whole_years = ((non_pp.round_age - non_pp.age) < (1/fpd.mpy)) * ((non_pp.round_age - non_pp.age) > 0)
        birthdays = non_pp.filter(whole_years)
        birthdays.get_method()
        #self.step_results['birthday_fraction'] = len(birthdays)/len(non_pp) # Debugs and tracks fraction of birthday months, remove comment if debugging

        return


    def check_mcpr(self):
        '''
        Track for purposes of calculating mCPR at the end of the timestep after all people are updated
        Not including LAM users in mCPR as this model counts all women passively using LAM but
        DHS data records only women who self-report LAM which is much lower.
        If wanting to include LAM here need to add "or self.lam == False" to 2nd if statemnt
        '''
        denominator = (self.pars['method_age'] <= self.age) * (self.age < self.pars['age_limit_fecundity']) * (self.pregnant == 0) * (self.sex == 0) * (self.alive)
        no_method = np.sum((self.method == 0) * denominator)
        on_method = np.sum((self.method != 0) * denominator)
        self.step_results['no_methods'] += no_method
        self.step_results['on_methods'] += on_method
        return


    def init_step_results(self):
        self.step_results = dict(
            deaths          = 0,
            births          = 0,
            stillbirths     = 0,
            total_births    = 0,
            maternal_deaths = 0,
            infant_deaths   = 0,
            on_methods      = 0,
            no_methods      = 0,
            pp0to5          = 0,
            pp6to11         = 0,
            pp12to23        = 0,
            total_women_fecund = 0,
            unintended_pregs = 0,
            birthday_fraction = None,
            birth_bins        = {},
            age_bin_totals    = {}
        )

        for key in fpd.age_bin_mapping.keys():
            self.step_results['birth_bins'][key] = 0
            self.step_results['age_bin_totals'][key] = 0

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
        methods.update_contraception()
        nonpreg.check_lam()
        nonpreg.update_postpartum() # Updates postpartum counter if postpartum
        lact.update_breastfeeding()
        nonpreg.check_conception()  # Decide if conceives and initialize gestation counter at 0

        # Update results
        fecund.update_age_bin_totals()
        #fecund.check_mcpr() TODO - build method to check mcpr at end of step, will be simpler than below
        #fecund.update_total_fecund_women()  TODO- build method to track all live women 15-49 for TFR, below not working

        # Update results
        self.check_mcpr()
        self.step_results['total_women_fecund'] = np.sum((self.sex == 0) * (15 <= self.age) * (self.age < self.pars['age_limit_fecundity']))

        # Age person at end of timestep after tabulating results
        alive_now.age_person()  # Important to keep this here so birth spacing gets recorded accurately

        return self.step_results



class Sim(fpb.BaseSim):
    '''
    The Sim class handles the running of the simulation
    '''

    def __init__(self, pars=None, label=None):
        super().__init__(pars) # Initialize and set the parameters as attributes

        # Test settings
        self.test_mode = False
        self.to_feather = False
        self.custom_feather_tables = None
        self.initialized = False

        # contains additional results of each timestep
        self.total_results = defaultdict(lambda: {})
        self.label = label
        fpu.set_metadata(self) # Set version, date, and git info
        return


    def initialize(self, force=False):
        if force or not self.initialized:
            fpu.set_seed(self.pars['seed'])
            self.init_results()
            self.init_people()
            self.interventions = {}  # dictionary for possible interventions to add to the sim
        return


    def init_results(self):
        resultscols = ['t', 'pop_size_months', 'births', 'deaths', 'stillbirths', 'total_births', 'maternal_deaths', 'infant_deaths', 'on_method',
                       'no_method', 'mcpr', 'pp0to5', 'pp6to11', 'pp12to23', 'nonpostpartum', 'total_women_fecund', 'unintended_pregs', 'birthday_fraction',
                       'total_births_10-14', 'total_births_15-19', 'total_births_20-24', 'total_births_25-29', 'total_births_30-34', 'total_births_35-39', 'total_births_40-44',
                       'total_births_45-49', 'total_women_10-14', 'total_women_15-19', 'total_women_20-24', 'total_women_25-29', 'total_women_30-34', 'total_women_35-39',
                       'total_women_40-44', 'total_women_45-49']
        self.results = {}
        for key in resultscols:
            self.results[key] = np.zeros(int(self.npts))
        self.results['tfr_years'] = []
        self.results['tfr_rates'] = []
        self.results['pop_size'] = []
        self.results['mcpr_by_year'] = []
        self.results['method_failures_over_year'] = []
        self.results['birthday_fraction'] = []
        self.results['asfr'] = {}

        for key in fpd.age_bin_mapping.keys():
            self.results['asfr'][key] = []

        return


    def get_age_sex(self, n):
        ''' For an ex nihilo person, figure out if they are male and female, and how old '''
        pyramid = self.pars['age_pyramid']
        self.m_frac = pyramid[:,1].sum() / pyramid[:,1:3].sum()

        ages = np.zeros(n)
        sexes = np.random.random(n) < self.m_frac  # Pick the sex based on the fraction of men vs. women
        f_inds = sc.findinds(sexes == 0)
        m_inds = sc.findinds(sexes == 1)

        age_data_min   = pyramid[:,0]
        age_data_max   = np.append(pyramid[1:,0], self.pars['max_age'])
        age_data_range = age_data_max - age_data_min
        for i,inds in enumerate([m_inds, f_inds]):
            if len(inds):
                age_data_prob  = pyramid[:,i+1]
                age_data_prob  = age_data_prob/age_data_prob.sum() # Ensure it sums to 1
                age_bins       = fpu.n_multinomial(age_data_prob, len(inds)) # Choose age bins
                ages[inds]     = age_data_min[age_bins] + age_data_range[age_bins]*np.random.random(len(inds)) # Uniformly distribute within this age bin


        return ages, sexes


    def make_people(self, n=1, age=None, sex=None, method=None):
        ''' Set up each person '''
        _age, _sex = self.get_age_sex(n)
        if age    is None: age    = _age
        if sex    is None: sex    = _sex
        if method is None: method = np.zeros(n, dtype=np.int64)
        barrier = fpu.n_multinomial(self.pars['barriers'][:], n)
        data = dict(age=age, sex=sex, method=method, barrier=barrier)
        return data


    def init_people(self, output=False, **kwargs):
        ''' Create the people '''
        p = sc.objdict(self.make_people(n=int(self.pars['n'])))
        self.people = People(pars=self.pars, age=p.age, sex=p.sex, method=p.method, barrier=p.barrier)
        return


    def add_intervention(self, intervention, year):
        '''Allow adding an intervention at the time point corresponding to the year passed in'''
        index = self.year2ind(year)
        self.interventions[index] = intervention
        return


    def update_methods_matrices(self):
        '''Update all contraceptive matrices to have probabilities that follow a trend closest to the
        year the sim is on based on mCPR in that year'''

        switch_general = {}
        start_postpartum = {}
        switch_postpartum = {}

        ind = sc.findnearest(self.pars['methods']['mcpr_years'], self.y)  # Find the closest year to the timestep we are on

        # Update general population switching matrices for current year mCPR - stratified by age
        for key, val in self.pars['methods']['probs_matrix'].items():
            switch_general[key] = sc.dcp(val)
            switch_general[key][0, 0] *= self.pars['methods']['trend'][ind]  # Takes into account mCPR during year of sim
            for i in range(len(switch_general[key])):
                denom = switch_general[key][i,:].sum()
                if denom > 0:
                    switch_general[key][i] = switch_general[key][i, :] / denom  # Normalize so probabilities add to 1
            self.pars['methods'][key] = switch_general[key]

        # Update postpartum initiation matrices for current year mCPR - stratified by age
        for key, val in self.pars['methods_postpartum']['probs_matrix_1'].items():
            start_postpartum[key] = sc.dcp(val)
            start_postpartum[key][0] *= self.pars['methods_postpartum']['trend'][ind]  # Takes into account mCPR during year of sim
            start_postpartum[key] = start_postpartum[key] / start_postpartum[key].sum()
            self.pars['methods_postpartum'][key] = start_postpartum[key]  # 1d array for probs coming from birth, binned by age

        # Update postpartum switching or discontinuation matrices from 1-6 months - stratified by age
        for key, val in self.pars['methods_postpartum']['probs_matrix_1-6'].items():
            switch_postpartum[key] = sc.dcp(val)
            switch_postpartum[key][0, 0] *= self.pars['methods_postpartum']['trend'][ind]  # Takes into account mCPR during year of sim
            for i in range(len(switch_postpartum[key])):
                denom = switch_postpartum[key][i,:].sum()
                if denom > 0:
                    switch_postpartum[key][i] = switch_postpartum[key][i,:] / denom  # Normalize so probabilities add to 1
            self.pars['methods_postpartum_switch'][key] = switch_postpartum[key]  # 10x10 matrix for probs of continuing or discontinuing method by 6 months postpartum

        return


    def update_mortality_probs(self):
        ''' Update infant and maternal mortality for the sim's current year.  Update general mortality trend
        as this uses a spline interpolation instead of an array'''

        ind = sc.findnearest(self.pars['age_mortality']['years'], self.y)
        gen_mortality_trend = self.pars['age_mortality']['trend'][ind]

        ind = sc.findnearest(self.pars['infant_mortality']['year'], self.y)
        infant_mort_prob = self.pars['infant_mortality']['probs'][ind]

        ind = sc.findnearest(self.pars['maternal_mortality']['year'], self.y)
        maternal_death_prob = self.pars['maternal_mortality']['probs'][ind]

        ind = sc.findnearest(self.pars['stillbirth_rate']['year'], self.y)
        stillbirth_prob = self.pars['stillbirth_rate']['probs'][ind]

        self.pars['mortality_probs'] = {
            'gen_trend': gen_mortality_trend,
            'infant': infant_mort_prob,
            'maternal': maternal_death_prob,
            'stillbirth': stillbirth_prob
        }

        return


    def apply_interventions(self):
        ''' Apply each intervention in the model '''
        if 'interventions' in self.pars:
            for i,intervention in enumerate(sc.tolist(self.pars['interventions'])):
                if isinstance(intervention, fpi.Intervention):
                    if not intervention.initialized: # pragma: no cover
                        intervention.initialize(self)
                    intervention.apply(self) # If it's an intervention, call the apply() method
                elif callable(intervention):
                    intervention(self) # If it's a function, call it directly
                else: # pragma: no cover
                    errormsg = f'Intervention {i} ({intervention}) is neither callable nor an Intervention object'
                    raise TypeError(errormsg)
        return


    def apply_analyzers(self):
        ''' Apply each analyzer in the model '''
        if 'analyzers' in self.pars:
            for i,analyzer in enumerate(sc.tolist(self.pars['analyzers'])):
                if isinstance(analyzer, fpi.Analyzer):
                    if not analyzer.initialized: # pragma: no cover
                        analyzer.initialize(self)
                    analyzer.apply(self) # If it's an intervention, call the apply() method
                elif callable(analyzer):
                    analyzer(self) # If it's a function, call it directly
                else: # pragma: no cover
                    errormsg = f'Analyzer {i} ({analyzer}) is neither callable nor an Analyzer object'
                    raise TypeError(errormsg)
        return


    def run(self, verbose=None):
        ''' Run the simulation '''

        T = sc.tic()

        # Reset settings and results
        if verbose is None:
            verbose = self.pars['verbose']
        self.initialize()

        # Main simulation loop

        for i in range(self.npts):  # Range over number of timesteps in simulation (ie, 0 to 261 steps)
            self.i = i # Timestep
            self.t = self.ind2year(i)  # t is time elapsed in years given how many timesteps have passed (ie, 25.75 years)
            self.y = self.ind2calendar(i)  # y is calendar year of timestep (ie, 1975.75)
            if verbose:
                if not (self.t % int(1.0/verbose)):
                    string = f'  Running {self.y:0.1f} of {self.pars["end_year"]}...'
                    sc.progressbar(i+1, self.npts, label=string, length=20, newline=True)

            # Apply interventions
            self.apply_interventions()

            # Update method matrices for year of sim to trend over years
            self.update_methods_matrices()

            # Update mortality probabilities for year of sim
            self.update_mortality_probs()

            # Call the interventions
            if i in self.interventions:
                self.interventions[i](self)

            # Update the people
            self.people.t = self.t
            step_results = self.people.update()
            r = fpu.dict2obj(step_results)

            # Start calculating results
            new_people = r.births - r.infant_deaths # Do not add agents who died before age 1 to population

            # Births
            data = self.make_people(n=new_people, age=np.zeros(new_people))

            people = People(pars=self.pars, n=new_people, **data)
            self.people += people

            # Results
            percent0to5   = (r.pp0to5 / r.total_women_fecund) * 100
            percent6to11  = (r.pp6to11 / r.total_women_fecund) * 100
            percent12to23 = (r.pp12to23 / r.total_women_fecund) * 100
            nonpostpartum = ((r.total_women_fecund - r.pp0to5 - r.pp6to11 - r.pp12to23)/r.total_women_fecund) * 100

            # Store results
            self.results['t'][i]               = self.tvec[i]
            self.results['pop_size_months'][i] = self.n
            self.results['births'][i]          = r.births
            self.results['deaths'][i]          = r.deaths
            self.results['stillbirths'][i]     = r.stillbirths
            self.results['total_births'][i]    = r.total_births
            self.results['maternal_deaths'][i] = r.maternal_deaths
            self.results['infant_deaths'][i]   = r.infant_deaths
            self.results['on_method'][i]       = r.on_methods
            self.results['no_method'][i]       = r.no_methods
            self.results['mcpr'][i]            = r.on_methods/(r.on_methods+r.no_methods)
            self.results['pp0to5'][i]          = percent0to5
            self.results['pp6to11'][i]         = percent6to11
            self.results['pp12to23'][i]           = percent12to23
            self.results['nonpostpartum'][i]      = nonpostpartum
            self.results['total_women_fecund'][i] = r.total_women_fecund
            self.results['unintended_pregs'][i]   = r.unintended_pregs

            # Store results of total births per age bin for ASFR
            self.results['total_births_10-14'][i]    = r.birth_bins['10-14']
            self.results['total_births_15-19'][i]    = r.birth_bins['15-19']
            self.results['total_births_20-24'][i]    = r.birth_bins['20-24']
            self.results['total_births_25-29'][i]    = r.birth_bins['25-29']
            self.results['total_births_30-34'][i]    = r.birth_bins['30-34']
            self.results['total_births_35-39'][i]    = r.birth_bins['35-39']
            self.results['total_births_40-44'][i]    = r.birth_bins['40-44']
            self.results['total_births_45-49'][i]    = r.birth_bins['45-49']

            # Store results of total fecund women per age bin for ASFR
            self.results['total_women_10-14'][i] = r.age_bin_totals['10-14']
            self.results['total_women_15-19'][i] = r.age_bin_totals['15-19']
            self.results['total_women_20-24'][i] = r.age_bin_totals['20-24']
            self.results['total_women_25-29'][i] = r.age_bin_totals['25-29']
            self.results['total_women_30-34'][i] = r.age_bin_totals['30-34']
            self.results['total_women_35-39'][i] = r.age_bin_totals['35-39']
            self.results['total_women_40-44'][i] = r.age_bin_totals['40-44']
            self.results['total_women_45-49'][i] = r.age_bin_totals['45-49']

            # Calculate metrics (TFR, mCPR, and unintended pregnancies) over the last year in the model and save whole years and stats to an array
            if i % fpd.mpy == 0:
                self.results['tfr_years'].append(self.y)
                start_index = (int(self.t)-1)*fpd.mpy
                stop_index = int(self.t)*fpd.mpy
                unintended_pregs_over_year = pl.sum(self.results['unintended_pregs'][start_index:stop_index]) # Grabs sum of unintended pregnancies due to method failures over the last 12 months of calendar year
                self.results['pop_size'].append(self.n)
                self.results['mcpr_by_year'].append(self.results['mcpr'][i])
                self.results['method_failures_over_year'].append(unintended_pregs_over_year)
                #self.results['birthday_fraction'].append(r.birthday_fraction)  # This helps track that birthday months are being tracked correctly, remove comment if needing to debug

                tfr = 0
                for key in fpd.age_bin_mapping.keys():
                        age_bin_births_year = pl.sum(self.results['total_births_'+key][start_index:stop_index])
                        age_bin_total_women_year = self.results['total_women_'+key][stop_index]
                        self.results['asfr'][key].append((age_bin_births_year / age_bin_total_women_year)*1000)
                        tfr += ((self.results['asfr'][key][-1])/1000)

                self.results['tfr_rates'].append(tfr*5)

            if self.test_mode:
                for state in fpd.debug_states:
                    self.total_results[self.y][state] = getattr(self.people, state)

        if self.test_mode:
            if not self.to_feather:
                sc.savejson(filename="sim_output/total_results.json", obj=self.total_results)
            else:
                if self.custom_feather_tables is None:
                    states = fpd.debug_states
                else:
                    states = self.custom_feather_tables
                for state in states:
                    state_frame = pd.DataFrame()
                    max_length = len(self.total_results[max(self.total_results.keys())][state])
                    for timestep, _ in self.total_results.items():
                        colname = str(timestep) + "_" + state
                        adjustment = max_length - len(self.total_results[timestep][state])
                        state_frame[colname] = list(self.total_results[timestep][state]) + [None] * adjustment # ONLY WORKS IF LAST YEAR HAS MOST PEOPLE

                    feather.write_feather(state_frame, f"sim_output/{state}_state")




        # Apply analyzers
        self.apply_analyzers()

        # Convert all results to Numpy arrays
        for key,arr in self.results.items():
            if isinstance(arr, list):
                self.results[key] = np.array(arr) # Convert any lists to arrays

        # self.results['tfr_rates']    = np.array(self.results['tfr_rates']) # Store TFR rates for each year of model
        # self.results['tfr_years']    = np.array(self.results['tfr_years']) # Save an array of whole years that model runs (ie, 1950, 1951...)
        # self.results['pop_size']     = np.array(self.results['pop_size'])  # Store population size array in years and not months for calibration
        # self.results['mcpr_by_year'] = np.array(self.results['mcpr_by_year'])


        print(f'Final population size: {self.n}.')

        elapsed = sc.toc(T, output=True)
        print(f'Run finished for "{self.pars["name"]}" after {elapsed:0.1f} s')

        return self.results


    def store_postpartum(self):

        '''Stores snapshot of who is currently pregnant, their parity, and various
        postpartum states in final step of model for use in calibration'''

        min_age = 12.5
        max_age = self.pars['age_limit_fecundity']

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


    def plot(self, dosave=None, doshow=True, figargs=None, plotargs=None, axisargs=None, as_years=True, new_fig=True):
        '''
        Plot the results -- can supply arguments for both the figure and the plots.

        Args:
            dosave (bool): Whether or not to save the figure. If a string, save to that filename.
            doshow (bool): Whether to show the plots at the end
            figargs (dict):  Dictionary of kwargs to be passed to pl.figure()
            plotargs (dict): Dictionary of kwargs to be passed to pl.plot()
            as_years (bool): Whether to plot the x-axis as years or time points
        '''

        if figargs  is None: figargs  = {'figsize':(16,8)}
        if plotargs is None: plotargs = {'lw':2, 'alpha':0.7, 'marker':'o'}
        if axisargs is None: axisargs = {'left':0.1, 'bottom':0.05, 'right':0.9, 'top':0.97, 'wspace':0.2, 'hspace':0.25}

        fig = pl.figure(**figargs) if new_fig else pl.gcf()
        pl.subplots_adjust(**axisargs)

        def getbest(res):
            ''' If it's best/high/low, return best; else return unchanged '''
            return res.best if hasattr(res, 'best') else res

        res = self.results # Shorten since heavily used

        x = getbest(res['t']) # Likewise
        if not as_years:
            x *= fpd.mpy
            x -= x[0]
            timelabel = 'Timestep'
        else:
            timelabel = 'Year'

        # Plot everything
        to_plot = sc.odict({
            'Population size': sc.odict({'pop_size_months':'Population size'}),
            'MCPR': sc.odict({'mcpr':'Modern contraceptive prevalence rate (%)'}),
            'Births': sc.odict({'births':'Births'}),
            'Deaths': sc.odict({'deaths':'Deaths'}),
            'Maternal mortality': sc.odict({'maternal_deaths':'Cumulative birth-related maternal deaths'}),
            'Infant mortality': sc.odict({'infant_deaths':'Cumulative infant deaths'}),
            })
        for p,title,keylabels in to_plot.enumitems():
            pl.subplot(2,3,p+1)
            for i,key,label in keylabels.enumitems():
                this_res = getbest(res[key])

                if label.startswith('Cumulative'):
                    y = pl.cumsum(this_res)
                elif key == 'mcpr':
                    y = this_res*100
                else:
                    y = this_res
                if not new_fig: # Replace with sim label to avoid duplicate labels
                    label = self.label
                pl.plot(x, y, label=label, **plotargs)
            fpu.fixaxis(useSI=fpd.useSI, set_lim=new_fig) # If it's not a new fig, don't set the lim
            if key == 'mcpr':
                pl.ylabel('Percentage')
            else:
                pl.ylabel('Count')
            pl.xlabel(timelabel)
            pl.title(title, fontweight='bold')

        # Ensure the figure actually renders or saves
        if dosave:
            if isinstance(dosave, str):
                filename = dosave # It's a string, assume it's a filename
            else:
                filename = 'fp_sim.png' # Just give it a default name
            pl.savefig(filename)
        if doshow:
            pl.show() # Only show if we're not saving

        return fig


    def plot_people(self):
        ''' Use imshow() to show all individuals as rows, with time as columns, one pixel per timestep per person '''
        # The test_mode might help with this
        raise NotImplementedError


class MultiSim(sc.prettyobj):
    '''
    The MultiSim class handles the running of multiple simulations
    '''

    def __init__(self, sims=None, label=None, n=None, **kwargs):

        # A single sim is supplied -- convert to a list
        if isinstance(sims, Sim):
            if n is None:
                errormsg = 'If providing a single sim, you must supply the n argument to specify how many duplicates of that sim you want to run'
                raise ValueError(errormsg)
            base_sim = sims
            sims = []
            for i in range(n):
                sim = sc.dcp(base_sim)
                sim['seed'] += i # Increment the seed
                if sim.label is None:
                    sim.label = f'Sim {i}'
                sims.append(sim)

        # Basic checks
        assert isinstance(sims, list), "Must supply sims as a list"
        assert len(sims)>0, "Must supply at least 1 sim"

        # Set properties
        self.sims      = sims
        self.base_sim  = sc.dcp(sims[0])
        self.label     = self.base_sim.label if label is None else label
        self.run_args  = sc.mergedicts(kwargs)
        self.results   = None
        self.which     = None # Whether the multisim is to be reduced, combined, etc.
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

        for reskey in base_sim.results.keys():
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


    def plot(self, doshow=True, plot_sims=True, fig_args=None, **kwargs):
        '''
        Plot the MultiSim
        '''
        fig_args = sc.mergedicts(fig_args)
        if plot_sims:
            fig = pl.figure(**fig_args)
            doshow = kwargs.pop('doshow', True)
            for sim in self.sims: # Note: produces duplicate legend entries
                sim.plot(new_fig=False, doshow=False, **kwargs)
            if doshow:
                pl.show()
            return fig
        else:
            return self.base_sim.plot(doshow=doshow, fig_args=fig_args, **kwargs)






def single_run(sim):
    ''' Helper function for multi_run(); rarely used on its own '''
    sim.run()
    return sim


def multi_run(sims, **kwargs):
    ''' Run multiple sims in parallel; usually used via the MultiSim class, not directly '''
    sims = sc.parallelize(single_run, iterarg=sims, **kwargs)
    return sims