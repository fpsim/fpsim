'''
Defines the Sim class, the core class of the FP model (FPsim).
'''

#%% Imports
import numpy as np # Needed for a few things not provided by pl
import pylab as pl
import sciris as sc
from scipy import interpolate as si
import pandas as pd
from . import defaults as fpd
from . import population as fpp
from . import utils as fpu
from . import base as fpb


# Specify all externally visible things this file defines
__all__ = ['People', 'Sim', 'single_run', 'multi_run']


#%% Define classes
def arr(n=None, val=0):
    ''' Shortcut for defining an empty array with the correct value and data type '''
    if sc.isarray(val):
        assert len(val) == n
        arr = val
    elif isinstance(val, list):
        arr = [sc.dcp(val)]*n
    else:
        dtype = object if sc.isstring(val) else None
        arr = np.full(shape=n, fill_value=val, dtype=dtype)
    return arr


class People(fpb.ParsObj):
    '''
    Class for all the people in the simulation.
    '''
    def __init__(self, pars, **kwargs):

        # Initialization
        self.update_pars(pars) # Set parameters
        d = sc.objdict(sc.mergedicts(fpd.person_defaults, kwargs)) # d = defaults
        n = self.pars['n']

        # Basic states
        self.uid      = np.arange(n)
        self.age      = arr(n, float(d.age)) # Age of the person (in years)
        self.sex      = arr(n, d.sex) # Female (0) or male (1)
        self.parity   = arr(n, d.parity) # Number of children
        self.method   = arr(n, d.method)  # Contraceptive method 0-9, see pars['methods']['map'], excludes LAM as method
        self.barrier  = arr(n, d.barrier)  # Reason for non-use
        self.alive    = arr(n, True)
        self.pregnant = arr(n, False)

        # Sexual and reproductive history
        self.sexually_active = arr(n, False)
        self.lactating       = arr(n, False)
        self.gestation       = arr(n, d.gestation)
        self.postpartum      = arr(n, False)
        self.postpartum_dur  = arr(n, d.postpartum_dur) # Tracks # months postpartum
        self.lam             = arr(n, False) # Separately tracks lactational amenorrhea, can be using both LAM and another method
        self.dobs            = arr(n, [-1]*self.parity) # Dates of births # TODO: refactor
        self.lactating       = arr(n, False)
        self.breastfeed_dur  = arr(n, d.breastfeed_dur)
        self.breastfeed_dur_total = arr(n, d.breastfeed_dur_total)

        # Fecundity variation
        fv = self.pars['fecundity_variation']
        self.personal_fecundity = arr(n, np.random.random(n)*(fv[1]-fv[0])+fv[0]) # Stretch fecundity by a factor bounded by [f_var[0], f_var[1]]
        self.remainder_months = arr(n, d.remainder_months)
        return


    def __len__(self):
        try:
            return self.pars['n']
        except Exception as E1:
            try:
                return len(self.uid)
            except Exception as E2:
                print(f'Warning: could not get length of People (could not get pars.n: {E1}, could not get self.uid: {E2})')
                return 0


    @property
    def is_female(self):
        return self.sex == 0

    @property
    def is_male(self):
        return self.sex == 1

    @property
    def int_ages(self):
        return np.array(self.ages, dtype=np.int64)

    def female_inds(self):
        return sc.findinds(self.is_female)

    def male_inds(self):
        return sc.findinds(self.is_male)


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
                inds = sc.findinds(match)

                matrix = self.pars['methods'][key]
                choices = matrix[m]
                new_methods = fpu.n_multinomial(choices, len(inds))
                self.method[inds] = new_methods

        return


    def get_method_postpartum(self, t, y):
        '''Utilizes data from birth to allow agent to initiate a method postpartum coming from birth by
         3 months postpartum and then initiate, continue, or discontinue a method by 6 months postpartum.
        Next opportunity to switch methods will be on whole calendar years, whenever that falls.
        '''
        # TODO- Probabilities need to be adjusted for postpartum women on the next annual draw in "get_method" since they may be less than one year

        # Probability of initiating a postpartum method at 0-3 months postpartum
        # Transitional probabilities are for the first 3 month time period after delivery from DHS data

        pp_methods = self.pars['methods_postpartum']
        pp_switch  = pp_methods['switch_postpartum']
        orig_methods = self.method

        postpartum3 = (self.postpartum_dur == 3)
        postpartum6 = (self.postpartum_dur == 6)

        # At 3 months, choice is by age but not previous method (since just gave birth)
        for key,(age_low, age_high) in fpd.method_age_mapping.items():
            match_low  = (self.age >= age_low)
            match_high = (self.age <  age_high)
            match = postpartum3 * match_low * match_high
            inds = sc.findinds(match)

            choices = pp_methods[key]
            new_methods = fpu.n_multinomial(choices, len(inds))
            self.method[inds] = new_methods

        # At 6 months, choice is by previous method but not age
        # Allow initiation, switching, or discontinuing with matrix at 6 months postpartum
        # Transitional probabilities are for 3 months, 4-6 months after delivery from DHS data
        for m in self.pars['methods']['map'].values():
            match_m    = (orig_methods == m)
            match = postpartum6 * match_m
            inds = sc.findinds(match)

            choices = pp_switch[m]
            new_methods = fpu.n_multinomial(choices, len(inds))
            self.method[inds] = new_methods

        return


    def check_mortality(self):
        '''Decide if person dies at a timestep'''

        timestep = self.pars['timestep']
        trend_val = self.pars['mortality_probs']['gen_trend']
        age_mort = self.pars['age_mortality']
        f_spline = age_mort['f_spline'] * trend_val
        m_spline = age_mort['m_spline'] * trend_val
        f_inds   = self.female_inds()
        m_inds   = self.male_inds()
        int_ages = self.int_ages
        f_ages = int_ages[f_inds]
        m_ages = int_ages[m_inds]

        f_mort_prob = fpu.annprob2ts(f_spline[f_ages], timestep)
        m_mort_prob = fpu.annprob2ts(m_spline[m_ages], timestep)

        f_died = f_inds[fpu.binomial_arr(f_mort_prob)]
        m_died = m_inds[fpu.binomial_arr(m_mort_prob)]
        died = sc.cat(f_died, m_died)
        self.alive[died] = False
        self.step_results['died'] = len(died)

        return


    def check_sexually_active(self):
        '''
        Decide if agent is sexually active based either on month postpartum or age if
        not postpartum.  Postpartum and general age-based data from DHS.
        '''
        n = len(self)
        probs = np.zeros(n)

        # Set postpartum probabilities
        match_low  = self.postpartum_dur > 0
        match_high = self.postpartum_dur <= 6
        pp_match = self.postpartum * match_low * match_high
        pp_inds = sc.findinds(pp_match)
        probs[pp_inds] = self.pars['sexual_activity_postpartum']['percent_active'][self.postpartum_dur[pp_inds]]

        # Set non-postpartum probabilities
        nonpp_inds = np.setdiff1d(np.arange(n), pp_inds)
        probs[nonpp_inds] = self.pars['sexual_activity'][self.int_ages[nonpp_inds]]

        # Evaluate likelihood in this time step of being sexually active
        # Can revert to active or not active each timestep
        self.sexually_active = fpu.binomial_arr(probs)

        return


    def check_conception(self):
        '''
        Decide if person (female) becomes pregnant at a timestep.
        '''

        # Find monthly probability of pregnancy based on fecundity and any use of contraception including LAM - from data

        preg_prob = utils.numba_preg_prob(self.pars['age_fecundity'], self.personal_fecundity, self.age, resolution,
                                          self.pars['method_efficacy'][self.method], self.lam,
                                          self.pars['LAM_efficacy'], mpy)

        # Adjust for decreased likelihood of conception if nulliparous vs already gravid - from data
        if self.parity == 0:   #Nulliparous
            ind = sc.findnearest(self.pars['fecundity_ratio_nullip'][0], self.age)
            fr_nullip = self.pars['fecundity_ratio_nullip'][1][ind]
            preg_prob *= fr_nullip

        # Adjust for probability of exposure to pregnancy episode at this timestep based on age and parity - encapsulates background factors - experimental and tunable
        ind = sc.findnearest(self.pars['exposure_correction_age'][0], self.age)
        exposure_range = self.pars['exposure_correction_age'][1][ind]
        exposure_correction_age = exposure_range # CK: was getting a warning about ragged arrays before; this is disabled for now np.random.uniform(exposure_range[0], exposure_range[1]) # Affect exposure by a random number between interval for agent's age
        # Using a range for age and a static number for parity as this is a work in progress -- how to represent this exposoure correction still up for debate
        ind2 = sc.findnearest(self.pars['exposure_correction_parity'][0], self.parity)
        exposure_correction_parity = self.pars['exposure_correction_parity'][1][ind2]

        preg_prob *= exposure_correction_age
        preg_prob *= exposure_correction_parity

        # Use a single binomial trial to check for conception successes this month
        pregnant = utils.bt(preg_prob)

        if pregnant:
            abortion = utils.bt(self.pars['abortion_prob']) # Check to see if this pregnancy ends in abortion, do not set up pregnancy if yes
            if abortion:
                self.postpartum = False
                self.postpartum_dur = 0  # Reset postpartum state if got pregnant and then aborted
                return

            self.pregnant = True
            self.gestation = 0  # Start the counter at 0 to allow full 9 months gestation
            self.preg_dur = (np.random.randint(self.pars['preg_dur'][0], self.pars['preg_dur'][1] + 1))  # Duration of this pregnancy
            self.postpartum = False
            self.postpartum_dur = 0
            self.reset_breastfeeding() # Stop lactating if becoming pregnant

        return


    def check_lam(self):
        '''
        Check to see if postpartum agent meets criteria for LAM in this time step
        '''

        if self.postpartum:
            if 0 <= self.postpartum_dur <= 5:
                lam_rate = self.pars['lactational_amenorrhea']['rate'][self.postpartum_dur]
                self.lam = utils.bt(lam_rate)

            if self.postpartum_dur > 5:
                self.lam = False

        else:
            self.lam = False

        return


    def update_breastfeeding(self):
        '''Track breastfeeding, and update time of breastfeeding for individual pregnancy.
        Currently agents breastfeed a random amount of time between 1 and 24 months.
        Not connected to rates of exclusive breastfeeding in data informaed by self.lam
        NOT currently being utilized to track lactataional amenorrhea, left here in case useful feature in the future'''
        if self.breastfeed_dur >= (np.random.randint(self.pars['breastfeeding_dur'][0],
                                                             self.pars['breastfeeding_dur'][
                                                                 1] + 1)):  # when each breastfeeding episode terminates (in months) within range set in parameters
            self.reset_breastfeeding()

        else:
            self.breastfeed_dur += self.pars['timestep']

        return


    def update_postpartum(self):
        '''Track duration of extended postpartum period (0-24 months after birth).  Only enter this function if agent is postpartum'''

        # Stop postpartum episode if reach max length (set to 24 months)
        if self.postpartum_dur >= (self.pars['postpartum_length']):
            self.postpartum = False
            self.postpartum_dur = 0

        # Count the state of the agent
        if self.postpartum:
            if 0 <= self.postpartum_dur < 6:
                self.step_results['pp0to5'] = True
            elif 6 <= self.postpartum_dur < 12:
                self.step_results['pp6to11'] = True
            elif 12 <= self.postpartum_dur < 24:
                self.step_results['pp12to23'] = True

            # Advance the state to next timestep
            self.postpartum_dur += self.pars['timestep']

        return


    def update_pregnancy(self):
        '''Advance pregnancy in time and check for miscarriage'''

        if self.pregnant:
            self.gestation += self.pars['timestep']

            # Check for probability of miscarriage this pregnancy and end pregnancy if miscarried
            miscarriage_prob = utils.numba_miscarriage_prob(self.pars['miscarriage_rates'], self.age, resolution)

            if self.gestation == (self.pars['end_first_tri']):
                miscarriage = utils.bt(miscarriage_prob)
                if miscarriage:
                    self.pregnant = False
                    self.postpartum = False
                    self.gestation = 0  # Reset gestation counter
        return


    def reset_breastfeeding(self):
        '''Stop breastfeeding, calculate total lifetime duration so far, and reset lactation episode to zero'''
        self.lactating = False
        self.breastfeed_dur_total += self.breastfeed_dur
        self.breastfeed_dur = 0

        return


    def maternal_mortality(self):
        '''Check for probability of maternal mortality'''

        self.step_results['maternal_death'] = utils.bt(self.pars['mortality_probs']['maternal'])
        if self.step_results['maternal_death']:
            self.alive = False
            self.step_results['died'] = True

        return


    def infant_mortality(self):
        '''Check for probability of infant mortality (death < 1 year of age)'''

        self.step_results['infant_death'] = utils.bt(self.pars['mortality_probs']['infant'])
        if self.step_results['infant_death']:
            self.reset_breastfeeding()

        return


    def check_delivery(self):
        '''Decide if pregnant woman gives birth and explore maternal mortality and child mortality'''

        if self.gestation >= self.preg_dur:  # Needs to be separate since this won't exist otherwise
            self.pregnant = False
            self.gestation = 0  # Reset gestation counter
            twins = utils.bt(self.pars['twins_prob'])
            if twins:
                self.step_results['gave_birth'] = 2
                self.parity += 2
                self.dobs.append(self.age) # Used for birth spacing only, only add one baby to dob
            else:
                self.step_results['gave_birth'] = 1
                self.parity += 1
                self.dobs.append(self.age)
            self.lactating = True  # Start lactating at time of birth
            self.postpartum = True # Start postpartum state at time of birth
            self.breastfeed_dur = 0  # Start at 0, will update before leaving timestep in separate function
            self.postpartum_dur = 0

            self.maternal_mortality()
            self.infant_mortality()

        return


    def age_person(self):
        '''Advance age in the simulation'''
        self.age += self.pars['timestep'] / fpd.mpy  # Age the person for the next timestep
        self.age = min(self.age, self.pars['max_age'])
        return


    def update_contraception(self, t, y):
        '''If eligible (age 15-49 and not pregnant), choose new method or stay with current one'''

        if self.postpartum and self.postpartum_dur <= 6:
            self.get_method_postpartum(t, y)

        # If switching frequency in months has passed, allows switching only on whole years
        else:
            if t % (self.pars['switch_frequency']/mpy) == 0:
                self.get_method(y)

        return


    def check_mcpr(self):
        '''
        Track for purposes of calculating mCPR at the end of the timestep after all people are updated
        Not including LAM users in mCPR as this model counts all women passively using LAM but
        DHS data records only women who self-report LAM which is much lower.
        If wanting to include LAM here need to add "or self.lam == False" to 2nd if statemnt
        '''

        if self.pars['method_age'] <= self.age < self.pars['age_limit_fecundity'] and not self.pregnant:   #Tally for mCPR
            if self.method == 0:
                self.step_results['no_method'] = 1
            else:
                self.step_results['on_method'] = 1

        return


    def init_step_results(self):
        self.step_results = {
            'died' : False,
            'gave_birth' : 0,
            'maternal_death' : False,
            'infant_death' : False,
            'on_method' : False,
            'no_method' : False,
            'pp0to5'   : False,
            'pp6to11'   : False,
            'pp12to23'  : False,
            'sex': None,
            'age': None,
        }
        return


    def update(self, t, y):
        '''Update the person's state for the given timestep.
        t is the time in the simulation in years (ie, 0-60), y is years of simulation (ie, 1960-2010)'''

        self.init_step_results()   # Initialize outputs
        self.step_results['sex'] = self.sex
        self.step_results['age'] = self.age

        if self.alive:  # Do not move through step if not alive

            self.age_person()  # Age person in units of the timestep
            self.check_mortality()  # Decide if person dies at this t in the simulation
            if not self.alive:
                return self.step_results

            if self.sex == 0 and self.age < \
                    self.pars['age_limit_fecundity']:  # If female and age 0-49

                if self.pregnant:
                    self.check_delivery()  # Deliver with birth outcomes if reached pregnancy duration
                    self.update_pregnancy()  # Advance gestation in timestep, handle miscarriage
                    if not self.alive:
                        return self.step_results

                if not self.pregnant:
                    self.check_sexually_active()
                    if self.pars['method_age'] <= self.age < self.pars['age_limit_fecundity']:
                        self.update_contraception(t, y)
                    self.check_lam()
                    if self.sexually_active:
                        self.check_conception()  # Decide if conceives and initialize gestation counter at 0
                    if self.postpartum:
                        self.update_postpartum() # Updates postpartum counter if postpartum

                if self.lactating:
                    self.update_breastfeeding()

                self.check_mcpr()

        return self.step_results



class Sim(fpb.BaseSim):
    '''
    The Sim class handles the running of the simulation
    '''

    def __init__(self, pars=None):
        super().__init__(pars) # Initialize and set the parameters as attributes
        fpu.set_seed(self.pars['seed'])
        self.init_results()
        self.init_people()
        self.interventions = {}  # dictionary for possible interventions to add to the sim
        return


    def init_results(self):
        resultscols = ['t', 'pop_size_months', 'births', 'deaths', 'maternal_deaths', 'infant_deaths', 'on_method',
                       'no_method', 'mcpr', 'pp0to5', 'pp6to11', 'pp12to23', 'nonpostpartum', 'total_women_fecund']
        self.results = {}
        for key in resultscols:
            self.results[key] = np.zeros(int(self.npts))
        self.results['tfr_years'] = []
        self.results['tfr_rates'] = []
        self.results['pop_size'] = []
        self.results['mcpr_by_year'] = []
        return


    def get_age_sex(self):
        ''' For an ex nihilo person, figure out if they are male and female, and how old '''
        sex = np.random.random() < self.m_frac  # Pick the sex based on the fraction of men vs. women
        spline = self.f_pop_spline if sex == 0 else self.m_pop_spline  # Pick the male or female population spline
        age = si.splev(np.random.random(), spline)  # Use the spline fit to pick the age
        return age, sex


    def make_person(self, age=None, sex=None, method=None):
        ''' Set up each person'''
        _age, _sex = self.get_age_sex()
        if age is None: age = _age
        if sex is None: sex = _sex
        if method is None: method = 0
        barrier_ind = utils.mt(self.pars['barriers'][:])
        barrier = self.pars['barriers'].keys()[barrier_ind]

        person = Person(self.pars, age=age, sex=sex, method=method, barrier=barrier) # Create the person

        return person


    def init_people(self):
        ''' Create the people '''
        self.m_pop_spline, self.f_pop_spline, self.m_frac = population.make_age_sex_splines(self.pars)
        self.people = sc.odict()  # Dictionary for storing the people
        for i in range(int(self.pars['n'])):  # Loop over each person
            person = self.make_person()
            self.people[person.uid] = person  # Save them to the dictionary
        return


    def add_intervention(self, intervention, year):
        '''Allow adding an intervention at the time point corresponding to the year passed in'''
        index = self.year2ind(year)
        self.interventions[index] = intervention
        return


    def update_methods_matrices(self, y):
        '''Update all contraceptive matrices to have probabilities that follow a trend closest to the
        year the sim is on based on mCPR in that year'''

        switch_general = {}
        start_postpartum = {}

        ind = sc.findnearest(self.pars['methods']['mcpr_years'], y)  # Find the closest year to the timestep we are on

        # Update general population switching matrices for current year mCPR - stratified by age
        for key, val in self.pars['methods']['probs_matrix'].items():
            switch_general[key] = sc.dcp(val)
            switch_general[key][0, 0] *= self.pars['methods']['trend'][ind]  # Takes into account mCPR during year of sim
            for i in range(len(switch_general[key])):
                switch_general[key][i] = switch_general[key][i, :] / switch_general[key][i,
                                                           :].sum()  # Normalize so probabilities add to 1
            self.pars['methods'][key] = switch_general[key]

        # Update postpartum initiation matrices for current year mCPR - stratified by age
        for key, val in self.pars['methods_postpartum']['probs_matrix_0-3'].items():
            start_postpartum[key] = sc.dcp(val)
            start_postpartum[key][0] *= self.pars['methods_postpartum']['trend'][ind]
            start_postpartum[key] = start_postpartum[key] / start_postpartum[key].sum()
            self.pars['methods_postpartum'][key] = start_postpartum[key]  # 1d array for probs coming from birth, binned by age

        # Update postpartum switching or discontinuation matrices - not age stratified
        switch_postpartum = sc.dcp(self.pars['methods_postpartum']['probs_matrix_4-6'])
        switch_postpartum[0, 0] *= self.pars['methods_postpartum']['trend'][ind]
        for i in range(len(switch_postpartum)):
            switch_postpartum[i] = switch_postpartum[i,:] / switch_postpartum[i,:].sum()  # Normalize so probabilities add to 1
        self.pars['methods_postpartum']['switch_postpartum'] = switch_postpartum  # 10x10 matrix for probs of continuing or discontinuing method by 6 months postpartum

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

        self.pars['mortality_probs'] = {
            'gen_trend': gen_mortality_trend,
            'infant': infant_mort_prob,
            'maternal': maternal_death_prob
        }

        return


    def run(self, verbose=None):
        ''' Run the simulation '''

        T = sc.tic()

        # Reset settings and results
        if verbose is None:
            verbose = self.pars['verbose']
        self.update_pars()
        self.init_results()
        self.init_people() # Actually create the children

        # Main simulation loop

        for i in range(self.npts):  # Range over number of timesteps in simulation (ie, 0 to 261 steps)
            t = self.ind2year(i)  # t is time elapsed in years given how many timesteps have passed (ie, 25.75 years)
            y = self.ind2calendar(i)  # y is calendar year of timestep (ie, 1975.75)
            if verbose:
                if not (t % int(1.0/verbose)):
                    string = f'  Running {y:0.1f} of {self.pars["end_year"]}...'
                    sc.progressbar(i+1, self.npts, label=string, length=20, newline=True)

            # Update method matrices for year of sim to trend over years
            self.update_methods_matrices(y)

            # Update mortality probabilities for year of sim
            self.update_mortality_probs(y)

            # Update each person
            deaths = 0
            births = 0
            maternal_deaths = 0
            infant_deaths = 0
            on_methods = 0
            no_methods = 0
            pp0to5 = 0
            pp6to11 = 0
            pp12to23 = 0
            total_women_fecund = 0

            for person in self.people.values():
                step_results = person.update(t, y) # Update and count new cases
                deaths          += step_results['died']
                births          += step_results['gave_birth']
                maternal_deaths += step_results['maternal_death']
                infant_deaths   += step_results['infant_death']
                on_methods      += step_results['on_method']
                no_methods      += step_results['no_method']
                pp0to5          += step_results['pp0to5']
                pp6to11         += step_results['pp6to11']
                pp12to23        += step_results['pp12to23']

                if person.sex == 0 and 15 <= person.age < self.pars['age_limit_fecundity']:
                    total_women_fecund += 1

            if i in self.interventions:
                self.interventions[i](self)

            new_people = births - infant_deaths # Do not add agents who died before age 1 to population

            # Births
            for birth in range(new_people):
                person = self.make_person(age=0) # Save them to the dictionary
                self.people[person.uid] = person

            self.n = len(self.people) - deaths # Calculate new population size

            percent0to5 = (pp0to5 / total_women_fecund) * 100
            percent6to11 = (pp6to11 / total_women_fecund) * 100
            percent12to23 = (pp12to23 / total_women_fecund) * 100
            nonpostpartum = ((total_women_fecund - pp0to5 - pp6to11 - pp12to23)/total_women_fecund) * 100

            # Store results
            self.results['t'][i]                = self.tvec[i]
            self.results['pop_size_months'][i]   = self.n
            self.results['births'][i]          = births
            self.results['deaths'][i]          = deaths
            self.results['maternal_deaths'][i] = maternal_deaths
            self.results['infant_deaths'][i]    = infant_deaths
            self.results['on_method'][i]       = on_methods
            self.results['no_method'][i]       = no_methods
            self.results['mcpr'][i]            = on_methods/(on_methods+no_methods)
            self.results['pp0to5'][i]          = percent0to5
            self.results['pp6to11'][i]         = percent6to11
            self.results['pp12to23'][i]           = percent12to23
            self.results['nonpostpartum'][i]      = nonpostpartum
            self.results['total_women_fecund'][i] = total_women_fecund

            # Calculate TFR over the last year in the model and save whole years and tfr rates to an array
            if i % mpy == 0:
                self.results['tfr_years'].append(y)
                start_index = (int(t)-1)*mpy
                stop_index = int(t)*mpy
                births_over_year = pl.sum(self.results['births'][start_index:stop_index])  # Grabs sum of birth over the last 12 months of calendar year
                self.results['tfr_rates'].append(35*(births_over_year/self.results['total_women_fecund'][i]))
                self.results['pop_size'].append(self.n)
                self.results['mcpr_by_year'].append(self.results['mcpr'][i])

        self.results['tfr_rates'] = pl.array(self.results['tfr_rates']) # Store TFR rates for each year of model
        self.results['tfr_years'] = pl.array(self.results['tfr_years']) # Save an array of whole years that model runs (ie, 1950, 1951...)
        self.results['pop_size'] = pl.array(self.results['pop_size'])  # Store population size array in years and not months for calibration
        self.results['mcpr_by_year'] = pl.array(self.results['mcpr_by_year'])

        for person in self.people.values():
            if person.lactating:
                person.reset_breastfeeding()

        print(f'Final population size: {self.n}.')

        elapsed = sc.toc(T, output=True)
        print(f'Run finished for "{self.pars["name"]}" after {elapsed:0.1f} s')
        return self.results


    def store_postpartum(self):

        '''Stores snapshot of who is currently pregnant, their parity, and various
        postpartum states in final step of model for use in calibration'''

        min_age = 12.5
        max_age = self.pars['age_limit_fecundity']

        people = self.people.values()

        rows = []
        for person in people:
            if person.alive and person.sex == 0 and min_age <= person.age < max_age:
                row = {'Age': None, 'PP0to5': None, 'PP6to11': None, 'PP12to23': None, 'NonPP': None, 'Pregnant': None, 'Parity': None}
                row['Age'] = int(round(person.age))
                row['NonPP'] = 1 if not person.postpartum else 0
                if person.postpartum:
                    row['PP0to5'] = 1 if 0 <= person.postpartum_dur < 6 else 0
                    row['PP6to11'] = 1 if 6 <= person.postpartum_dur < 12 else 0
                    row['PP12to23'] = 1 if 12 <= person.postpartum_dur <= 24 else 0
                row['Pregnant'] = 1 if person.pregnant else 0
                row['Parity'] = person.parity
                rows.append(row)

        pp = pd.DataFrame(rows, index = None, columns = ['Age', 'PP0to5', 'PP6to11', 'PP12to23', 'NonPP', 'Pregnant', 'Parity'])
        pp.fillna(0, inplace=True)
        return pp


    def plot(self, dosave=None, figargs=None, plotargs=None, axisargs=None, as_years=True):
        '''
        Plot the results -- can supply arguments for both the figure and the plots.

        Parameters
        ----------
        dosave : bool or str
            Whether or not to save the figure. If a string, save to that filename.

        figargs : dict
            Dictionary of kwargs to be passed to pl.figure()

        plotargs : dict
            Dictionary of kwargs to be passed to pl.plot()

        as_years : bool
            Whether to plot the x-axis as years or time points

        Returns
        -------
        Figure handle

        '''

        if figargs  is None: figargs  = {'figsize':(20,8)}
        if plotargs is None: plotargs = {'lw':2, 'alpha':0.7, 'marker':'o'}
        if axisargs is None: axisargs = {'left':0.1, 'bottom':0.05, 'right':0.9, 'top':0.97, 'wspace':0.2, 'hspace':0.25}

        fig = pl.figure(**figargs)
        pl.subplots_adjust(**axisargs)

        res = self.results # Shorten since heavily used

        x = res['t'] # Likewise
        if not as_years:
            x *= mpy
            x -= x[0]
            timelabel = 'Timestep'
        else:
            timelabel = 'Year'

        # Plot everything
        to_plot = sc.odict({
            'Population size': sc.odict({'pop_size_months':'Population size'}),
            'MCPR': sc.odict({'mcpr':'Modern contraceptive prevalence rate (%)'}),
            'Births and deaths': sc.odict({'births':'Births', 'deaths':'Deaths'}),
            'Birth-related mortality': sc.odict({'maternal_deaths':'Cumulative birth-related maternal deaths', 'infant_deaths':'Cumulative infant deaths'}),
            })
        for p,title,keylabels in to_plot.enumitems():
            pl.subplot(2,2,p+1)
            for i,key,label in keylabels.enumitems():
                if label.startswith('Cumulative'):
                    y = pl.cumsum(res[key])
                elif key == 'mcpr':
                    y = res[key]*100
                else:
                    y = res[key]
                pl.plot(x, y, label=label, **plotargs)
            utils.fixaxis(useSI=useSI)
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
        else:
            pl.show() # Only show if we're not saving

        return fig


    def plot_people(self):
        ''' Use imshow() to show all individuals as rows, with time as columns, one pixel per timestep per person '''
        raise NotImplementedError


def single_run(sim):
    ''' Helper function for multi_run(); rarely used on its own '''
    sim.run()
    return sim


def multi_run(orig_sim, n=4, verbose=None):

    raise NotImplementedError('Memory leak, do not use!')

    # Copy the simulations
    sims = []
    for i in range(n):
        new_sim = sc.dcp(orig_sim)
        new_sim.pars['seed'] += i # Reset the seed, otherwise no point!
        new_sim.pars['n'] = int(new_sim.pars['n']/n) # Reduce the population size accordingly
        sims.append(new_sim)

    finished_sims = sc.parallelize(single_run, iterarg=sims)

    output_sim = sc.dcp(finished_sims[0])
    output_sim.pars['parallelized'] = n # Store how this was parallelized
    output_sim.pars['n'] *= n # Restore this since used in later calculations -- a bit hacky, it's true

    for sim in finished_sims[1:]: # Skip the first one
        output_sim.people.update(sim.people)
        for key,val in sim.results.items():
            if key != 't':
                output_sim.results[key] += sim.results[key]

    return output_sim


