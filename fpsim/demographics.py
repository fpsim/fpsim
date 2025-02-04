"""
Methods and functions related to basic demographics such as urban and
age at first partnership.
"""

# %% Imports
import numpy as np  # Needed for a few things not provided by pl
import sciris as sc
import starsim as ss
from . import defaults as fpd
from . import subnational as fpsn
from . import utils as fpu
from . import parameters as fpp

#%% Global defaults
useSI          = True
mpy            = 12   # Months per year, to avoid magic numbers
eps            = 1e-9 # To avoid divide-by-zero
min_age        = 15   # Minimum age to be considered eligible to use contraceptive methods
max_age        = 99   # Maximum age (inclusive)
max_age_preg   = 50   # Maximum age to become pregnant
max_parity     = 20   # Maximum number of children to track - also applies to abortions, miscarriages, stillbirths
max_parity_spline = 20   # Used for parity splines

class FamilyPlanning(ss.Demographics):
    def __init__(self, pars=None, location=None, empowerment_module=None, education_module=None, **kwargs):
        super().__init__(pars=pars, **kwargs)

        # Set default pars
        self.define_pars(
            location=None,
            track_children=False,
            regional=False,
            contraception_module=None,
            empowerment_module=empowerment_module,
            education_module=education_module,

            # Fecundity and exposure
            fecundity_var_low=0.7,
            fecundity_var_high= 1.1,
            primary_infertility= 0.05,
            exposure_factor= 1.0,  # Overall exposure correction factor

            # p_debut_age = ss.
            p_personal_fecundity=ss.uniform(),
        )

        self.pars['p_fertile']=ss.bernoulli(1 - self.pars['primary_infertility'])

        location_pars = fpp.pars(location)

        # TODO remove these from location_pars because they're part of core modules now
        location_pars.pop('verbose', None)
        location_pars.pop('timestep', None)

        # Update the parameters with the location-specific parameters
        self.define_pars(pars=location_pars)

        # Override any remaining parameters with user-specified pars
        self.update_pars(pars=pars, **kwargs)

        # Do we need to check for unmatched pars?

        # set default states

        self.define_states(

            # Contraception
            ss.State('on_contra', default=False),  # whether she's on contraception
            ss.FloatArr('method', default=0),  # Which method to use. 0 used for those on no method
            ss.FloatArr('ti_contra', default=0),  # time point at which to set method
            ss.FloatArr('barrier', default=0),
            ss.State('ever_used_contra', default=False),  # Ever been on contraception. 0 for never having used

            # Sexual and reproductive history
            ss.FloatArr('parity', default=0),
            ss.State('pregnant', default=False),
            ss.State('fertile', default=False),
            ss.State('sexually_active', default=False),
            ss.State('sexual_debut', default=False),
            ss.FloatArr('sexual_debut_age', default=-1),
            ss.FloatArr('fated_debut', default=-1),
            ss.FloatArr('first_birth_age', default=-1),
            ss.State('lactating', default=False),
            ss.FloatArr('gestation', default=0),
            ss.FloatArr('preg_dur', default=0),
            ss.FloatArr('stillbirth', default=0),
            ss.FloatArr('miscarriage', default=0),
            ss.FloatArr('abortion', default=0),
            ss.FloatArr('pregnancies', default=0),
            ss.FloatArr('months_inactive', default=0),
            ss.State('postpartum', default=False),
            ss.FloatArr('mothers', default=-1),
            ss.FloatArr('short_interval', default=0),
            ss.FloatArr('secondary_birth', default=0),
            ss.FloatArr('postpartum_dur', default=0),
            ss.State('lam', default=False),
            ss.FloatArr('breastfeed_dur', default=0),
            ss.FloatArr('breastfeed_dur_total', default=0),

            # Fecundity
            ss.FloatArr('remainder_months', default=0),
            ss.FloatArr('personal_fecundity', default=0),

            # Empowerment - states will remain at these values if use_empowerment is False
            ss.State('paid_employment', default=False),
            ss.State('decision_wages', default=False),
            ss.State('decision_health', default=False),
            ss.State('decision_purchase', default=False),
            ss.State('buy_decision_major', default=False),  # whether she has decision making ability over major purchases
            ss.State('buy_decision_daily', default=False),  # whether she has decision making over daily household purchases
            ss.State('buy_decision_clothes', default=False),  # whether she has decision making over clothing purchases
            ss.State('decide_spending_partner', default=False),  # whether she has decision makking over her partner's wages
            ss.State('has_savings', default=False),  # whether she has savings
            ss.State('has_fin_knowl', default=False),  # whether she knows where to get financial info
            ss.State('has_fin_goals', default=False),  # whether she has financial goals
            ss.State('sexual_autonomy', default=False),  # whether she has ability to refuse sex

            # Composite empowerment attributes
            ss.FloatArr('financial_autonomy', default=0),
            ss.FloatArr('decision_making', default=0),

            # Empowerment - fertility intent
            ss.State('fertility_intent', default=False),
            ss.Arr('categorical_intent', dtype="<U6", default="no" ), # default listed as "cannot", but its overridden with "no" during init
            ss.State('intent_to_use', default=False),
            # for women not on contraception, whether she has intent to use contraception

            # Partnership information -- states will remain at these values if use_partnership is False
            ss.State('partnered', default=False),
            ss.FloatArr('partnership_age', default=-1),

            # Urban (basic demographics) -- state will remain at these values if use_urban is False
            ss.State('urban', default=True),
            ss.Arr('region', dtype=str, default=None),
            ss.FloatArr('wealthquintile', default=3),
            # her current wealth quintile, an indicator of the economic status of her household, 1: poorest quintile; 5: wealthiest quintile

            # Education - states will remain at these values if use_education is False
            ss.FloatArr('edu_objective', default=0),
            ss.FloatArr('edu_attainment', default=0),
            ss.FloatArr('edu_dropout', default=0),
            ss.FloatArr('edu_interrupted', default=0),
            ss.FloatArr('edu_completed', default=0),
            ss.FloatArr('edu_started', default=0),


        )


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

    def update_time_to_choose(self):
        """
        Initialise the counter to determine when girls/women will have to first choose a method.
        """
        ppl = self.sim.people
        fecund_females = (ppl.female) & (ppl.age < self.pars['age_limit_fecundity'])
        time_to_debut = (self.fated_debut[fecund_females] - ppl.age[fecund_females]) / self.t.dt
        self.ti_contra[fecund_females] = np.maximum(time_to_debut, 0)

        # Validation
        time_to_set_contra = self.ti_contra[fecund_females] == 0
        if not np.array_equal(((ppl.age[fecund_females] - self.fated_debut[fecund_females]) > -self.t.dt), time_to_set_contra):
            errormsg = 'Should be choosing contraception for everyone past fated debut age.'
            raise ValueError(errormsg)
        return

    def update_fertility_intent(self):
        if self.pars['fertility_intent'] is None:
            return
        self.update_fertility_intent_by_age()
        return

    def update_intent_to_use(self):
        if self.pars['intent_to_use'] is None:
            return
        self.update_intent_to_use_by_age()
        return

    def update_wealthquintile(self):
        if self.pars['wealth_quintile'] is None:
            return
        wq_probs = self.pars['wealth_quintile']['percent']
        vals = np.random.choice(len(wq_probs), size=n, p=wq_probs) + 1
        self.wealthquintile = vals
        return


    def get_urban(self, n):
        """ Get initial distribution of urban """
        urban_prop = self.pars['urban_prop']
        urban = fpu.n_binomial(urban_prop, n)
        return urban

    def set_age_sex(self):
        """
        Override the default age and sex of the population to match the age pyramid
        """
        pyramid = self.pars['age_pyramid']
        m_frac = pyramid[:, 1].sum() / pyramid[:, 1:3].sum()

        ppl = self.sim.people
        n = len(ppl)

        ages = np.zeros(n)
        sexes = np.random.random(n) < m_frac  # Pick the sex based on the fraction of men vs. women



        # TODO clean this up using ss people filters
        f_inds = sc.findinds(sexes == 0)
        m_inds = sc.findinds(sexes == 1)
        ppl.female[ss.uids(f_inds)] = True
        ppl.female[ss.uids(m_inds)] = False

        age_data_min = pyramid[:, 0]
        age_data_max = np.append(pyramid[1:, 0], self.pars['max_age'])
        age_data_range = age_data_max - age_data_min
        for i, inds in enumerate([m_inds, f_inds]):
            if len(inds):
                age_data_prob = pyramid[:, i + 1]
                age_data_prob = age_data_prob / age_data_prob.sum()  # Ensure it sums to 1
                age_bins = fpu.n_multinomial(age_data_prob, len(inds))  # Choose age bins
                #ages[inds] = age_data_min[age_bins] + age_data_range[age_bins] * np.random.random(
                #    len(inds))  # Uniformly distribute within this age bin
                ppl.age[ss.uids(inds)] = age_data_min[age_bins] + age_data_range[age_bins] * np.random.random(
                    len(inds))  # Uniformly distribute within this age bin

        return ages, sexes

    def init_pre(self, sim, force=False):
        super().init_pre(sim, force)
        # add additional n-dimensional stats to track. We initialize here because we need to know n_agents
        self.child_inds = np.full(shape=(sim.pars['n_agents'], max_parity), fill_value=-1, dtype=int)
        self.birth_ages = np.full(shape=(sim.pars['n_agents'], max_parity), fill_value=np.nan, dtype=float)
        self.stillborn_ages = np.full(shape=(sim.pars['n_agents'], max_parity), fill_value=np.nan, dtype=float)
        self.miscarriage_ages = np.full(shape=(sim.pars['n_agents'], max_parity), fill_value=np.nan, dtype=float)
        self.abortion_ages = np.full(shape=(sim.pars['n_agents'], max_parity), fill_value=np.nan, dtype=float)


    def init_post(self):
        # update the age and sex of each agent
        super().init_post()

        self.set_age_sex()


        if not self.pars['use_subnational']:
            _urban = self.get_urban(len(self.sim.people))
        else:
            _urban = fpsn.get_urban_init_vals(self.sim.people)
        self.urban = _urban

        # Parameters on sexual and reproductive history
        self.fertile = self.pars['p_fertile'].filter()



        # Fertility intent
        # Update distribution of fertility intent with location-specific values if it is present in self.pars
        self.update_fertility_intent()

        # Intent to use contraception
        self.update_intent_to_use()

        # Update the distribution of wealth quintile
        self.update_wealthquintile()

        # Default initialization for fated_debut; subnational debut initialized in subnational.py otherwise
        if not self.pars['use_subnational']:
            # TODO see if we can swap out fpu.n_multinomial for ss.choice or other dist?
            self.fated_debut = self.pars['debut_age']['ages'][fpu.n_multinomial(self.pars['debut_age']['probs'], self.sim.people.n_uids)]
        else:
            self.fated_debut = fpsn.get_debut_init_vals(self)

        # Fecundity variation
        fv = [self.pars['fecundity_var_low'], self.pars['fecundity_var_high']]
        fac = (fv[1] - fv[0]) + fv[0]  # Stretch fecundity by a factor bounded by [f_var[0], f_var[1]]
        self.personal_fecundity = self.pars.p_personal_fecundity.rvs() * fac

        # Initialise ti_contra based on age and fated debut
        self.update_time_to_choose()


        if self.empowerment_module is not None:
            self.empowerment_module.initialize(self.filter(self.is_female))

        if self.education_module is not None:
            self.education_module.initialize(self)

        # Partnership
        if self.pars['use_partnership']:
            init_partnership_states(self.sim.people)

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

    #def init_post(self):





# %% Initialization methods
def init_urban_states(ppl):
    """Demographics on whether a person lives in a rural or urban setting"""
    # Get init vals and populate state in one step
    ppl.urban = get_urban_init_vals(ppl)


def init_partnership_states(ppl):
    """Demographics on whether a person is in a partnership, and their expected age at first partnership in a rural or urban setting"""

    # Get init values for these sociodemographic states
    partnered, partnership_age = get_partnership_init_vals(ppl)

    # Populate states
    ppl.partnered = partnered
    ppl.partnership_age = partnership_age


def get_urban_init_vals(ppl, urban_prop=None):
    """ Get initial distribution of urban """

    n = len(ppl)
    urban = np.ones(n, dtype=bool)

    if urban_prop is None:
        if ppl.pars['urban_prop'] is not None:
            urban_prop = ppl.pars['urban_prop']

    if urban_prop is not None:
        urban = fpu.n_binomial(urban_prop, n)

    return urban


def get_partnership_init_vals(ppl):
    """Get initial distribution of age at first partnership from location-specific data"""
    partnership_data = ppl.pars['age_partnership']
    n = len(ppl)
    partnered = np.zeros(n, dtype=bool)
    partnership_age = np.zeros(n, dtype=float)

    # Get female agents indices and ages
    f_inds = sc.findinds(ppl.is_female)
    f_ages = ppl.age[f_inds]

    # Select age at first partnership
    partnership_age[f_inds] = np.random.choice(partnership_data['age'], size=len(f_inds),
                                               p=partnership_data['partnership_probs'])

    # Check if age at first partnership => than current age to set partnered
    p_inds = sc.findinds((f_ages >= partnership_age[f_inds]))
    partnered[f_inds[p_inds]] = True

    return partnered, partnership_age
