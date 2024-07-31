"""
Methods and functions related to empowerment and education
TODO: maybe move education stuff to another module, education.py
"""

# %% Imports
import numpy as np  # Needed for a few things not provided by pl
import sciris as sc
from . import utils as fpu
from . import defaults as fpd


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


def init_empowerment_states(ppl):
    """
    If ppl.pars['use_empowerment'] == True, location-specific data
    are expected to exist to populate empowerment states/attributes.

    If If ppl.pars['use_empowerment'] == False, related
    attributes will be initialized with values found in defaults.py.
    """

    # Initialize empowerment-related attributes with location-specific data
    # TODO: check whether the corresponding data dictionary exists in ppl.pars,
    # if it doesn't these states could be intialised with a user-defined distribution.
    emp = get_empowerment_init_vals(ppl)

    # Populate empowerment states with location-specific data
    ppl.paid_employment = emp['paid_employment']
    ppl.sexual_autonomy = emp['sexual_autonomy']
    ppl.decision_wages  = emp['decision_wages']   # Decision-making autonomy over household purchases/wages
    ppl.decision_health = emp['decision_health']  # Decision-making autonomy over her health
    return


def init_education_states(ppl):
    """
    If ppl.pars['use_education'] == True, location-specific data
    are expected to exist to populate education states/attributes.

    If If ppl.pars['use_education'] == False, related
    attributes will be initialized with values found in defaults.py.
    """

    # Get distributions of initial values from location-specific data
    edu = get_education_init_vals(ppl)

    # Populate People's education states
    ppl.edu_objective = edu['edu_objective']  # Highest-ideal level of education to be completed (in years)
    ppl.edu_attainment = edu['edu_attainment']  # Current level of education achieved in years
    ppl.edu_dropout = edu['edu_dropout']  # Whether a person has dropped out before reaching their goal
    ppl.edu_interrupted = edu['edu_interrupted']  # Whether a person/woman has had their education temporarily interrupted, but can resume
    ppl.edu_completed = edu['edu_completed']  # Whether a person/woman has reached their education goals
    ppl.edu_started = edu['edu_started']  # Whether a person/woman has started thier education


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


def get_empowerment_init_vals(ppl):
    """
    Initialize empowerment atrtibutes with location-specific data,
    expected to be found in ppl.pars['empowerment']

    # NOTE-PSL: this function could be generally used to update empowerment states as a function of age,
    at every time step. Probably need to rename it to get_vals_from_data() or just get_empowerment_vals(), or
    something else.

    >> subpop = ppl.filter(some_criteria)
    >> get_empowerment()
    """
    empowerment_dict = ppl.pars['empowerment']
    # NOTE: we assume that either probabilities or metrics in empowerment_dict are defined over all possible ages
    # from 0 to 100 years old.
    n = len(ppl)

    empwr_states = ['paid_employment', 'sexual_autonomy', 'decision_wages', 'decision_health']
    empowerment = {empwr_state: np.zeros(n, dtype=fpd.person_defaults[empwr_state].dtype) for empwr_state in empwr_states}

    # Get female agents indices and ages
    f_inds = sc.findinds(ppl.is_female)
    f_ages = ppl.age[f_inds]

    # Create age bins because ppol.age is a continous variable
    age_cutoffs = np.hstack((empowerment_dict['age'], empowerment_dict['age'].max() + 1))
    age_inds = np.digitize(f_ages, age_cutoffs) - 1

    # Paid employment
    paid_employment_probs = empowerment_dict['paid_employment']
    empowerment['paid_employment'][f_inds] = fpu.binomial_arr(paid_employment_probs[age_inds])

    # Make other metrics
    for metric in ['decision_wages', 'decision_health', 'sexual_autonomy']:
        empowerment[metric][f_inds] = empowerment_dict[metric][age_inds]

    return empowerment


def get_education_init_vals(ppl):
    """
    Define initial values (distributions) for People's education attributes from location-specific data,
    expected to be found in ppl.pars

    Get initial distributions of education goal, attainment and whether
    a woman has reached their education goal
    """
    education_data = ppl.pars['education'] # Location-specific education data
    n = len(ppl)

    # Initialise individual education attainment (number of education years completed at start of asimulation)
    # Assess whether a woman has completed her education based on the values of the education attainment and
    edu_states = ['edu_objective',
                  'edu_attainment',
                  'edu_started',
                  'edu_interrupted',
                  'edu_completed',
                  'edu_dropout']
    education = {edu_state: np.zeros(n, dtype=fpd.person_defaults[edu_state].dtype) for edu_state in edu_states}

    # Initialise individual education objectives from a 2d array of probs with dimensions (urban, edu_years)
    f_inds_urban = sc.findinds(ppl.is_female & ppl.urban)
    f_inds_rural = sc.findinds(ppl.is_female & ~ppl.urban)

    # Set objectives based on geo setting
    probs_urban = education_data['edu_objective'][0, :]
    probs_rural = education_data['edu_objective'][1, :]

    edu_years = np.arange(len(probs_rural))
    education['edu_objective'][f_inds_rural] = np.random.choice(edu_years, size=len(f_inds_rural),
                                                                p=probs_rural)  # Probs in rural settings
    education['edu_objective'][f_inds_urban] = np.random.choice(edu_years, size=len(f_inds_urban),
                                                                p=probs_urban)  # Probs in urban settings

    # Initialise education attainment - ie, current state of education at the start of the simulation
    f_inds = sc.findinds(ppl.is_female)

    # Get ages for female agents and round them so we can use them as indices
    f_ages = np.floor(ppl.age[f_inds]).astype(int)
    # Set the initial number of education years an agent has based on her age
    education['edu_attainment'][f_inds] = np.floor((education_data['edu_attainment'][f_ages]))

    # Check people who started their education
    started_inds = sc.findinds(education['edu_attainment'][f_inds] > 0.0)
    # Check people who completed their education
    completed_inds = sc.findinds(education['edu_objective'][f_inds] - education['edu_attainment'][f_inds] <= 0.0)
    # Set attainment to edu_objective, for cases that initial edu_attainment > edu_objective
    education['edu_attainment'][f_inds[completed_inds]] = education['edu_objective'][f_inds[completed_inds]]
    education['edu_completed'][f_inds[completed_inds]] = True
    education['edu_started'][f_inds[started_inds]] = True

    return education

# %% Methods to update empowerment attributes based on an agent's age

def update_decision_health(ppl):
    pass
    # """Assumes ppl object received is only female agents"""
    # age_inds = np.round(ppl.age).astype(int)
    # ppl.decision_health = ppl.pars['empowerment']['decision_health'][age_inds]


def update_decision_wages(ppl):
    pass
    # age_inds = np.round(ppl.age).astype(int)
    # ppl.decision_health = ppl.pars['empowerment']['decision_wages'][age_inds]


def update_paid_employment(ppl):
    pass


def update_sexual_autonomy(ppl):
    pass


def update_empowerment(ppl):
    """
    Update empowerment metrics based on age(ing).
    ppl is assumed to be a filtered People object, with only the agents who have had their bdays.
    """
    # This would update the corresponding attributes from location-specific data based on age
    update_decision_wages(ppl)
    update_decision_health(ppl)
    update_paid_employment(ppl)
    update_sexual_autonomy(ppl)



# %% Methods to update education

def update_education(ppl):
    start_education(ppl)  # Check if anyone needs to start school
    advance_education(ppl)  # Advance attainment, determine who reaches their objective, who dropouts, who has their education interrupted
    resume_education(ppl)  # Determine who goes back to school after an interruption
    graduate(ppl)  # Check if anyone achieves their education goal


def start_education(ppl):
    """
    Begin education
    """
    new_students = ppl.filter(~ppl.edu_started & (ppl.age >= ppl.pars["education"]["age_start"]))
    new_students.edu_started = True
    # TODO: check whether their edu_objective is zero and set it here?


def interrupt_education(ppl):
    """
    Interrupt education due to pregnancy. This method hinders education progression if a
    woman is pregnant and towards the end of the first trimester
    """
    # Hinder education progression if a woman is pregnant and towards the end of the first trimester
    pregnant_students = ppl.filter(ppl.pregnant & (ppl.gestation == ppl.pars['end_first_tri']))
    # Disrupt education
    pregnant_students.edu_interrupted = True


def dropout_education(ppl, parity):
    dropout_dict = ppl.pars['education']['edu_dropout_probs'][parity]
    age_cutoffs = dropout_dict['age']  # bin edges
    age_inds = np.searchsorted(age_cutoffs, ppl.age,"right") - 1  # NOTE: faster than np.digitize for large arrays
    # Decide who will drop out
    ppl.edu_dropout = fpu.binomial_arr(dropout_dict['percent'][age_inds])


def advance_education(ppl):
    """
    Advance education attainment in the simulation, determine if agents have completed their education
    """

    # Filter people who have not: completed education, dropped out or had their education interrupted
    students = ppl.filter((ppl.edu_started & ~ppl.edu_completed & ~ppl.edu_dropout & ~ppl.edu_interrupted))
    # Advance education attainment
    students.edu_attainment += ppl.pars['timestep'] / fpd.mpy
    # Check who will experience an interruption
    interrupt_education(students)
    # Make some students dropout based on dropout | parity probabilities
    par1 = students.filter(students.parity == 1)
    dropout_education(par1, '1')  # Women with parity 1
    par2plus = students.filter(students.parity >= 2)
    dropout_education(par2plus, '2+')  # Women with parity 2+


def resume_education(ppl):
    """
    # Basic mechanism to resume education post-pregnancy:
    # If education was interrupted due to pregnancy, resume after 9 months pospartum ()
    #TODO: check if there's any evidence supporting this assumption
    """
    # Basic mechanism to resume education post-pregnancy:
    # If education was interrupted due to pregnancy, resume after 9 months pospartum
    filter_conds = (ppl.postpartum & ppl.edu_interrupted & ~ppl.edu_completed & ~ppl.edu_dropout &
                    (ppl.postpartum_dur > 0.5 * ppl.pars['postpartum_dur']))
    postpartum_students = ppl.filter(filter_conds)
    postpartum_students.edu_interrupted = False


def graduate(ppl):
    completed_inds = sc.findinds(ppl.edu_attainment >= ppl.edu_objective)
    tmp = ppl.edu_completed
    tmp[completed_inds] = True
    ppl.edu_completed = tmp

