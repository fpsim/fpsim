"""
Methods and functions related to empowerment
"""

# %% Imports
import numpy as np  # Needed for a few things not provided by pl
import sciris as sc
from . import utils as fpu
from . import defaults as fpd


# %% Initialization methods

def init_empowerment(ppl):
    """ Initialize empowerment metrics """

    # Don't use this function if not modeling empowerment
    if not ppl.pars['use_empowerment']:
        return

    # Sociodemographic attributes
    partnered, partnership_age = initialize_partnered(ppl)
    ppl.partnered = partnered
    ppl.partnership_age = partnership_age
    ppl.urban = initialize_urban(ppl)

    # Initialize empowerment-related attributes
    emp = initialize_empowerment(ppl)
    edu = initialize_education(ppl)
    ppl.decision_wages = emp['decision_wages']  # Decision-making autonomy over household purchases/wages
    ppl.decision_health = emp['decision_health']  # Decision-making autonomy over her health

    # Empowerment-education attributes
    ppl.edu_objective = edu['edu_objective']  # Highest-ideal level of education to be completed (in years)
    ppl.edu_attainment = edu['edu_attainment']  # Current level of education achieved in years
    ppl.edu_dropout = edu['edu_dropout']  # Whether a person has dropped out before reaching their goal
    ppl.edu_interrupted = edu['edu_interrupted']  # Whether a person/woman has had their education temporarily interrupted, but can resume
    ppl.edu_completed = edu['edu_completed']  # Whether a person/woman has reached their education goals
    ppl.edu_started = edu['edu_started']  # Whether a person/woman has started thier education

    return


def initialize_urban(ppl, urban_prop=None):
    """ Get initial distribution of urban """

    n = len(ppl)
    urban = np.ones(n, dtype=bool)

    if urban_prop is None:
        if ppl.pars['urban_prop'] is not None:
            urban_prop = ppl.pars['urban_prop']

    if urban_prop is not None:
        urban = fpu.n_binomial(urban_prop, n)

    return urban


def initialize_partnered(ppl):
    """Get initial distribution of age at first partnership"""
    partnership_data = ppl.pars['age_partnership']
    n = len(ppl)
    partnered = np.zeros(n, dtype=bool)
    partnership_age = np.zeros(n, dtype=float)

    # Get female agents indices and ages
    f_inds = sc.findinds(ppl.sex == 0)
    f_ages = ppl.age[f_inds]

    # Select age at first partnership
    partnership_age[f_inds] = np.random.choice(partnership_data['age'], size=len(f_inds),
                                               p=partnership_data['partnership_probs'])

    # Check if age at first partnership => than current age to set partnered
    p_inds = sc.findinds((f_ages >= partnership_age[f_inds]))
    partnered[f_inds[p_inds]] = True

    return partnered, partnership_age


def initialize_empowerment(ppl):
    """Get initial distribution of women's empowerment metrics/attributes"""
    # NOTE: we assume that either probabilities or metrics in empowerment_dict are defined over all possible ages
    # from 0 to 100 years old.
    n = len(ppl)
    empowerment_dict = ppl.pars['empowerment']

    # Empowerment dictionary
    empowerment = {}
    empowerment['paid_employment'] = np.zeros(n, dtype=bool)
    empowerment['sexual_autonomy'] = np.zeros(n, dtype=float)
    empowerment['decision_wages'] = np.zeros(n, dtype=float)
    empowerment['decision_health'] = np.zeros(n, dtype=float)

    # Get female agents indices and ages
    f_inds = sc.findinds(ppl.sex == 0)
    f_ages = ppl.age[f_inds]

    # Create age bins
    age_cutoffs = np.hstack((empowerment_dict['age'], empowerment_dict['age'].max() + 1))
    age_inds = np.digitize(f_ages, age_cutoffs) - 1

    # Paid employment
    paid_employment_probs = empowerment_dict['paid_employment']
    empowerment['paid_employment'][f_inds] = fpu.binomial_arr(paid_employment_probs[age_inds])

    for metric in ['decision_wages', 'decision_health', 'sexual_autonomy']:
        empowerment[metric][f_inds] = empowerment_dict[metric][age_inds]

    return empowerment


def initialize_education(ppl):
    """Get initial distribution of education goal, attainment and whether
    a woman has reached their education goal"""
    education_dict = ppl.pars['education']
    n = len(ppl)
    urban = ppl.urban

    # Initialise individual education attainment - number of education years completed at start of simulation
    # Assess whether a woman has completed her education based on the values of the two previous attributes
    # Education dictionary
    education = {'edu_objective': np.zeros(n, dtype=float),
                 'edu_attainment': np.zeros(n, dtype=float),
                 'edu_started': np.zeros(n, dtype=bool),
                 'edu_interrupted': np.zeros(n, dtype=bool),
                 'edu_completed': np.zeros(n, dtype=bool),
                 'edu_dropout': np.zeros(n, dtype=bool)}

    # Initialise individual education objectives from a 2d array of probs with dimensions (urban, edu_years)
    f_inds_urban = sc.findinds(ppl.sex == 0, urban == True)
    f_inds_rural = sc.findinds(ppl.sex == 0, urban == False)

    # Set objectives
    probs_rural = education_dict['edu_objective'][1, :]
    probs_urban = education_dict['edu_objective'][0, :]
    edu_years = np.arange(len(probs_rural))
    education['edu_objective'][f_inds_rural] = np.random.choice(edu_years, size=len(f_inds_rural),
                                                                p=probs_rural)  # Probs in rural settings
    education['edu_objective'][f_inds_urban] = np.random.choice(edu_years, size=len(f_inds_urban),
                                                                p=probs_urban)  # Probs in urban settings

    # Initialise education attainment - ie, current state of education at the start of the simulation
    f_inds = sc.findinds(ppl.sex == 0)

    # Get ages for female agents and round them so we can use them as indices
    f_ages = np.floor(ppl.age[f_inds]).astype(int)
    # Set the initial number of education years an agent has based on her age
    education['edu_attainment'][f_inds] = np.floor((education_dict['edu_attainment'][f_ages]))

    # Check people who started their education
    started_inds = sc.findinds(education['edu_attainment'][f_inds] > 0.0)
    # Check people who completed their education
    completed_inds = sc.findinds(education['edu_objective'][f_inds] - education['edu_attainment'][f_inds] <= 0)
    # Set attainment to edu_objective, just in case that initial edu_attainment > edu_objective
    education['edu_attainment'][f_inds[completed_inds]] = education['edu_objective'][f_inds[completed_inds]]
    education['edu_completed'][f_inds[completed_inds]] = True
    education['edu_started'][f_inds[started_inds]] = True

    return education


# %% Methods to update education

def check_education(ppl):
    start_education(ppl)  # Check if anyone needs to start school
    update_education(ppl)  # Advance attainment, determine who reaches their objective, who dropouts, who has their education interrupted
    resume_education(ppl)  # Determine who goes back to school after an interruption
    graduate(ppl)  # Check if anyone achieves their education goal


def start_education(ppl):
    """
    Begin education
    """
    new_students = ppl.filter(~ppl.edu_started & (ppl.age >= ppl.pars["education"]["age_start"]))
    new_students.edu_started = True


def interrupt_education(ppl):
    """
    Interrupt education due to pregnancy. This method hinders education progression if a
    woman is pregnant and towards the end of the first trimester
    """
    # Hinder education progression if a woman is pregnant and towards the end of the first trimester
    pregnant_students = ppl.filter(ppl.pregnant)
    end_first_tri = pregnant_students.filter(pregnant_students.gestation == ppl.pars['end_first_tri'])
    # Disrupt education
    end_first_tri.edu_interrupted = True


def dropout_education(ppl, parity):
    dropout_dict = ppl.pars['education']['edu_dropout_probs'][parity]
    age_cutoffs = np.hstack((dropout_dict['age'], dropout_dict['age'].max() + 1))
    age_inds = np.digitize(ppl.age, age_cutoffs) - 1
    # Decide who will drop out
    ppl.edu_dropout = fpu.binomial_arr(dropout_dict['percent'][age_inds])


def update_education(ppl):
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
    pospartum_students = ppl.filter(
        ppl.postpartum & ppl.edu_interrupted & ~ppl.edu_completed & ~ppl.edu_dropout)
    resume_inds = sc.findinds(pospartum_students.postpartum_dur > 0.5 * ppl.pars['postpartum_dur'])
    tmp = pospartum_students.edu_interrupted
    tmp[resume_inds] = False
    pospartum_students.edu_interrupted = tmp


def graduate(ppl):
    completed_inds = sc.findinds(ppl.edu_attainment >= ppl.edu_objective)
    # NOTE: the two lines below were necessary because edu_completed was not being updating as expected
    tmp = ppl.edu_completed
    tmp[completed_inds] = True
    ppl.edu_completed = tmp

