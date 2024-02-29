"""
Methods and functions related to education
"""

# %% Imports
import numpy as np  # Needed for a few things not provided by pl
import sciris as sc
from . import utils as fpu
from . import defaults as fpd


# %% Initialization methods


def init_partnership_states(ppl):
    """Demographics on whether a person is in a partnership, and their expected age at first partnership in a rural or urban setting"""

    # Get init values for these sociodemographic states
    partnered, partnership_age = get_partnership_init_vals(ppl)

    # Populate states
    ppl.partnered = partnered
    ppl.partnership_age = partnership_age


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


def get_education_init_vals(ppl):
    """
    Define initial values (distributions) for People's education attributes from location-specific data,
    expected to be found in ppl.pars

    Get initial distributions of education goal, attainment and whether
    a woman has reached their education goal
    """
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
    f_inds_urban = sc.findinds(ppl.is_female & ppl.urban)
    f_inds_rural = sc.findinds(ppl.is_female & ~ppl.urban)

    # Set objectives based on geo setting
    probs_urban = education_dict['edu_objective'][0, :]
    probs_rural = education_dict['edu_objective'][1, :]

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
    education['edu_attainment'][f_inds] = np.floor((education_dict['edu_attainment'][f_ages]))

    # Check people who started their education
    started_inds = sc.findinds(education['edu_attainment'][f_inds] > 0.0)
    # Check people who completed their education
    completed_inds = sc.findinds(education['edu_objective'][f_inds] - education['edu_attainment'][f_inds] <= 0.0)
    # Set attainment to edu_objective, for cases that initial edu_attainment > edu_objective
    education['edu_attainment'][f_inds[completed_inds]] = education['edu_objective'][f_inds[completed_inds]]
    education['edu_completed'][f_inds[completed_inds]] = True
    education['edu_started'][f_inds[started_inds]] = True

    return education



# %% Class for updating education

class Education:
    def __init__(self):
        """ TODO: move par initialization here? """
        return

    def update(self, ppl):
        self.start_education(ppl)  # Check if anyone needs to start school
        self.advance_education(ppl)  # Advance attainment, determine who reaches their objective, who dropouts, who has their education interrupted
        self.resume_education(ppl)  # Determine who goes back to school after an interruption
        self.graduate(ppl)  # Check if anyone achieves their education goal

    @staticmethod
    def start_education(ppl):
        """
        Begin education
        """
        new_students = ppl.filter(~ppl.edu_started & (ppl.age >= ppl.pars["education"]["age_start"]))
        new_students.edu_started = True

    @staticmethod
    def interrupt_education(ppl):
        """
        Interrupt education due to pregnancy. This method hinders education progression if a
        woman is pregnant and towards the end of the first trimester
        """
        # Hinder education progression if a woman is pregnant and towards the end of the first trimester
        pregnant_students = ppl.filter(ppl.pregnant & (ppl.gestation == ppl.pars['end_first_tri']))
        # Disrupt education
        pregnant_students.edu_interrupted = True

    @staticmethod
    def dropout_education(ppl, parity):
        dropout_dict = ppl.pars['education']['edu_dropout_probs'][parity]
        age_cutoffs = dropout_dict['age']  # bin edges
        age_inds = np.searchsorted(age_cutoffs, ppl.age,"right") - 1  # NOTE: faster than np.digitize for large arrays
        # Decide who will drop out
        ppl.edu_dropout = fpu.binomial_arr(dropout_dict['percent'][age_inds])

    def advance_education(self, ppl):
        """
        Advance education attainment in the simulation, determine if agents have completed their education
        """

        # Filter people who have not: completed education, dropped out or had their education interrupted
        students = ppl.filter((ppl.edu_started & ~ppl.edu_completed & ~ppl.edu_dropout & ~ppl.edu_interrupted))
        # Advance education attainment
        students.edu_attainment += ppl.pars['timestep'] / fpd.mpy
        # Check who will experience an interruption
        self.interrupt_education(students)
        # Make some students dropout based on dropout | parity probabilities
        par1 = students.filter(students.parity == 1)
        self.dropout_education(par1, '1')  # Women with parity 1
        par2plus = students.filter(students.parity >= 2)
        self.dropout_education(par2plus, '2+')  # Women with parity 2+

    @staticmethod
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

    @staticmethod
    def graduate(ppl):
        completed_inds = sc.findinds(ppl.edu_attainment >= ppl.edu_objective)
        tmp = ppl.edu_completed
        tmp[completed_inds] = True
        ppl.edu_completed = tmp

