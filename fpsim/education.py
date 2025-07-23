"""
Methods and functions related to education
"""

# %% Imports
import numpy as np  # Needed for a few things not provided by pl
from . import utils as fpu
from . import defaults as fpd
from . import locations as fplocs



# %% Class for updating education

class Education:
    def __init__(self, location=None):
        # Handle location
        location = fpd.get_location(location)
        # Get the correct module, from either registry or built-in
        if location in fpd.location_registry:
            location_module = fpd.location_registry[location]
        else:
            location_module = fplocs  # fallback to built-in only if not registered

        education_dict, _ = location_module.data_utils.education_distributions(location)  # This function returns extrapolated and raw data
        self.pars = education_dict
        return

    def initialize(self, ppl, uids):
        """ Initialize with people """
        education_dict = self.pars

        # Initialise individual education objectives from a 2d array of probs with dimensions (urban, edu_years)
        f_uids_urban = uids[(ppl.female[uids] & ppl.urban[uids])]
        f_uids_rural = uids[(ppl.female[uids] & ~ppl.urban[uids])]

        # Set objectives based on geo setting
        probs_urban = education_dict['edu_objective'][0, :]
        probs_rural = education_dict['edu_objective'][1, :]

        edu_years = np.arange(len(probs_rural))
        ppl.edu_objective[f_uids_rural] = np.random.choice(edu_years, size=len(f_uids_rural),
                                                                    p=probs_rural)  # Probs in rural settings
        ppl.edu_objective[f_uids_urban] = np.random.choice(edu_years, size=len(f_uids_urban),
                                                                    p=probs_urban)  # Probs in urban settings

        # Initialise education attainment - ie, current state of education at the start of the simulation
        f_uids = (ppl.female).uids

        # Get ages for female agents and round them so we can use them as indices
        f_ages = np.floor(ppl.age[f_uids]).astype(int)
        # Set the initial number of education years an agent has based on her age
        ppl.edu_attainment[f_uids] = np.floor((education_dict['edu_attainment'][f_ages]))

        # Check people who started their education
        started_uids = f_uids[(ppl.edu_attainment[f_uids] > 0.0)]
        # Check people who completed their education
        completed_uids = f_uids[(ppl.edu_objective[f_uids] - ppl.edu_attainment[f_uids] <= 0.0)]
        # Set attainment to edu_objective, for cases that initial edu_attainment > edu_objective
        ppl.edu_attainment[completed_uids] = ppl.edu_objective[completed_uids]
        ppl.edu_completed[completed_uids] = True
        ppl.edu_started[started_uids] = True

        return

    def update(self, ppl, uids):
        if uids is None:
            uids = ppl.alive.uids
        self.start_education(ppl, uids)  # Check if anyone needs to start school
        self.advance_education(ppl, uids)  # Advance attainment, determine who reaches their objective, who dropouts, who has their education interrupted
        self.resume_education(ppl, uids)  # Determine who goes back to school after an interruption
        self.graduate(ppl, uids)  # Check if anyone achieves their education goal

    def start_education(self, ppl, uids):
        """
        Begin education
        """
        new_students = uids[(~ppl.edu_started[uids] & (ppl.age[uids] >= self.pars["age_start"]))]
        ppl.edu_started[new_students] = True

    @staticmethod
    def interrupt_education(ppl, uids):
        """
        Interrupt education due to pregnancy. This method hinders education progression if a
        woman is pregnant and towards the end of the first trimester
        """
        # Hinder education progression if a woman is pregnant and towards the end of the first trimester
        pregnant_students = uids[(ppl.pregnant[uids] & (ppl.gestation[uids] == ppl.sim.fp_pars['end_first_tri']))]
        # Disrupt education
        ppl.edu_interrupted[pregnant_students] = True

    def dropout_education(self, ppl, uids, parity):
        dropout_dict = self.pars['edu_dropout_probs'][parity]
        age_cutoffs = dropout_dict['age']  # bin edges
        age_uids = np.searchsorted(age_cutoffs, ppl.age[uids], "right") - 1  # NOTE: faster than np.digitize for large arrays
        # Decide who will drop out
        ppl.edu_dropout[uids] = fpu.binomial_arr(dropout_dict['percent'][age_uids])

    def advance_education(self, ppl, uids):
        """
        Advance education attainment in the simulation, determine if agents have completed their education
        """

        # Filter people who have not: completed education, dropped out or had their education interrupted
        students = uids[(ppl.edu_started[uids] & ~ppl.edu_completed[uids] & ~ppl.edu_dropout[uids] & ~ppl.edu_interrupted[uids])]
        # Advance education attainment
        ppl.edu_attainment[students] += ppl.sim.t.dt_year
        # Check who will experience an interruption
        self.interrupt_education(ppl, students)
        # Make some students dropout based on dropout | parity probabilities
        par1 = students[(ppl.parity[students] == 1)]
        self.dropout_education(ppl, par1, '1')  # Women with parity 1
        par2plus = students[(ppl.parity[students] >= 2)]
        self.dropout_education(ppl, par2plus, '2+')  # Women with parity 2+

    @staticmethod
    def resume_education(ppl, uids):
        """
        # Basic mechanism to resume education post-pregnancy:
        # If education was interrupted due to pregnancy, resume after 9 months pospartum ()
        #TODO: check if there's any evidence supporting this assumption
        """
        # Basic mechanism to resume education post-pregnancy:
        # If education was interrupted due to pregnancy, resume after 9 months pospartum
        postpartum_students = uids[(ppl.postpartum[uids] & ppl.edu_interrupted[uids] & ~ppl.edu_completed[uids] & ~ppl.edu_dropout[uids] &
                        (ppl.postpartum_dur[uids] > 0.5 * ppl.sim.fp_pars['postpartum_dur']))]
        ppl.edu_interrupted[postpartum_students] = False

    @staticmethod
    def graduate(ppl, uids):
        completed_uids = uids[(ppl.edu_attainment[uids] >= ppl.edu_objective[uids])]
        ppl.edu_completed[completed_uids] = True