"""
Methods and functions related to empowerment
<<<<<<< HEAD
We consider four empowerment metrics: paid_employment, decision_health, decision_wages, sexual_autonomy
=======
>>>>>>> main
"""

# %% Imports
import numpy as np
import sciris as sc
from . import utils as fpu
from . import locations as fplocs


# %% Class for updating empowerment

<<<<<<< HEAD
class Empowerment:
    def __init__(self, location='kenya', seed=None, empowerment_file=None):

        # Handle location
        location = location.lower()
        if location == 'kenya':
            empowerment_dict, _ = fplocs.kenya.empowerment_distributions(seed=seed)
            self.pars = fplocs.kenya.empowerment_update_pars()
            self.empowerment_dict = empowerment_dict
        else:
            errormsg = f'Location "{location}" is not currently supported for empowerment analyses'
            raise NotImplementedError(errormsg)

        self.metrics = list(self.pars.keys())

        return

    def initialize(self, ppl):
        """
        Initialize by setting values for people
        """
        empowerment_dict = self.empowerment_dict

        # Get female agents indices and ages
        f_inds = sc.findinds(ppl.is_female)
        f_ages = ppl.age[f_inds]

        # Create age bins because ppl.age is a continous variable
        age_cutoffs = np.hstack((empowerment_dict['age'], empowerment_dict['age'].max() + 1))
        age_inds = np.digitize(f_ages, age_cutoffs) - 1

        # Paid employment
        paid_employment_probs = empowerment_dict['paid_employment']
        ppl.paid_employment[f_inds] = fpu.binomial_arr(paid_employment_probs[age_inds])
=======
def init_empowerment_states(ppl):
    """
    If ppl.pars['use_empowerment'] == True, location-specific data
    are expected to exist to populate empowerment states/attributes.
>>>>>>> main

        # Populate empowerment states with location-specific data
        ppl.sexual_autonomy[age_inds] = empowerment_dict['sexual_autonomy'][age_inds]
        ppl.decision_wages[age_inds] = empowerment_dict['decision_wages'][age_inds]
        ppl.decision_health[age_inds] = empowerment_dict['decision_health'][age_inds]

        return

    def update(self, ppl):
        for metric in self.metrics:
            p = self.pars[metric]
            rhs = p.intercept
            for vname, vval in p.items():
                if vname not in ['intercept', 'wealthquintile']:
                    rhs += vval * ppl[vname]

            prob_1 = 1 / (1+np.exp(-rhs))

<<<<<<< HEAD
            new_vals = fpu.binomial_arr(prob_1)
            changers = (new_vals != ppl[metric]).nonzero()[-1]  # People whose empowerment changes
            ppl.ti_contra_update[changers] = ppl.ti  # Trigger update to contraceptive choices if empowerment changes
            ppl[metric] = new_vals

        return
=======
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
>>>>>>> main
