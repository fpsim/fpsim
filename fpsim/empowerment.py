"""
Methods and functions related to empowerment
We consider four empowerment metrics: paid_employment, decision_health, decision_wages, sexual_autonomy
"""

# %% Imports
import numpy as np
import sciris as sc
from . import utils as fpu
from . import locations as fplocs


# %% Class for updating empowerment

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

            new_vals = fpu.binomial_arr(prob_1)
            changers = (new_vals != ppl[metric]).nonzero()[-1]  # People whose empowerment changes
            ppl.ti_contra_update[changers] = ppl.ti  # Trigger update to contraceptive choices if empowerment changes
            ppl[metric] = new_vals

        return
