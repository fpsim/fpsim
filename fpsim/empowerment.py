"""
Methods and functions related to empowerment
We consider four empowerment metrics: paid_employment, decision_health, decision_wages, sexual_autonomy
"""

# %% Imports
import numpy as np
import sciris as sc
import pandas as pd
from . import utils as fpu
from . import defaults as fpd


# %% Class for updating empowerment

class Empowerment:
    def __init__(self, empowerment_file):
        self.pars = self.process_empowerment_pars(empowerment_file)
        self.metrics = list(self.pars.keys())
        return

    @staticmethod
    def process_empowerment_pars(empowerment_file):
        raw_pars = pd.read_csv(empowerment_file)
        pars = sc.objdict()
        metrics = raw_pars.lhs.unique()
        for metric in metrics:
            pars[metric] = sc.objdict()
            thisdf = raw_pars.loc[raw_pars.lhs == metric]
            for var_dict in thisdf.to_dict('records'):
                var_name = var_dict['rhs'].replace('_0', '').replace('(', '').replace(')', '').lower()
                if var_name == 'contraception': var_name = 'on_contra'
                pars[metric][var_name] = var_dict['Estimate']
        return pars

    @staticmethod
    def initialize(ppl):
        """
        Initialize by setting values for people
        TODO: I think all the empowerment pars should live here...
        """
        empowerment_dict = ppl.pars['empowerment']
        # NOTE: we assume that either probabilities or metrics in empowerment_dict are defined over all possible ages
        # from 0 to 100 years old.
        n = len(ppl)

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
        # try:
        ppl.sexual_autonomy[age_inds] = empowerment_dict['sexual_autonomy'][age_inds]
        # except:
        #     import traceback; traceback.print_exc(); import pdb; pdb.set_trace()
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
            ppl[metric] = fpu.binomial_arr(prob_1)

        return
