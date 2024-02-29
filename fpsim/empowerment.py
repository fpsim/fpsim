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


# %% Initialization methods

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

    # Empowerment dictionary
    empowerment = {}
    empowerment['paid_employment'] = np.zeros(n, dtype=bool)
    empowerment['sexual_autonomy'] = np.zeros(n, dtype=bool)
    empowerment['decision_wages']  = np.zeros(n, dtype=bool)
    empowerment['decision_health'] = np.zeros(n, dtype=bool)

    # Get female agents indices and ages
    f_inds = sc.findinds(ppl.is_female)
    f_ages = ppl.age[f_inds]

    # Create age bins
    age_cutoffs = np.hstack((empowerment_dict['age'], empowerment_dict['age'].max() + 1))
    age_inds = np.digitize(f_ages, age_cutoffs) - 1

    # Paid employment
    paid_employment_probs = empowerment_dict['paid_employment']
    empowerment['paid_employment'][f_inds] = fpu.binomial_arr(paid_employment_probs[age_inds])

    # Make other metrics
    for metric in ['decision_wages', 'decision_health', 'sexual_autonomy']:
        empowerment[metric][f_inds] = empowerment_dict[metric][age_inds]

    return empowerment


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


