"""
Methods and functions related to empowerment
We consider four empowerment metrics: paid_employment, decision_health, decision_wages, sexual_autonomy
"""

# %% Imports
import numpy as np
import sciris as sc
from . import utils as fpu
from . import locations as fplocs
from . import defaults as fpd

# %% Class for updating empowerment states and pars (probabilities from csv files)
class Empowerment:
    def __init__(self, location='kenya', seed=None, empowerment_file=None):

        # Handle location
        location = location.lower()
        if location == 'kenya':
            # This dictionary contains the coefficients of the regression model that
            # defines how empowerment proababilities change
            self.empower_update_pars = fplocs.kenya.empowerment_update_pars()
            # This dictionary contains the coefficients of the regression model that
            # defines how intent to use contraception changes e
            self.intent_update_pars = fplocs.kenya.contraception_intent_update_pars()

            # This dictionary contains the default/baseline empowerment
            # probabilities, and loading coefficients for composite measures,
            # as well as ages for which probs are defined.
            self.empowerment_pars = fplocs.kenya.make_empowerment_pars(seed=seed)
        else:
            errormsg = f'Location "{location}" is not currently supported for empowerment analyses'
            raise NotImplementedError(errormsg)

        # List only empowerment metrics that are defined in empowerment.csv
        self.metrics = list(self.empowerment_pars["avail_metrics"])
        # Metrics that will be updated using value from empower_coef.csv
        self.up_metrics = sorted(list(self.empower_update_pars.keys()))
        self.cm_metrics = ["financial_autonomy", "decision_making"]

        return

    @property
    def fa_metrics(self):
        """ Base metric that combined with loadings make up a woman's
        financial autonomy measure"""
        return ["has_savings", "has_fin_knowl", "has_fin_goals"]

    @property
    def dm_metrics(self):
        """ Base metric that combined with loadings make up a woman's decision making
        measure"""
        return ["buy_decision_major", "buy_decision_daily", "buy_decision_clothes", "decision_health"]

    def initialize(self, ppl):
        self.prob2state_init(ppl)
        self.calculate_composite_measures(ppl)
        return

    def prob2state_init(self, ppl):
        """
        Use empowerment probabilities in self.empowerment_pars to set the
        homonymous boolean states in ppl.

        Arguments:
            ppl (fpsim.People object): filtered people object containing only
            alive female agents

        Returns:
            None
        """
        # Fitler female agents that are outside the inclusive range 15-49
        eligible_inds = sc.findinds(ppl.age >= fpd.min_age,
                                    ppl.age <  fpd.max_age_preg)

        # Transform ages to integers
        ages = fpu.digitize_ages_1yr(ppl.age[eligible_inds])
        # Transform to indices of available ages in pars
        data_inds = ages - fpd.min_age

        for empwr_state in self.metrics:
            probs = self.empowerment_pars[empwr_state][data_inds]  # empirical probabilities per age 15-49
            new_vals = fpu.binomial_arr(probs)
            ppl[empwr_state][eligible_inds] = new_vals
        return

    def update_empwr_states(self, ppl):
        """
        Set the homonymous boolean states in ppl. Expects people to be filtered
        by the appropriate conditions.

        Arguments:
            ppl (fpsim.People object): filtered people object containing only
            alive female agents, that are on their bday and within the
            appropriate age range.

        Returns:
            None
        """
        # Transform ages to integers
        ages = fpu.digitize_ages_1yr(ppl.age)
        # Transform to indices of available ages in empowerment_pars
        data_inds = ages - fpd.min_age

        # NOTE: The code below should work but it doesn't -- some clash with, or undefined setitem, somewhere ..
        # for empwr_state in self.metrics:
        #     probs = self.empowerment_pars[empwr_state][data_inds]  # empirical probabilities per age 15-49
        #     new_vals = fpu.binomial_arr(probs)
        #     ppl[empwr_state] = new_vals

        for empwr_state in self.metrics:
            setattr(ppl, empwr_state, fpu.binomial_arr(self.empowerment_pars[empwr_state][data_inds]))

        return

    def update_empwr_states_by_coeffs(self, ppl):
        # Update based on coefficients
        for lhs in self.metrics:
            # lhs -- metric to be updated as a function of rhs variables
            p = self.empower_update_pars[lhs]
            rhs = p.intercept * np.ones(len(ppl[lhs]))

            for predictor, beta_p in p.items():
                 # TODO: update the bit below; iterating over specific attributes
                 #  is a temporary fix because ppl does not have all the
                 #  states in p.items()
                 #  keys in p, not represented in ppl: "wealthquintile", "nsage, knots"
                if predictor in ["on_contra", "paid_employment", "edu_attainment", "parity", "urban"]:
                    rhs += beta_p * ppl[predictor]

            # Logit
            prob_t = 1.0 / (1.0 + np.exp(-rhs))

            if lhs in self.cm_metrics:
                continue
            else:
                # base empowerment states are boolean, we do not currently track probs,
                new_vals = fpu.binomial_arr(prob_t)
            old_vals = ppl[lhs]
            changers = sc.findinds(new_vals != old_vals)  # People whose empowerment changes
            ppl.ti_contra[changers] = ppl.ti  # Trigger update to contraceptive choices if empowerment changes
            setattr(ppl, lhs, new_vals)
        return

    def update_intent_to_use_by_coeffs(self, ppl):
        # Update her intent to use contraception based on coefficients
        lhs = "intent_to_use"
        # lhs -- metric to be updated as a function of rhs variables
        p = self.intent_update_pars[lhs]
        rhs = p.intercept * np.ones(len(ppl[lhs]))

        for predictor, beta_p in p.items():
             # TODO: update the bit below; iterating over specific attributes
             #  is a temporary fix because ppl does not have all the
             #  states in p.items()
             #  keys in p, not represented in ppl: "wealthquintile", "nsage, knots"
            if predictor in ["intent_to_use", "edu_attainment", "parity", "urban"]:
                rhs += beta_p * ppl[predictor]

        # Handle predictors based on fertility intent
        beta_p = p["fertility_intentno"] * np.ones(len(ppl[lhs]))
        beta_p[sc.findinds(ppl["fertility_intent"] == True)] = p["fertility_intentyes"]
        rhs += beta_p*ppl["fertility_intent"]

        # Logit
        prob_t = 1.0 / (1.0 + np.exp(-rhs))

        # base empowerment states are boolean, we do not currently track probs
        new_vals = fpu.binomial_arr(prob_t)
        old_vals = ppl[lhs]
        changers = sc.findinds(new_vals != old_vals)  # People whose empowerment changes
        ppl.ti_contra[changers] = ppl.ti  # Trigger update to contraceptive choices if empowerment changes
        setattr(ppl, lhs, new_vals)
        return

    def update(self, ppl):
        """ Update empowerment states and intent to use based on regression coefficients"""
        self.update_empwr_states_by_coeffs(ppl)
        self.update_intent_to_use_by_coeffs(ppl)
        self.calculate_composite_measures(ppl)
        return

    def calculate_composite_measures(self, ppl):
        temp_fa = 0.0
        temp_dm = 0.0
        for metric in self.fa_metrics:
            temp_fa += ppl[metric] * self.empowerment_pars["loadings"][metric]
        for metric in self.dm_metrics:
            temp_dm += ppl[metric] * self.empowerment_pars["loadings"][metric]

        ppl.financial_autonomy = temp_fa
        ppl.decision_making = temp_dm
        return
