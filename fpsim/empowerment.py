"""
Methods and functions related to empowerment
We consider four empowerment metrics: paid_employment, decision_health, decision_wages, sexual_autonomy
"""

# %% Imports
import numpy as np
import pandas as pd
import sciris as sc
from . import utils as fpu
from . import locations as fplocs
from . import defaults as fpd

empow_path = sc.thispath(__file__)

# %% Class for updating empowerment states and pars (probabilities from csv files)
class Empowerment:
    def __init__(self, pars=None, location='kenya', seed=None):
        default_pars = dict(
            age_bins=[15, 20, 25, 30, 35, 40, 45, 50],
            age_weights=None,
            nbins=None
        )
        self.pars = sc.mergedicts(default_pars, pars)
        self.pars['nbins'] = len(self.pars['age_bins'])-1
        if self.pars['age_weights'] is None:
            self.pars['age_weights'] = np.zeros(self.pars['nbins'])

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

            # Store the age spline
            self.age_spline = fplocs.kenya.empowerment_age_spline()
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

    def get_longitud_data(self, ppl, term, ti, tiperyear):
        """
        Uses the ppl 'longitude' parameter to extract the term data for the respective people from one year prior

        Arguments:
            ppl (fpsim.People object): filtered people object containing people for whom we're extracting previous year data
            term (str): attribute from ppl object that we are extracting
            ti (int): current timestep
            tiperyear (int): timesteps per year (in current simulation)

        Returns:
            data (np.arr):  array of the ppl.term values from one year prior to current timestep
        """
        # Calculate correct index for data 1 year prior
        if len(ppl) > 0:
            year_ago_index = (ti+1) % tiperyear
            data = ppl['longitude'][term][ppl.inds, year_ago_index]
        else:
            data = np.empty((0,))

        return data

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

    def update_empwr_states_by_coeffs(self, ppl, ti, tiperyear):
        # Add age spline setup
        int_age = ppl.int_age
        int_age[int_age < fpd.min_age] = fpd.min_age
        int_age[int_age >= fpd.max_age_preg] = fpd.max_age_preg - 1
        dfa = self.age_spline.loc[int_age]

        # Update based on coefficients
        for lhs in self.metrics:
            # If lhs empowerment metric not in the lhs variables from empower_coef.csv
            if lhs not in self.empower_update_pars.keys() and f'{lhs}_prev' not in self.empower_update_pars.keys():
                continue

            # Extract coefficients and initialize rhs with intercept
            p = self.empower_update_pars[lhs]
            rhs = p.intercept * np.ones(len(ppl[lhs]))

            for predictor, beta_p in p.items():
                if predictor == 'intercept':
                    continue
                if predictor.endswith('_prev'):     # For longitudinal predictors
                    rhs += beta_p * self.get_longitud_data(ppl, predictor.removesuffix('_prev'), ti, tiperyear)
                elif 'knots' in predictor:          # For age predictors
                    knot = predictor[-1]
                    rhs += beta_p * dfa[f'knot_{knot}'].values
                else:
                    rhs += beta_p * ppl[predictor]

            # Age-based weights
            age_group = fpu.digitize_ages(ppl.age, self.pars['age_bins'])  # Finds the age-group index to get the correct age weight
            # Get age weights
            age_weights = self.pars['age_weights'][age_group]
            # TODO: check with Marita whether directly adding a weight to the rhs is ok
            rhs += age_weights

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
            if predictor in ["intent_to_use", "edu_attainment", "parity", "urban", "wealthquintile"]:
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

    def update(self, ppl, ti, tiperyear):
        """ Update empowerment states and intent to use based on regression coefficients"""
        self.update_empwr_states_by_coeffs(ppl, ti, tiperyear)
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
