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
            # TODO:PSL: consolidate these two dictionaries.
            # This dictionary contains the coefficients of the model that
            # defines how empowerment proababilities change
            self.update_pars = fplocs.kenya.empowerment_update_pars()

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
        self.up_metrics = sorted(list(self.update_pars.keys()))
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

    def prob2state(self, ppl):
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
        # Transform to indices of available ages in pars
        data_inds = ages - fpd.min_age

        # for empwr_state in self.metrics:
        #     probs = self.empowerment_pars[empwr_state][data_inds]  # empirical probabilities per age 15-49
        #     new_vals = fpu.binomial_arr(probs)
        #     ppl[empwr_state] = new_vals
        ppl.paid_employment = fpu.binomial_arr(self.empowerment_pars["paid_employment"][data_inds])
        return

    def update(self, ppl):
        """ Update empowerment probs and re-calculate empowerment states"""
        self.prob2state(ppl)

        # NOTE:  Update annually on her birthday, ie update empowerment by age,
        # This will set empowerment state for women who where < 15 at the start of
        # the simulation and for women (agents) who are botn within the simulation
        # birthdays = ppl.birthday_filter()
        # if len(birthdays):
        #     breakpoint()
        # self.prob2state(ppl)
        #
        # ppl = birthdays.unfilter()

        # for mi, metric in enumerate(self.up_metrics):
        #     p = self.update_pars[metric]
        #     rhs = p.intercept * np.ones(len(ppl[metric]))
        #
        #     for vname, vval in p.items():
        #          # TODO: update the bit below; this is a temporary fix because ppl does not have all the
        #          # states in p.items()
        #          # keys in p, not represented in ppl: "wealthquintile", "nsage, knots"
        #         if vname in ["on_contra", "paid_employment", "edu_attainment", "parity", "urban"]:
        #             rhs += vval * ppl[vname]
        #
        #     prob_1 = 1 / (1+np.exp(-rhs))
        #     if metric in self.cm_metrics:  #not probabilities
        #         new_vals = prob_1
        #     else:  # probabilities
        #         # base empowerment states are boolean, we do not currently track probs,
        #         new_vals = fpu.binomial_arr(prob_1)
        #     changers = sc.findinds(new_vals != ppl[metric])  # People whose empowerment changes
        #     ppl.ti_contra[changers] = ppl.ti  # Trigger update to contraceptive choices if empowerment changes
        #     ppl[metric] = new_vals

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
