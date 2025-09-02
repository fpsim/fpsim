"""
Methods and functions related to basic demographics such as urban and
age at first partnership.
"""

# %% Imports
import numpy as np  # Needed for a few things not provided by pl
import sciris as sc
import starsim as ss
import fpsim as fp


# %% Deaths module
class Deaths(ss.Deaths):
    """
    Subclass of ss.Deaths to handle FPsim's specific mortality implementation
    Use of this is optional, and FPsim should run with the base class as well.
    """
    def make_p_death(self):

        self.pars['mortality_probs'] = {}

        ind = sc.findnearest(self.pars['age_mortality']['year'], self.t.now('year'))
        val = self.pars.fp['age_mortality']['probs'][ind]
        self.pars['mortality_probs']['gen_trend'] = val

        sim = self.sim
        ppl = sim.people
        uids = ppl.auids  # Get the UIDs of all alive people
        death_rate = np.empty(uids.shape, dtype=ss.dtypes.float)

        trend_val = self.pars['mortality_probs']['gen_trend']
        age_mort = self.pars['age_mortality']
        f_spline = age_mort['f_spline'] * trend_val
        m_spline = age_mort['m_spline'] * trend_val
        over_one = ppl.age[uids] >= 1
        female = uids[over_one & ppl.female[uids]]
        male = uids[over_one & ppl.male[uids]]
        f_ages = ppl.int_age(female)
        m_ages = ppl.int_age(male)

        death_rate[male] = m_spline[m_ages]
        death_rate[female] = f_spline[f_ages]
        death_rate *= self.pars.rate_units * self.pars.rel_death

        # Scale from rate to probability
        death_rate = ss.peryear(death_rate)
        p_death = death_rate.to_prob(self.t.dt)  # Convert to probability per timestep
        return p_death


# %% Initialization methods
def init_partnership_states(ppl):
    """
    Initialize partnership status and expected age at first partnership by rural/urban status
    """

    # Get init values for these sociodemographic states
    partnered, partnership_age = get_partnership_init_vals(ppl)

    # Populate states
    ppl.partnered = partnered
    ppl.partnership_age = partnership_age


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
