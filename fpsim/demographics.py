"""
Methods and functions related to basic demographics such as urban and
age at first partnership.
"""

# %% Imports
import numpy as np  # Needed for a few things not provided by pl
import sciris as sc
import starsim as ss
import fpsim as fp


# %% Initialization methods
class Deaths(ss.Deaths):
    """Subclass of Starsim Deaths to handle deaths"""

    def __init__(self, location=None, **kwargs):
        super().__init__()
        # Read data
        location_module = fp.get_location_module(location)
        death_pars = location_module.make_death_pars()
        self.define_pars(**default_pars)
        self.update_pars(pars, **kwargs)

        return


# %% Initialization methods
def init_partnership_states(ppl):
    """Demographics on whether a person is in a partnership, and their expected age at first partnership in a rural or urban setting"""

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
