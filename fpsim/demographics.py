"""
Methods and functions related to basic demographics such as urban and
age at first partnership.
"""

# %% Imports
import numpy as np  # Needed for a few things not provided by pl
import sciris as sc
from . import utils as fpu


# %% Initialization methods
def init_urban_states(ppl):
    """Demographics on whether a person lives in a rural or urban setting"""
    # Get init vals and populate state in one step
    ppl.urban = get_urban_init_vals(ppl)


def init_partnership_states(ppl):
    """Demographics on whether a person is in a partnership, and their expected age at first partnership in a rural or urban setting"""

    # Get init values for these sociodemographic states
    partnered, partnership_age = get_partnership_init_vals(ppl)

    # Populate states
    ppl.partnered = partnered
    ppl.partnership_age = partnership_age


def get_urban_init_vals(ppl, urban_prop=None):
    """ Get initial distribution of urban """

    n = len(ppl)
    urban = np.ones(n, dtype=bool)

    if urban_prop is None:
        if ppl.pars['urban_prop'] is not None:
            urban_prop = ppl.pars['urban_prop']

    if urban_prop is not None:
        urban = fpu.n_binomial(urban_prop, n)

    return urban


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
