"""
Methods and functions related to subnational dynamics
"""

# %% Imports
import numpy as np  # Needed for a few things not provided by pl
import sciris as sc
import pandas as pd
from . import utils as fpu
from . import defaults as fpd


# %% Initialization methods
def init_regional_states(ppl):
    """
    If ppl.pars['use_subnational'] == True, location-specific data
    are expected to exist to populate regional states/attributes.

    If If ppl.pars['use_subnational'] == False, related
    attributes will be initialized with values found in defaults.py.
    """

    # Initialize subnational-related attributes with location-specific data
    # Any other parameters that are region-specific will be initialized from here
    ppl.region = get_region_init_vals(ppl)

    return


def get_region_init_vals(ppl):
    """Get initial distribution of regions"""

    region_dict = ppl.pars['region']
    n = len(ppl)

    # Initialise individual region
    # Region dictionary
    region = np.zeros(n, dtype=str)

    if region_dict is not None:
        # Set distribution for individuals based on regional proportions
        region_names = region_dict['region']
        region_probs = region_dict['mean']
        region = np.random.choice(region_names, size=n, p=region_probs)

    return region


def get_debut_init_vals(ppl):
    """Get initial distribution of fated debut based on regional country data"""

    n = len(ppl)
    debut_age_dict = ppl.pars['debut_age_region']
    debut_age_by_region = pd.DataFrame({'region': debut_age_dict['region'],
                                        'age': debut_age_dict['age'],
                                        'prob': debut_age_dict['prob']})
    debut_age_values = np.zeros(n, dtype=int)
    for r in debut_age_by_region['region'].unique():
        # Find indices in region array
        f_inds = sc.findinds(ppl.region == r)
        region_debut_ages = debut_age_by_region.loc[debut_age_by_region['region'] == r, 'age'].values
        region_debut_probs = debut_age_by_region.loc[debut_age_by_region['region'] == r, 'prob'].values
        debut_age_dist = region_debut_ages[fpu.n_multinomial(region_debut_probs, len(f_inds))]
        debut_age_values[f_inds] = debut_age_dist

    return debut_age_values


def get_urban_init_vals(ppl, urban_prop=None):
    """Get initial distribution of urban based on regional country data"""

    n = len(ppl)
    urban = np.ones(n, dtype=bool)
    region_dict = ppl.pars['region']
    urban_by_region = pd.DataFrame({'region': region_dict['region'], 'urban': region_dict['urban']})
    # For each region defined in region.csv, assign a regional distribution of urban/rural population
    for r in urban_by_region['region']:
        # Find indices in region array
        f_inds = sc.findinds(ppl.region==r)
        region_urban_prop = urban_by_region.loc[urban_by_region['region']==r, 'urban'].values[0]
        urban_values = np.random.choice([True, False], size=len(f_inds), p=[region_urban_prop, 1-region_urban_prop])
        urban[f_inds] = urban_values

    return urban


"""
def initialize_lam_region(self, n, lam_region):
    lam_region_dict = self.pars['lactational_amenorrhea_region']
    lam_by_region = pd.DataFrame({'region': lam_region_dict['region'],
                               'month': lam_region_dict['month'],
                               'rate': lam_region_dict['rate']})
    lam_values = np.zeros(n, dtype=bool)
    for r in lam_by_region['region'].unique():
        # Find indices in region array
        f_inds = sc.findinds(lam_region == r)
        for month in lam_by_region[lam_by_region['region'] == r]['month'].unique(): 
            month_data = lam_by_region[(lam_by_region['region'] == r) & (lam_by_region['month'] == month)]
            lam_values[f_inds] = np.random.choice([True, False], size=len(f_inds), p=month_data['rate'].values)
    return lam_values
"""
