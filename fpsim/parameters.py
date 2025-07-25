'''
Handle sim parameters
'''

import sciris as sc
from . import defaults as fpd

__all__ = ['pars', 'validate', 'pars_to_json', 'pars_from_json', 'default_pars', 'default_sim_pars']


def validate(valid_pars, to_validate_pars, die=True):
    '''
    Perform validation on the parameters

    Args:
        die (bool): whether to raise an exception if an error is encountered
        update (bool): whether to update the method and age maps
    '''
    # Check that keys are correct
    valid_keys = set(valid_pars.keys())
    keys = set(to_validate_pars.keys())
    if keys != valid_keys:
        diff1 = valid_keys - keys
        diff2 = keys - valid_keys
        errormsg = ''
        if diff1:
            errormsg += 'The parameter set is not valid since the following keys are missing:\n'
            errormsg += f'{sc.strjoin(diff1)}\n'
        if diff2:
            errormsg += 'The parameter set is not valid since the following keys are not recognized:\n'
            errormsg += f'{sc.strjoin(diff2)}\n'
        if die:
            raise ValueError(errormsg)
        else:
            print(errormsg)

    return

def pars_to_json(pars, filename):
    # write the parameters to a json file
    sc.savejson(filename, pars)

def pars_from_json(filename):
    # load the parameters from a json file
    return sc.loadjson(filename)

# %% Parameter creation functions

# Dictionary with all parameters used within an FPsim.
# All parameters that don't vary across geographies are defined explicitly here.
# Keys for all location-specific parameters are also defined here with None values.

default_sim_pars = {
    'n_agents':             1_000,  # Number of agents
    'pop_scale':            None,   # Scaled population / total population size
    'start':                1960,   # Start year of simulation
    'stop':                 2020,   # End year of simulation
    'dt':                   1/12,      # The simulation timestep in 'unit's
    'unit':                 'year',   # The unit of time for the simulation
    'rand_seed':            1,      # Random seed
    'verbose':              1/12,   # Verbosity level
    'use_aging':            True,   # Whether to age the population
    'interventions':        None,   # Interventions to apply
    'analyzers':            None,   # Analyzers to apply
    'connectors':           None,   # Connectors to apply
    'people':               None,

}

default_pars = {
    # Basic parameters
    'location':             None,   # CONTEXT-SPECIFIC ####
    'contraception_module': None,
    'education_module':   None,
    'empowerment_module': None,

    # Settings - what aspects are being modeled - TODO, remove
    'use_partnership':      0,      #

    # Age limits (in years)
    'method_age':           15,
    'age_limit_fecundity':  50,
    'max_age':              99,

    # Durations (in months)
    'end_first_tri':        3,      # Months
    'preg_dur_low':         9,      # Months
    'preg_dur_high':        9,      # Months
    'max_lam_dur':          5,      # Duration of lactational amenorrhea (months)
    'short_int':            24,     # Duration of a short birth interval between live births (months)
    'low_age_short_int':    0,      # age limit for tracking the age-specific short birth interval
    'high_age_short_int':   20,     # age limit for tracking the age-specific short birth interval
    'postpartum_dur':       35,     # Months
    'breastfeeding_dur_mean': None,   # CONTEXT-SPECIFIC #### - Parameter of truncated norm distribution
    'breastfeeding_dur_sd': None,  # CONTEXT-SPECIFIC #### - Parameter of truncated norm distribution

    # Pregnancy outcomes
    'abortion_prob':        None,   # CONTEXT-SPECIFIC ####
    'twins_prob':           None,   # CONTEXT-SPECIFIC ####
    'LAM_efficacy':         0.98,   # From Cochrane review: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6823189/
    'maternal_mortality_factor': 1,

    # Fecundity and exposure
    'fecundity_var_low':    0.7,
    'fecundity_var_high':   1.1,
    'primary_infertility':  0.05,
    'exposure_factor':      1.0,    # Overall exposure correction factor

    # Other sim parameters
    'mortality_probs':      {},

    ###################################
    # Context-specific data-derived parameters, all defined within location files
    ###################################
    'filenames':            None,
    'age_pyramid':          None,
    'age_mortality':        None,
    'maternal_mortality':   None,
    'infant_mortality':     None,
    'miscarriage_rates':    None,
    'stillbirth_rate':      None,
    'age_fecundity':        None,
    'fecundity_ratio_nullip': None,
    'lactational_amenorrhea': None,
    'sexual_activity':      None,
    'sexual_activity_pp':   None,
    'debut_age':            None,
    'exposure_age':         None,
    'exposure_parity':      None,
    'spacing_pref':         None,
    'barriers':             None,
    'urban_prop':           None,
    'wealth_quintile':      None,
    'age_partnership':      None,
    'mcpr':                 None,

    # Newer parameters, associated with empowerment, but that are not empowerment metrics
    # NOTE, these will be None unless running analyses from the kenya_empowerment repo
    'fertility_intent':     None,
    'intent_to_use':        None,

    'region':               None,
    'track_children':   False,  # Whether to track children
    'regional':         None,
}


def pars(location=None, rand_seed=None, **kwargs):
    '''
    Create a parameter set for a given location

    Args:
        location (str): the location to use
        kwargs (dict): additional parameters to update

    Returns:
        pars (dict): the parameter set
    '''
    pars = sc.dcp(default_pars)

    # Handle location
    if location is None and 'location' in kwargs:
        location = kwargs.pop('location')
    location = fpd.get_location(location, printmsg=True)  # Handle location
    pars['location'] = location

    if rand_seed is None:
        rand_seed = default_sim_pars['rand_seed']

    """
    # Pull out values needed for the location-specific make_pars functions
    loc_kwargs = dict(seed=pars['seed'])

    # Use external registry for locations first
    if location in fpd.location_registry:
        location_module = fpd.location_registry[location]
        location_pars = location_module.make_pars(**loc_kwargs)
    elif hasattr(fplocs, location):
        location_pars = getattr(fplocs, location).make_pars(**loc_kwargs)
    else:
        raise NotImplementedError(f'Could not find location function for "{location}"')

    pars = sc.mergedicts(pars, location_pars)
    """
    # Load the location-specific parameters
    from . import locations as fplocs
    loc_kwargs = dict(seed=rand_seed)

    # Use external registry for locations first
    if location in fpd.location_registry:
        location_module = fpd.location_registry[location]
        location_pars = location_module.make_pars(**loc_kwargs)
    elif hasattr(fplocs, location):
        location_pars = getattr(fplocs, location).make_pars(**loc_kwargs)
    else:
        raise NotImplementedError(f'Could not find location function for "{location}"')

    pars.update(sc.dcp(location_pars))
    pars.update(**kwargs)

    validate(default_pars, pars)

    return pars

# Shortcut for accessing default keys
par_keys = default_pars.keys()
sim_par_keys = default_sim_pars.keys()
