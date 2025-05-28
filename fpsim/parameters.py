'''
Handle sim parameters
'''

import numpy as np
import sciris as sc
from . import utils as fpu
from . import defaults as fpd

__all__ = ['Pars', 'pars', 'default_pars']


# %% Pars (parameters) class

def getval(v):
    """ Handle different ways of supplying a value -- number, distribution, function """
    if sc.isnumber(v):
        return v
    elif isinstance(v, dict):
        return fpu.sample(**v)[0]  # [0] since returns an array
    elif callable(v):
        return v()


class Pars(dict):
    '''
    Class to hold a dictionary of parameters, and associated methods.

    Usually not called by the user directly -- use ``fp.pars()`` instead.

    Args:
        pars (dict): dictionary of parameters
    '''
    def __init__(self, pars=None, *args, **kwargs):
        if pars is None:
            pars = {}
        super().__init__(*args, **kwargs)
        self.update(pars)
        return

    def __repr__(self, *args, **kwargs):
        ''' Use odict repr, but with a custom class name and no quotes '''
        return sc.odict.__repr__(self, quote='', numsep='.', classname='fp.Parameters()', *args, **kwargs)

    def copy(self):
        ''' Shortcut for deep copying '''
        return sc.dcp(self)

    def to_dict(self):
        ''' Return parameters as a new dictionary '''
        return {k: v for k, v in self.items()}

    def to_json(self, filename, **kwargs):
        '''
        Export parameters to a JSON file.

        Args:
            filename (str): filename to save to
            kwargs (dict): passed to ``sc.savejson``

        **Example**::
            sim.pars.to_json('my_pars.json')
        '''
        return sc.savejson(filename=filename, obj=self.to_dict(), **kwargs)

    def from_json(self, filename, **kwargs):
        '''
        Import parameters from a JSON file.

        Args:
            filename (str): filename to load from
            kwargs (dict): passed to ``sc.loadjson``

        **Example**::
            sim.pars.from_json('my_pars.json')
        '''
        pars = sc.loadjson(filename=filename, **kwargs)
        self.update(pars)
        return self

    def validate(self, die=True, update=True):
        '''
        Perform validation on the parameters

        Args:
            die (bool): whether to raise an exception if an error is encountered
            update (bool): whether to update the method and age maps
        '''
        # Check that keys are correct
        valid_keys = set(default_pars.keys())
        keys = set(self.keys())
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

        return self


    def _as_ind(self, key, allow_none=True):
        '''
        Take a method key and convert to an int, e.g. 'Condoms' → 7.

        If already an int, do validation.
        '''

        mapping = self['methods']['map']
        keys = list(mapping.keys())

        # Validation
        if key is None and not allow_none:
            errormsg = "No key supplied; did you mean 'None' instead of None?"
            raise ValueError(errormsg)

        # Handle options
        if key in fpd.none_all_keys:
            ind = slice(None)  # This is equivalent to ":" in matrix[:,:]
        elif isinstance(key, str):  # Normal case, convert from key to index
            try:
                ind = mapping[key]
            except KeyError as E:
                errormsg = f'Key "{key}" is not a valid method'
                raise sc.KeyNotFoundError(errormsg) from E
        elif isinstance(key, int):  # Already an int, do nothing
            ind = key
            if ind < len(keys):
                key = keys[ind]
            else:
                errormsg = f'Method index {ind} is out of bounds for methods {sc.strjoin(keys)}'
                raise IndexError(errormsg)
        else:
            errormsg = f'Could not process key of type {type(key)}: must be str or int'
            raise TypeError(errormsg)
        return ind

    def _as_key(self, ind):
        '''
        Convert ind to key, e.g. 7 → 'Condoms'.

        If already a key, do validation.
        '''
        keys = list(self['methods']['map'].keys())
        if isinstance(ind, int):  # Normal case, convert to string
            if ind < len(keys):
                key = keys[ind]
            else:
                errormsg = f'Method index {ind} is out of bounds for methods {sc.strjoin(keys)}'
                raise IndexError(errormsg)
        elif isinstance(ind, str):  # Already a string, do nothing
            key = ind
            if key not in keys:
                errormsg = f'Name "{key}" is not a valid method: choices are {sc.strjoin(keys)}'
                raise sc.KeyNotFoundError(errormsg)
        else:
            errormsg = f'Could not process index of type {type(ind)}: must be int or str'
            raise TypeError(errormsg)
        return key


# %% Parameter creation functions

# Dictionary with all parameters used within an FPsim.
# All parameters that don't vary across geographies are defined explicitly here.
# Keys for all location-specific parameters are also defined here with None values.

default_pars = {
    # Basic parameters
    'location':             None,   # CONTEXT-SPECIFIC ####
    'n_agents':             1_000,  # Number of agents
    'scaled_pop':           None,   # Scaled population / total population size
    'start_year':           1960,   # Start year of simulation
    'end_year':             2020,   # End year of simulation
    'timestep':             1,      # The simulation timestep in months
    'seed':                 1,      # Random seed
    'verbose':              1,      # How much detail to print during the simulation

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
    'interventions':        [],
    'analyzers':            [],

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
}

# Shortcut for accessing default keys
par_keys = default_pars.keys()


def pars(location=None, validate=True, die=True, update=True, **kwargs):
    """
    Function for updating parameters.

    Args:
        location (str): the location to use for the parameters; use 'test' for a simple test set of parameters
        validate (bool): whether to perform validation on the parameters
        die      (bool): whether to raise an exception if validation fails
        update   (bool): whether to update values during validation
        kwargs   (dict): custom parameter values

    **Example**::
        pars = fp.pars(location='senegal')
    """

    from . import locations as fplocs # Here to avoid circular import

    # Set test parameters
    if location == 'test':
        kwargs.setdefault('n_agents', 100)
        kwargs.setdefault('verbose', 0)
        kwargs.setdefault('start_year', 2000)
        kwargs.setdefault('end_year', 2010)

    # Handle location
    location = fpd.get_location(location)  # Handle location

    # Initialize parameter dict, which will be updated with location data
    pars = sc.mergedicts(default_pars, kwargs, _copy=True)  # Merge all pars with kwargs and copy

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

    # TODO: regional locations not supported yet
    # valid_ethiopia_regional_locs = dir(fplocs.ethiopia.regions)
    # if location in valid_ethiopia_regional_locs:
    #     location_pars = getattr(fplocs.ethiopia.regions, location).make_pars(**loc_kwargs)

    # Merge again, so that we ensure the user-defined values overwrite any location defaults
    pars = sc.mergedicts(pars, kwargs, _copy=True)

    # Convert to the class
    pars = Pars(pars)

    # Perform validation
    if validate:
        pars.validate(die=die, update=update)

    return pars

