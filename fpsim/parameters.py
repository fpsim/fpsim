'''
Define defaults for use throughout FPsim
'''

import numpy as np
import sciris as sc

__all__ = ['pars']


class Pars(dict):
    '''
    Lightweight class to hold a dictionary of parameters.

    Args:
        pars (dict): dictionary of parameters
    '''
    def __init__(self, pars, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update(pars)
        return


    def __repr__(self, *args, **kwargs):
        ''' Use odict repr, but with a custom class name and no quotes '''
        return sc.odict.__repr__(self, quote='', numsep='.', classname='fp.Parameters()', *args, **kwargs)


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
            if die: raise ValueError(errormsg)
            else:   print(errormsg)

        # Validate method matrices
        new_method_map = self['methods']['map']
        new_method_age_map = self['methods']['age_map']
        n = len(new_method_map)
        raw = self['methods']['raw']
        age_keys = set(new_method_age_map.keys())
        for mkey in ['annual', 'pp0to1', 'pp1to6']:
            m_age_keys = set(raw[mkey].keys())
            if age_keys != m_age_keys:
                errormsg = f'Matrix "{mkey}" has inconsistent keys: "{sc.strjoin(age_keys)}" ≠ "{sc.strjoin(m_age_keys)}"'
                if die: raise ValueError(errormsg)
                else:   print(errormsg)
        for k in age_keys:
            shape = raw['pp0to1'][k].shape
            if shape != (n,):
                errormsg = f'Postpartum method initiation matrix for ages {k} has unexpected shape: should be ({n},), not {shape}'
                if die: raise ValueError(errormsg)
                else:   print(errormsg)
            for mkey in ['annual', 'pp1to6']:
                shape = raw[mkey][k].shape
                if shape != (n,n):
                    errormsg = f'Method matrix {mkey} for ages {k} has unexpected shape: should be ({n},{n}), not {shape}'
                    if die: raise ValueError(errormsg)
                    else:   print(errormsg)

        # Copy to defaults, making use of mutable objects to preserve original object ID
        if update:
            for k in list(method_map.keys()):     method_map.pop(k) # Remove all items
            for k in list(method_age_map.keys()): method_age_map.pop(k) # Remove all items
            for k,v in new_method_map.items():
                method_map[k] = v
            for k,v in new_method_age_map.items():
                method_age_map[k] = v

        return


    def to_dict(self):
        ''' Return parameters as a new dictionary '''
        return {k:v for k,v in self.items()}


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



def sim_pars():
    ''' Additional parameters used in the sim '''
    sim_pars = dict(
        mortality_probs = {}, # CK: TODO: rethink implementation
        interventions   = [],
        analyzers       = [],
    )
    return sim_pars


def pars(location=None, validate=True, **kwargs):
    '''
    Function for getting default parameters.

    Args:
        location (str): the location to use for the parameters; use 'test' for a simple test set of parameters
        validate (bool): whether to perform validation on the parameters
        kwargs (dict): custom parameter values

    **Example**::
        pars = fp.pars(location='senegal')
    '''
    from . import locations as fplocs # Here to avoid circular import

    if not location:
        location = 'default'

    # Set test parameters
    if location == 'test':
        location = 'default'
        kwargs.setdefault('n_agents', 100)
        kwargs.setdefault('verbose', 0)
        kwargs.setdefault('start_year', 2000)
        kwargs.setdefault('end_year', 2010)

    # Define valid locations
    if location in ['senegal', 'default']:
        pars = fplocs.senegal.make_pars()

    # Else, error
    else:
        errormsg = f'Location "{location}" is not currently supported'
        raise NotImplementedError(errormsg)

    # Merge with sim_pars and kwargs and copy
    pars.update(sim_pars())
    pars = sc.mergedicts(pars, kwargs, _copy=True)

    # Convert to the class
    pars = Pars(pars)

    # Perform validation
    if validate:
        pars.validate()

    return pars


def validate_pars(pars):
    ''' Perform internal validation checks and other housekeeping '''



    return


#%% Global defaults
useSI          = True
mpy            = 12   # Months per year, to avoid magic numbers
eps            = 1e-9 # To avoid divide-by-zero
max_age        = 99   # Maximum age
max_age_preg   = 50   # Maximum age to become pregnant
max_parity     = 20   # Maximum number of children

#%% Defaults when creating a new person
person_defaults = dict(
    uid                  = -1,
    age                  = 0,
    sex                  = 0,
    parity               = 0,
    method               = 0,
    barrier              = 0,
    postpartum_dur       = 0,
    gestation            = 0,
    preg_dur             = 0,
    stillbirth           = 0,
    miscarriage          = 0,
    abortion             = 0,
    remainder_months     = 0,
    breastfeed_dur       = 0,
    breastfeed_dur_total = 0,
    alive                = True,
    pregnant             = False,
    sexually_active      = False,
    sexual_debut         = False,
    sexual_debut_age     = -1,
    first_birth_age      = -1,
    lactating            = False,
    postpartum           = False,
    lam                  = False,
    mothers              = -1,
)

# Postpartum keys to months
postpartum_map = {
    'pp0to5':   [ 0, 6],
    'pp6to11':  [6, 12],
    'pp12to23': [12, 24]
}

## Age bins for tracking age-specific fertility rate
age_bin_map = {
    '10-14':   [10, 15],
    '15-19':   [15, 20],
    '20-24':   [20, 25],
    '25-29':   [25, 30],
    '30-34':   [30, 35],
    '35-39':   [35, 40],
    '40-44':   [40, 45],
    '45-49':   [45, 50]
}

# Age and parity splines
spline_ages      = np.arange(max_age + 1)
spline_preg_ages = np.arange(max_age_preg + 1)
spline_parities  = np.arange(max_parity + 1)

# Define allowable keys to select all (all ages, all methods, etc)
none_all_keys = [None, 'all', ':', [None], ['all'], [':']]

# Definition of contraceptive methods and corresponding numbers -- can be overwritten by locations
method_map = {
    'None'              : 0,
    'Pill'              : 1,
    'IUDs'              : 2,
    'Injectables'       : 3,
    'Condoms'           : 4,
    'BTL'               : 5,
    'Withdrawal'        : 6,
    'Implants'          : 7,
    'Other traditional' : 8,
    'Other modern'      : 9,
}

# Age bins for different method switching matrices -- can be overwritten by locations
method_age_map = {
    '<18':   [ 0, 18],
    '18-20': [18, 20],
    '21-25': [20, 25],
    '>25':   [25, max_age+1], # +1 since we're using < rather than <=
}

# Finally, create default parameters to use for accessing keys etc
default_pars = pars(validate=False) # Do not validate since default parameters are used for validation
par_keys = default_pars.keys()