'''
Define defaults for use throughout FPsim
'''

import numpy as np
import sciris as sc
from . import utils as fpu

__all__ = ['pars']



def getval(v):
    ''' Handle different ways of supplying a value -- number, distribution, function '''
    if sc.isnumber(v):
        return v
    elif isinstance(v, dict):
        return fpu.sample(**v)[0] # [0] since returns an array
    elif callable(v):
        return v()


class Pars(dict):
    '''
    Lightweight class to hold a dictionary of parameters.

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


    def key2ind(self, key, allow_none=True):
        """
        Take a method key and convert to an int, e.g. 'Condoms' → 7
        """
        ind = key
        if key is None and not allow_none:
            errormsg = "No key supplied; did you mean 'None' instead of None?"
            raise ValueError(errormsg)
        if ind in none_all_keys:
            ind = slice(None) # This is equivalent to ":" in matrix[:,:]
        elif isinstance(ind, str):
            ind = self['methods']['map'][key]
        return ind


    def update_method_eff(self, method, value=None, verbose=False):
        '''
        Update efficacy of one or more contraceptive methods.

        Args:
            method (str/dict): method to update, or dict of method:value pairs
            value (float): new value of contraceptive efficacy (not required if method is a dict)

        **Examples**::
            pars.update_method_eff('Injectables', 0.99)
            pars.update_method_eff({'Injectables':0.99, 'Condoms':0.50})
        '''

        # Validation
        if not isinstance(method, dict):
            if value is None:
                errormsg = 'Must supply a value to update the contraceptive efficacy to'
                raise ValueError(errormsg)
            else:
                method = {method:value}

        # Perform updates
        for k,rawval in method.items():
            try:
                self.key2ind(k)
            except:
                errormsg = f'Key "{k}" is not a valid method: are you sure this is an efficacy change?'
                raise ValueError(errormsg)
            v = getval(rawval)
            ind = self.key2ind(k)
            orig = self['method_efficacy'][ind]
            self['method_efficacy'][ind] = v
            if verbose:
                print(f'Efficacy for method {k} was changed from {orig:0.3f} to {v:0.3f}')

        return


    def update_method_prob(self, source=None, dest=None, factor=None, value=None, ages=None, matrix=None):
        '''
        Updates the probability matrices with a new value. Usually used via the
        intervention ``fp.update_methods()``.

        Args:
            source (str/int): the method to switch from
            dest   (str/int): the method to switch to
            factor (float):   if supplied, multiply the probability by this factor
            value (float):    if supplied, change the probability to this value
            ages (str/list):  the ages to modify (default: all)
            matrix (str):     which switching matrix to modify (default: annual)
        '''

        raw = self['methods']['raw'] # We adjust the raw matrices, so the effects are persistent

        # Convert from strings to indices
        source = self.key2ind(source, allow_none=False)
        dest   = self.key2ind(dest, allow_none=False)

        # Replace age keys with all ages if so asked
        if ages in none_all_keys:
            ages = raw['annual'].keys()
        else:
            ages = sc.tolist(ages)

        # Check matrix is valid
        if matrix not in raw:
            errormsg = f'Invalid matrix "{matrix}"; valid choices are: {sc.strjoin(raw.keys())}'
            raise sc.KeyNotFoundError(errormsg)

        # Actually loop over the matrices and apply the changes
        for k in ages:
            arr = raw[matrix][k]
            if matrix == 'pp0to1': # Handle the postpartum initialization *vector*
               orig = arr[dest]
               if factor is not None:
                   arr[dest] *= getval(factor)
               elif value is not None:
                   val = getval(value)
                   arr[dest] = 0
                   arr *= (1-val)/arr.sum()
                   arr[dest] = val
                   assert np.isclose(arr.sum(), 1, atol=1e-3), f'Matrix should sum to 1, not {arr.sum()}'
               if self.verbose:
                   print(f'Matrix {matrix} for age group {k} was changed from:\n{orig}\nto\n{arr[dest]}')
            else: # Handle annual switching *matrices*
                orig = arr[source, dest]
                if factor is not None:
                    arr[source, dest] *= getval(factor)
                elif value is not None:
                    val = getval(value)
                    arr[source, dest] = 0
                    arr[source, :] *= (1-val)/arr[source, :].sum()
                    arr[source, dest] = val
                    assert np.isclose(arr[source, :].sum(), 1, atol=1e-3), f'Matrix should sum to 1, not {arr.sum()}'
                if self.verbose:
                    print(f'Matrix {matrix} for age group {k} was changed from:\n{orig}\nto\n{arr[source, dest]}')

        return


    def add_method(self, name, pos=None):
        '''
        Add a new contraceptive method to the switching matrices.

        Methods must be added before the sim is run.

        Note: the matrices are stored in ``pars['methods']['raw']``; this method
        is a helper function for modifying those. For more flexibility, modify
        them directly. The ``fp.update_methods()`` intervention can be used to
        modify the switching probabilities later.

        Args:
            name (str): the name of the new method
            pos (int): where in the matrix to insert the new method (default: end)

        **Examples**::
            pars = fp.pars()
            pars.add_method('New method') # Create a new method with no initiation/discontinuation
            pars.add_method(name='Male pill', pos=5)
        '''

        return


    def rm_method(self, name):
        '''
       Removes a contraceptive method from the switching matrices.

        Methods must be removed before the sim is run.

        Args:
            name (str/ind): the name or index of the method to remove

        **Example**::
            pars = fp.pars()
            pars.rm_method('Other modern')
        '''

        return


    def reorder_methods(self, order):
        '''
       Reorder the contraceptive method matrices.

        Methods must be reordered before the sim is run.

        Args:
            order (arr): the new order of methods

        **Example**::
            pars = fp.pars()
            pars.reorder_methods([2, 6, 4, 7, 0, 8, 5, 1, 3])
        '''


        return




def sim_pars():
    ''' Additional parameters used in the sim '''
    sim_pars = dict(
        mortality_probs = {}, # CK: TODO: rethink implementation
        interventions   = [],
        analyzers       = [],
    )
    return sim_pars


def pars(location=None, validate=True, die=True, update=True, **kwargs):
    '''
    Function for getting default parameters.

    Args:
        location (str): the location to use for the parameters; use 'test' for a simple test set of parameters
        validate (bool): whether to perform validation on the parameters
        die      (bool): whether to raise an exception if validation fails
        update   (bool): whether to update values during validation
        kwargs   (dict): custom parameter values

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
        pars.validate(die=die, update=update)

    return pars



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