'''
Specify the core interventions available in FPsim. Other interventions can be
defined by the user by inheriting from these classes.
'''

import numpy as np
import pylab as pl
import sciris as sc
import inspect
from . import utils as fpu
from . import defaults as fpd


#%% Generic intervention classes

__all__ = ['Intervention', 'change_par', 'update_methods']


class Intervention:
    '''
    Base class for interventions. By default, interventions are printed using a
    dict format, which they can be recreated from. To display all the attributes
    of the intervention, use disp() instead.

    To retrieve a particular intervention from a sim, use sim.get_intervention().

    Args:
        label       (str): a label for the intervention (used for plotting, and for ease of identification)
        show_label (bool): whether or not to include the label in the legend
        do_plot    (bool): whether or not to plot the intervention
        line_args  (dict): arguments passed to pl.axvline() when plotting
    '''
    def __init__(self, label=None, show_label=False, do_plot=None, line_args=None):
        self._store_args() # Store the input arguments so the intervention can be recreated
        if label is None: label = self.__class__.__name__ # Use the class name if no label is supplied
        self.label = label # e.g. "Close schools"
        self.show_label = show_label # Do not show the label by default
        self.do_plot = do_plot if do_plot is not None else True # Plot the intervention, including if None
        self.line_args = sc.mergedicts(dict(linestyle='--', c='#aaa', lw=1.0), line_args) # Do not set alpha by default due to the issue of overlapping interventions
        self.days = [] # The start and end days of the intervention
        self.initialized = False # Whether or not it has been initialized
        self.finalized = False # Whether or not it has been initialized
        return


    def __repr__(self, jsonify=False):
        ''' Return a JSON-friendly output if possible, else revert to short repr '''

        if self.__class__.__name__ in __all__ or jsonify:
            try:
                json = self.to_json()
                which = json['which']
                pars = json['pars']
                parstr = ', '.join([f'{k}={v}' for k,v in pars.items()])
                output = f"cv.{which}({parstr})"
            except Exception as E:
                output = type(self) + f' (error: {str(E)})' # If that fails, print why
            return output
        else:
            return f'{self.__module__}.{self.__class__.__name__}()'


    def disp(self):
        ''' Print a detailed representation of the intervention '''
        return sc.pr(self)


    def _store_args(self):
        ''' Store the user-supplied arguments for later use in to_json '''
        f0 = inspect.currentframe() # This "frame", i.e. Intervention.__init__()
        f1 = inspect.getouterframes(f0) # The list of outer frames
        parent = f1[2].frame # The parent frame, e.g. change_beta.__init__()
        _,_,_,values = inspect.getargvalues(parent) # Get the values of the arguments
        if values:
            self.input_args = {}
            for key,value in values.items():
                if key == 'kwargs': # Store additional kwargs directly
                    for k2,v2 in value.items():
                        self.input_args[k2] = v2 # These are already a dict
                elif key not in ['self', '__class__']: # Everything else, but skip these
                    self.input_args[key] = value
        return


    def initialize(self, sim=None):
        '''
        Initialize intervention -- this is used to make modifications to the intervention
        that can't be done until after the sim is created.
        '''
        self.initialized = True
        self.finalized = False
        return


    def finalize(self, sim=None):
        '''
        Finalize intervention

        This method is run once as part of `sim.finalize()` enabling the intervention to perform any
        final operations after the simulation is complete (e.g. rescaling)
        '''
        if self.finalized:
            raise RuntimeError('Intervention already finalized')  # Raise an error because finalizing multiple times has a high probability of producing incorrect results e.g. applying rescale factors twice
        self.finalized = True
        return


    def apply(self, sim):
        '''
        Apply the intervention. This is the core method which each derived intervention
        class must implement. This method gets called at each timestep and can make
        arbitrary changes to the Sim object, as well as storing or modifying the
        state of the intervention.

        Args:
            sim: the Sim instance

        Returns:
            None
        '''
        raise NotImplementedError


    def plot_intervention(self, sim, ax=None, **kwargs):
        '''
        Plot the intervention

        This can be used to do things like add vertical lines on days when
        interventions take place. Can be disabled by setting self.do_plot=False.

        Note 1: you can modify the plotting style via the ``line_args`` argument when
        creating the intervention.

        Note 2: By default, the intervention is plotted at the days stored in self.days.
        However, if there is a self.plot_days attribute, this will be used instead.

        Args:
            sim: the Sim instance
            ax: the axis instance
            kwargs: passed to ax.axvline()

        Returns:
            None
        '''
        line_args = sc.mergedicts(self.line_args, kwargs)
        if self.do_plot or self.do_plot is None:
            if ax is None:
                ax = pl.gca()
            if hasattr(self, 'plot_days'):
                days = self.plot_days
            else:
                days = self.days
            if sc.isiterable(days):
                label_shown = False # Don't show the label more than once
                for day in days:
                    if sc.isnumber(day):
                        if self.show_label and not label_shown: # Choose whether to include the label in the legend
                            label = self.label
                            label_shown = True
                        else:
                            label = None
                        ax.axvline(day, label=label, **line_args)
        return


    def to_json(self):
        '''
        Return JSON-compatible representation

        Custom classes can't be directly represented in JSON. This method is a
        one-way export to produce a JSON-compatible representation of the
        intervention. In the first instance, the object dict will be returned.
        However, if an intervention itself contains non-standard variables as
        attributes, then its `to_json` method will need to handle those.

        Note that simply printing an intervention will usually return a representation
        that can be used to recreate it.

        Returns:
            JSON-serializable representation (typically a dict, but could be anything else)
        '''
        which = self.__class__.__name__
        pars = sc.jsonify(self.input_args)
        output = dict(which=which, pars=pars)
        return output


class change_par(Intervention):
    '''
    Change a parameter at a specified point in time.

    Args:
        par   (str): the parameter to change
        years (float/arr): the year(s) at which to apply the change
        vals  (any): a value or list of values to change to (if a list, must have the same length as years); or a dict of year:value entries

    If any value is ``'reset'``, reset to the original value.

    **Example**::

        ec0 = fp.change_par(par='exposure_factor', years=[2000, 2010], vals=[0.0, 2.0]) # Reduce exposure factor
        ec0 = fp.change_par(par='exposure_factor', vals={2000:0.0, 2010:2.0}) # Equivalent way of writing
        sim = fp.Sim(interventions=ec0).run()
    '''
    def __init__(self, par, years=None, vals=None, verbose=False):
        super().__init__()
        self.par   = par
        self.verbose = verbose
        if isinstance(years, dict): # Swap if years is supplied as a dict, so can be supplied first
            vals = years
        if vals is None:
            errormsg = 'Values must be supplied'
            raise ValueError(errormsg)
        if isinstance(vals, dict):
            years = sc.dcp(list(vals.keys()))
            vals = sc.dcp(list(vals.values()))
        else:
            if years is None:
                errormsg = 'If vals is not supplied as a dict, then year(s) must be supplied'
                raise ValueError(errormsg)
            else:
                years = sc.toarray(sc.dcp(years))
                vals = sc.dcp(vals)
                if sc.isnumber(vals):
                    vals = sc.tolist(vals) # We want to be careful not to take something that might already be an array and interpret different values as years
                n_years = len(years)
                n_vals = len(vals)
                if n_years != n_vals:
                    errormsg = f'Number of years ({n_years}) does not match number of values ({n_vals})'
                    raise ValueError(errormsg)

        self.years = years
        self.vals = vals

        return


    def initialize(self, sim):
        super().initialize()

        # Validate parameter name
        if self.par not in sim.pars:
            errormsg = f'Parameter "{self.par}" is not a valid sim parameter'
            raise ValueError(errormsg)

        # Validate years and values
        years = self.years
        min_year = min(years)
        max_year = max(years)
        if min_year < sim['start_year']:
            errormsg = f'Intervention start {min_year} is before the start of the simulation'
            raise ValueError(errormsg)
        if max_year > sim['end_year']:
            errormsg = f'Intervention end {max_year} is after the end of the simulation'
            raise ValueError(errormsg)
        if years != sorted(years):
            errormsg = f'Years {years} should be monotonic increasing'
            raise ValueError(errormsg)

        # Convert intervention years to sim timesteps
        self.counter = 0
        self.inds = sc.autolist()
        for y in years:
            self.inds += sc.findnearest(sim.tvec, y)

        # Store original value
        self.orig_val = sc.dcp(sim[self.par])

        return


    def apply(self, sim):
        if len(self.inds) > self.counter:
            ind = self.inds[self.counter] # Find the current index
            if sim.i == ind: # Check if the current timestep matches
                curr_val = sc.dcp(sim[self.par])
                val = self.vals[self.counter]
                if val == 'reset':
                    val = self.orig_val
                sim[self.par] = val # Update the parameter value -- that's it!
                if self.verbose:
                    label = f'Sim "{sim.label}": ' if sim.label else ''
                    print(f'{label}On {sim.y}, change {self.counter+1}/{len(self.inds)} applied: "{self.par}" from {curr_val} to {sim[self.par]}')
                self.counter += 1
        return


    def finalize(self):
        # Check that all changes were applied
        n_counter = self.counter
        n_vals = len(self.vals)
        if n_counter != n_vals:
            errormsg = f'Not all values were applied ({n_vals} ≠ {n_counter})'
            raise RuntimeError(errormsg)
        return


def key2ind(sim, key):
    """
    Take a method key and convert to an int, e.g. 'Condoms' → 7
    """
    ind = key
    if ind in fpd.none_all_keys:
        ind = slice(None) # This is equivalent to ":" in matrix[:,:]
    elif isinstance(ind, str):
        ind = sim.pars['methods']['map'][key]
    return ind


def getval(v):
    ''' Handle different ways of supplying a value -- number, distribution, function '''
    if sc.isnumber(v):
        return v
    elif isinstance(v, dict):
        return fpu.sample(**v)[0]
    elif callable(v):
        return v()


class update_methods(Intervention):
    """
    Intervention to modify method efficacy and/or switching matrix.

    Args:
        year (float): The year we want to change the method.

        eff (dict):
            An optional key for changing efficacy; its value is a dictionary with the following schema:

                {method: efficacy}
                    Where method is the method to be changed, and efficacy is the new efficacy (can include multiple keys).


        probs (list): A list of dictionaries where each dictionary has the following keys:

            source (str): The source method to be changed.
            dest (str) The destination method to be changed.
            factor (float): The factor by which to multiply existing probability; OR
            value (float): The value to replace the switching probability value.
            keys (list): A list of strings representing age groups to affect.
            matrix (str): One of ['probs', 'probs1', 'probs1to6'] where:

                probs:     Changes the specified uptake at the corresponding year regardless of state.
                probs1:    Changes the specified uptake for all individuals in their first month postpartum.
                probs1to6: Changes the specified uptake for all individuals that are in the first 6 months postpartum.
    """

    def __init__(self, year, eff=None, probs=None, matrix=None, verbose=False):
        super().__init__()
        self.year    = year
        self.eff     = eff
        self.probs   = probs
        self.matrix  = matrix
        self.verbose = verbose

        # Validation
        if self.year is None:
            errormsg = 'A year must be supplied'
            raise ValueError(errormsg)
        if self.eff is None and self.probs is None:
            errormsg = 'Either efficacy or probabilities must be supplied'
            raise ValueError(errormsg)

        self.applied = False

        return


    def apply(self, sim):
        """
        Applies the efficacy or contraceptive uptake changes if it is the specified year
        based on scenario specifications.
        """

        if not self.applied and sim.y >= self.year:
            self.applied = True # Ensure we don't apply this more than once

            # Implement efficacy
            if self.eff is not None:
                for k,rawval in self.eff.items():
                    try:
                        key2ind(sim, k)
                    except:
                        errormsg = f'Key "{k}" is not a valid method: are you sure this is an efficacy change?'
                        raise ValueError(errormsg)
                    v = getval(rawval)
                    ind = key2ind(sim, k)
                    orig = sim['method_efficacy'][ind]
                    sim['method_efficacy'][ind] = v
                    if self.verbose:
                        print(f'At time {sim.y:0.1f}, efficacy for method {k} was changed from {orig:0.3f} to {v:0.3f}')

            # Implement method mix shift
            raw = sim['methods']['raw'] # We adjust the raw matrices, so the effects are persistent
            if self.probs is not None:
                probs = sc.tolist(self.probs)
                for entry in probs:
                    entry = sc.dcp(entry)
                    s_matrix = entry.pop('matrix', self.matrix) # Switching matrix
                    ages     = entry.pop('ages', None)
                    source   = entry.pop('source', None)
                    dest     = entry.pop('dest', None)
                    method   = entry.pop('method', None)
                    factor   = entry.pop('factor', None)
                    value    = entry.pop('value', None)
                    i_factor = entry.pop('init_factor', None)
                    d_factor = entry.pop('discont_factor', None)
                    i_value  = entry.pop('init_value', None)
                    d_value  = entry.pop('discont_value', None)

                    # Supply default matrix
                    if s_matrix is None:
                        s_matrix = 'annual'

                    # Validation # CK: TODO: move validation to initialization
                    if len(entry) != 0:
                        errormsg = f'Keys "{sc.strjoin(entry.keys())}" not valid entries; see fp.make_scen() for valid args'
                        raise ValueError(errormsg)

                    # Validate method/source/dest
                    if method is not None:
                        if (source is not None or dest is not None):
                            errormsg = 'You can supply "method" as an alternative to "source" and "dest", but not both'
                            raise ValueError(errormsg)
                        else:
                            source = method
                            dest = method

                    # Ensure correct number of inputs are given
                    n_vals = len(sc.mergelists(factor, value, i_factor, d_factor, i_value, d_value))
                    if n_vals != 1:
                        errormsg = f'Must supply one and only one of factor, value, or initiation/discontinuation factors/values; you supplied {n_vals}'
                        raise ValueError(errormsg)

                    # Check nothing strange has happened
                    is_switch  = len(sc.mergelists(factor, value))
                    is_init    = len(sc.mergelists(i_value, i_factor))
                    is_discont = len(sc.mergelists(d_value, d_factor))
                    if is_switch + is_init + is_discont != 1:
                        errormsg = f'Could not figure out what to do: switching={is_switch}, initiation={is_init}, discontinuation={is_discont}, but only one should happen'
                        raise ValueError(errormsg)

                    if is_init: # It's initiation
                        source = 'None'
                        dest = method
                    elif is_discont: # It's discontinuation
                        source = method
                        dest = 'None'
                    elif (source is None) and (dest is None):
                        errormsg = 'Must supply a source or a destination'
                        raise ValueError(errormsg)

                    # Convert from strings to indices
                    source = key2ind(sim, source)
                    dest = key2ind(sim, dest)

                    # Decide if it's a factor or a value modification
                    factor = sc.mergelists(factor, i_factor, d_factor)
                    value  = sc.mergelists(value, i_value, d_value)
                    factor = factor[0] if factor else None
                    value  = value[0]  if value  else None

                    # Replace age keys with all ages if so asked
                    if ages in fpd.none_all_keys:
                        ages = raw['annual'].keys()
                    else:
                        ages = sc.tolist(ages)

                    # Actually loop over the matrices and apply the changes
                    for k in ages:
                        matrix = raw[s_matrix][k]
                        if s_matrix == 'pp0to1': # Handle the postpartum initialization *vector*
                           orig = matrix[dest]
                           if factor is not None:
                               matrix[dest] *= getval(factor)
                           elif value is not None:
                               val = getval(value)
                               matrix[dest] = 0
                               matrix *= (1-val)/matrix.sum()
                               matrix[dest] = val
                               assert np.isclose(matrix.sum(), 1, atol=1e-3), f'Matrix should sum to 1, not {matrix.sum()}'
                           if self.verbose:
                               print(f'At time {sim.y:0.1f}, matrix {s_matrix} for age group {k} was changed from:\n{orig}\nto\n{matrix[dest]}')
                        else: # Handle annual switching *matrices*
                            orig = matrix[source, dest]
                            if factor is not None:
                                matrix[source, dest] *= getval(factor)
                            elif value is not None:
                                val = getval(value)
                                matrix[source, dest] = 0
                                matrix[source, :] *= (1-val)/matrix[source, :].sum()
                                matrix[source, dest] = val
                                assert np.isclose(matrix[source, :].sum(), 1, atol=1e-3), f'Matrix should sum to 1, not {matrix.sum()}'
                            if self.verbose:
                                print(f'At time {sim.y:0.1f}, matrix {self.matrix} for age group {k} was changed from:\n{orig}\nto\n{matrix[source, dest]}')

        return