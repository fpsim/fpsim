'''
Base classes for loading parameters and for running simulations with FP model
'''

import numpy as np
import pandas as pd
import sciris as sc
import pylab as pl
from . import utils as fpu
from . import defaults as fpd

obj_get = object.__getattribute__ # Alias the default getattribute method
obj_set = object.__setattr__


__all__ = ['ParsObj', 'BasePeople', 'BaseSim']


class FlexPretty(sc.prettyobj):
    '''
    A class that supports multiple different display options: namely obj.brief()
    for a one-line description and obj.disp() for a full description.
    '''

    def __repr__(self):
        ''' Use brief repr by default '''
        try:
            string = self._brief()
        except Exception as E:
            string = sc.objectid(self)
            string += f'Warning, something went wrong printing object:\n{str(E)}'
        return string

    def _disp(self):
        ''' Verbose output -- use Sciris' pretty repr by default '''
        return sc.prepr(self)

    def disp(self, output=False):
        ''' Print or output verbose representation of the object '''
        string = self._disp()
        if not output:
            print(string)
        else:
            return string

    def _brief(self):
        ''' Brief output -- use a one-line output, a la Python's default '''
        return sc.objectid(self)

    def brief(self, output=False):
        ''' Print or output a brief representation of the object '''
        string = self._brief()
        if not output:
            print(string)
        else:
            return string



class ParsObj(FlexPretty):
    '''
    A class based around performing operations on a self.pars dict.
    '''

    def __init__(self, pars, **kwargs):
        self.update_pars(pars, create=True, **kwargs)
        return


    def __getitem__(self, key):
        ''' Allow sim['par_name'] instead of sim.pars['par_name'] '''
        try:
            return self.pars[key]
        except:
            all_keys = '\n'.join(list(self.pars.keys()))
            errormsg = f'Key "{key}" not found; available keys:\n{all_keys}'
            raise sc.KeyNotFoundError(errormsg)


    def __setitem__(self, key, value):
        ''' Ditto '''
        if key in self.pars:
            self.pars[key] = value
        else:
            all_keys = '\n'.join(list(self.pars.keys()))
            errormsg = f'Key "{key}" not found; available keys:\n{all_keys}'
            raise sc.KeyNotFoundError(errormsg)
        return


    def update_pars(self, pars=None, create=False, **kwargs):
        '''
        Update internal dict with new pars.

        Args:
            pars (dict): the parameters to update (if None, do nothing)
            create (bool): if create is False, then raise a KeyNotFoundError if the key does not already exist
            kwargs (dict): additional parameters
        '''
        pars = sc.mergedicts(pars, kwargs)
        if pars:
            if not isinstance(pars, dict):
                raise TypeError(f'The pars object must be a dict; you supplied a {type(pars)}')
            if not hasattr(self, 'pars'):
                self.pars = pars
            if not create:
                available_keys = list(self.pars.keys())
                mismatches = [key for key in pars.keys() if key not in available_keys]
                if len(mismatches):
                    errormsg = f'Key(s) {mismatches} not found; available keys are {available_keys}'
                    raise sc.KeyNotFoundError(errormsg)
            self.pars.update(pars)
        return


class BasePeople(sc.prettyobj):
    '''
    Class for all the people in the simulation.
    '''

    def __init__(self):
        ''' Initialize essential attributes used for filtering '''
        obj_set(self, '_keys', []) # Since getattribute is overwritten
        obj_set(self, '_inds', None)
        return


    def __len__(self):
        try:
            return len(self.uid)
        except Exception as E:
            print(f'Warning: could not get length of People (could not get self.uid: {E})')
            return 0


    def __getitem__(self, key):
        ''' Allow people['attr'] instead of getattr(people, 'attr')
            If the key is an integer, alias `people.person()` to return a `Person` instance
        '''
        try:
            return self.__getattribute__(key)
        except: # pragma: no cover
            if isinstance(key, int):
                return self.person(key)
            else:
                errormsg = f'Key "{key}" is not a valid attribute of people'
                raise AttributeError(errormsg)


    def __setitem__(self, key, value):
        ''' Ditto '''
        self.__dict__[key] = value
        return


    def _is_filtered(self, attr):
        ''' Determine if a given attribute is filtered (e.g. people.age is, people.inds isn't) '''
        is_filtered = (self._inds is not None and attr in self._keys)
        return is_filtered


    def __getattribute__(self, attr):
        ''' For array quantities, handle filtering '''
        output  = obj_get(self, attr)
        if attr[0] == '_': # Short-circuit for built-in methods to save time
            return output
        else:
            try: # Unclear wy this fails, but sometimes it does during initialization/pickling
                keys = obj_get(self, '_keys')
            except:
                keys = []
            if attr not in keys:
                return output
            else:
                if self._is_filtered(attr):
                    output = output[self.inds]
        return output


    def __setattr__(self, attr, value):
        ''' Ditto '''
        if self._is_filtered(attr):
            array = obj_get(self, attr)
            array[self.inds] = value
        else:   # If not initialized, rely on the default behavior
            obj_set(self, attr, value)
        return


    def __add__(self, people2):
        ''' Combine two people arrays '''

        # Preliminaries
        newpeople = self
        keys      = self.keys()
        n_orig    = len(newpeople)
        max_uid   = newpeople.uid.max() + 1
        n_new     = len(people2)

        # Merge arrays
        for key in keys:
            npval = newpeople[key]
            p2val = people2[key]
            if isinstance(npval, np.ndarray):
                newpeople[key] = np.concatenate([npval, p2val], axis=0)
            elif isinstance(npval, list):
                newpeople[key] += p2val
            else:
                errormsg = f'Not sure what to do with object of type {type(npval)}'
                raise TypeError(errormsg)

        # Validate
        for key in keys:
            assert len(newpeople[key]) == len(newpeople)
        newpeople.uid[n_orig:] = max_uid + np.arange(n_new) # Reassign UIDs so they're unique

        return newpeople


    def __radd__(self, people2):
        ''' Allows sum() to work correctly '''
        if not people2: return self
        else:           return self.__add__(people2)


    def keys(self):
        ''' Returns keys for all properties of the people object '''
        try: # Unclear wy this fails, but sometimes it does during initialization/pickling
            keys = obj_get(self, '_keys')[:]
        except:
            keys = []
        return keys

    @property
    def is_female(self):
        ''' Boolean array of everyone female '''
        return self.sex == 0

    @property
    def is_male(self):
        ''' Boolean array of everyone male '''
        return self.sex == 1

    @property
    def int_age(self):
        ''' Return ages as an integer '''
        return np.array(self.age, dtype=np.int64)

    @property
    def ceil_age(self):
        ''' Rounds age up to the next highest integer'''
        return np.array(np.ceil(self.age))

    @property
    def int_age_clip(self):
        ''' Return ages as integers, clipped to maximum allowable age for pregnancy '''
        return np.minimum(self.int_age, fpd.max_age_preg)

    @property
    def n(self):
        ''' Number of people alive '''
        return self.alive.sum()

    @property
    def inds(self):
        ''' Alias to self._inds to prevent accidental overwrite & increase speed '''
        return self._inds

    @property
    def len_inds(self):
        ''' Alias to len(self) '''
        if self._inds is not None:
            return len(self._inds)
        else:
            return len(self)

    @property
    def len_people(self):
        ''' Full length of People array, ignoring filtering '''
        return len(self.unfilter())


    def plot(self, fig_args=None, hist_args=None):
        ''' Plot histograms of each quantity '''

        fig_args  = sc.mergedicts(fig_args)
        hist_args = sc.mergedicts(dict(bins=50), hist_args)
        keys = self.keys()
        nkeys = len(keys)
        rows,cols = sc.get_rows_cols(nkeys)

        fig = pl.figure(**fig_args)

        for k,key in enumerate(keys):
            pl.subplot(rows,cols,k+1)
            try:
                data = np.array(self[key], dtype=float)
                mean = data.mean()
                label = f'mean: {mean}'
                pl.hist(data, label=label, **hist_args)
                pl.title(key)
                pl.legend()
            except:
                pl.title(f'Could not plot {key}')

        return fig


    def filter(self, criteria=None, inds=None):
        '''
        Store indices to allow for easy filtering of the People object.

        Args:
            criteria (array): a boolean array for the filtering critria
            inds (array): alternatively, explicitly filter by these indices

        Returns:
            A filtered People object, which works just like a normal People object
            except only operates on a subset of indices.
        '''

        # Create a new People object with the same properties as the original
        filtered = object.__new__(self.__class__) # Create a new People instance
        BasePeople.__init__(filtered) # Perform essential initialization
        filtered.__dict__ = {k:v for k,v in self.__dict__.items()} # Copy pointers to the arrays in People

        # Perform the filtering
        if criteria is None: # No filtering: reset
            filtered._inds = None
            if inds is not None: # Unless indices are supplied directly, in which case use them
                filtered._inds = inds
        else: # Main use case: perform filtering
            if len(criteria) == len(self): # Main use case: a new filter applied on an already filtered object, e.g. filtered.filter(filtered.age > 5)
                new_inds = criteria.nonzero()[0] # Criteria is already filtered, just get the indices
            elif len(criteria) == self.len_people: # Alternative: a filter on the underlying People object is applied to the filtered object, e.g. filtered.filter(people.age > 5)
                new_inds = criteria[filtered.inds].nonzero()[0] # Apply filtering before getting the new indices
            else:
                errormsg = f'"criteria" must be boolean array matching either current filter length ({self.len_inds}) or else the total number of people ({self.len_people}), not {len(criteria)}'
                raise ValueError(errormsg)
            if filtered.inds is None: # Not yet filtered: use the indices directly
                filtered._inds = new_inds
            else: # Already filtered: map them back onto the original People indices
                filtered._inds = filtered.inds[new_inds]

        return filtered


    def unfilter(self):
        '''
        An easy way of unfiltering the People object, returning the original.
        '''
        unfiltered = self.filter(criteria=None)
        return unfiltered


    def binomial(self, prob, as_inds=False, as_filter=False):
        '''
        Return indices either by a single probability or by an array of probabilities.
        By default just return the boolean array, but can also return the indices,
        or the filtered People object.

        Args:
            prob (float/array): either a scalar probability, or an array of probabilities of the same length as People
            as_inds (bool): return as list of indices instead of a boolean array
            as_filter (bool): return as filter instead than boolean array
        '''
        if sc.isnumber(prob):
            arr = fpu.n_binomial(prob, len(self))
        elif sc.isarray(prob):
            arr = fpu.binomial_arr(prob)
        else:
            errormsg = f'Could not recognize {type(prob)} as a scalar or array'
            raise TypeError(errormsg)
        if as_inds:
            output = sc.findinds(arr)
        elif as_filter:
            output = self.filter(arr)
        else:
            output = arr
        return output


class BaseSim(ParsObj):
    '''
    The BaseSim class handles the dynamics of the simulation.
    '''

    def year2ind(self, year):
        index = int((year - self.pars['start_year']) * fpd.mpy / self.pars['timestep'])
        return index

    def ind2year(self, ind):
        year = ind * self.pars['timestep'] / fpd.mpy  # Months
        return year

    def ind2calendar(self, ind):
        year = self.ind2year(ind) + self.pars['start_year']
        return year

    @property
    def npts(self):
        ''' Count the number of points in timesteps between the starting year and the ending year.'''
        try:
            return int(fpd.mpy * (self.pars['end_year'] - self.pars['start_year']) / self.pars['timestep'] + 1)
        except:
            return 0

    @property
    def tvec(self):
        ''' Create a time vector array at intervals of the timestep in years '''
        try:
            return self.pars['start_year'] + np.arange(self.npts) * self.pars['timestep'] / fpd.mpy
        except:
            return np.array([])

    @property
    def n(self):
        return self.people.alive.sum()


    def _brief(self):
        '''
        Return a one-line description of a sim -- used internally and by repr();
        see sim.brief() for the user version.
        '''
        # Try to get a detailed description of the sim...
        try:
            if self.already_run:
                s = self.summary
                results = f'b={s.births:n} ☠={s.deaths:n} pop={s.final:n}'
            else:
                results = 'not run'
    
            # Set label string
            labelstr = f'"{self.label}"' if self.label else '<no label>'
    
            start = self['start_year']
            end = self['end_year']
            n_agents = self['n_agents']
            string   = f'Sim({labelstr}; n={n_agents:n}; {start}-{end}; results: {results})'

        # ...but if anything goes wrong, return the default with a warning
        except Exception as E: # pragma: no cover
            string = sc.objectid(self)
            string += f'Warning, sim appears to be malformed; use sim.disp() for details:\n{str(E)}'

        return string


    def _get_ia(self, which, label=None, partial=False, as_list=False, as_inds=False, die=True, first=False):
        ''' Helper method for get_interventions() and get_analyzers(); see get_interventions() docstring '''

        # Handle inputs
        if which not in ['interventions', 'analyzers']: # pragma: no cover
            errormsg = f'This method is only defined for interventions and analyzers, not "{which}"'
            raise ValueError(errormsg)

        ia_list = sc.tolist(self.pars[which]) # List of interventions or analyzers
        n_ia = len(ia_list) # Number of interventions/analyzers

        if label == 'summary': # Print a summary of the interventions
            df = pd.DataFrame(columns=['ind', 'label', 'type'])
            for ind,ia_obj in enumerate(ia_list):
                df = df.append(dict(ind=ind, label=str(ia_obj.label), type=type(ia_obj)), ignore_index=True)
            print(f'Summary of {which}:')
            print(df)
            return

        else: # Standard usage case
            position = 0 if first else -1 # Choose either the first or last element
            if label is None: # Get all interventions if no label is supplied, e.g. sim.get_interventions()
                label = np.arange(n_ia)
            if isinstance(label, np.ndarray): # Allow arrays to be provided
                label = label.tolist()
            labels = sc.promotetolist(label)

            # Calculate the matches
            matches = []
            match_inds = []
            for label in labels:
                if sc.isnumber(label):
                    matches.append(ia_list[label]) # This will raise an exception if an invalid index is given
                    label = n_ia + label if label<0 else label # Convert to a positive number
                    match_inds.append(label)
                elif sc.isstring(label) or isinstance(label, type):
                    for ind,ia_obj in enumerate(ia_list):
                        if sc.isstring(label) and ia_obj.label == label or (partial and (label in str(ia_obj.label))):
                            matches.append(ia_obj)
                            match_inds.append(ind)
                        elif isinstance(label, type) and isinstance(ia_obj, label):
                            matches.append(ia_obj)
                            match_inds.append(ind)
                else: # pragma: no cover
                    errormsg = f'Could not interpret label type "{type(label)}": should be str, int, list, or {which} class'
                    raise TypeError(errormsg)

            # Parse the output options
            if as_inds:
                output = match_inds
            elif as_list: # Used by get_interventions()
                output = matches
            else:
                if len(matches) == 0: # pragma: no cover
                    if die:
                        errormsg = f'No {which} matching "{label}" were found'
                        raise ValueError(errormsg)
                    else:
                        output = None
                else:
                    output = matches[position] # Return either the first or last match (usually), used by get_intervention()

            return output


    def get_interventions(self, label=None, partial=False, as_inds=False):
        '''
        Find the matching intervention(s) by label, index, or type. If None, return
        all interventions. If the label provided is "summary", then print a summary
        of the interventions (index, label, type).

        Args:
            label (str, int, Intervention, list): the label, index, or type of intervention to get; if a list, iterate over one of those types
            partial (bool): if true, return partial matches (e.g. 'beta' will match all beta interventions)
            as_inds (bool): if true, return matching indices instead of the actual interventions
        '''
        return self._get_ia('interventions', label=label, partial=partial, as_inds=as_inds, as_list=True)


    def get_intervention(self, label=None, partial=False, first=False, die=True):
        '''
        Like get_interventions(), find the matching intervention(s) by label,
        index, or type. If more than one intervention matches, return the last
        by default. If no label is provided, return the last intervention in the list.

        Args:
            label (str, int, Intervention, list): the label, index, or type of intervention to get; if a list, iterate over one of those types
            partial (bool): if true, return partial matches (e.g. 'beta' will match all beta interventions)
            first (bool): if true, return first matching intervention (otherwise, return last)
            die (bool): whether to raise an exception if no intervention is found
        '''
        return self._get_ia('interventions', label=label, partial=partial, first=first, die=die, as_inds=False, as_list=False)


    def get_analyzers(self, label=None, partial=False, as_inds=False):
        '''
        Same as get_interventions(), but for analyzers.
        '''
        return self._get_ia('analyzers', label=label, partial=partial, as_list=True, as_inds=as_inds)


    def get_analyzer(self, label=None, partial=False, first=False, die=True):
        '''
        Same as get_intervention(), but for analyzers.
        '''
        return self._get_ia('analyzers', label=label, partial=partial, first=first, die=die, as_inds=False, as_list=False)

