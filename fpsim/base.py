'''
Base classes for loading parameters and for running simulations with FP model
'''

import numpy as np
import sciris as sc
import pylab as pl
from . import defaults as fpd
from . import utils as fpu
obj_get = object.__getattribute__ # Alias the default getattribute method
obj_set = object.__setattr__


__all__ = ['ParsObj', 'BasePeople', 'BaseSim']


class ParsObj(sc.prettyobj):
    '''
    A class based around performing operations on a self.pars dict.
    '''

    def __init__(self, pars):
        self.update_pars(pars)
        return

    def __getitem__(self, key):
        ''' Allow sim['par_name'] instead of sim.pars['par_name'] '''
        return self.pars[key]

    def __setitem__(self, key, value):
        ''' Ditto '''
        self.pars[key] = value
        self.update_pars()
        return

    def update_pars(self, pars=None):
        ''' Update internal dict with new pars '''
        if not hasattr(self, 'pars'):
            if pars is None:
                raise Exception('Must call update_pars either with a pars dict or with existing pars')
            else:
                self.pars = pars
        elif pars is not None:
            self.pars.update(pars)
        return


class BasePeople(sc.prettyobj):
    '''
    Class for all the people in the simulation.
    '''

    def __init__(self):
        obj_set(self, '_keys', []) # Since getattribute is overwritten
        self.inds = None
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
        is_filtered = (attr in self.keys() and self.inds is not None)
        return is_filtered


    def __getattribute__(self, attr):
        ''' Route property access to the underlying entity '''
        output  = obj_get(self, attr)
        keys = obj_get(self, 'keys')()
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
        return obj_get(self, '_keys')[:]

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
    def int_age_clip(self):
        ''' Return ages as integers, clipped to maximum allowable age for pregnancy '''
        return np.minimum(self.int_age, fpd.max_age_preg)

    def female_inds(self):
        return sc.findinds(self.is_female)

    def male_inds(self):
        return sc.findinds(self.is_male)

    @property
    def n(self):
        return self.alive.sum()

    @property
    def len_inds(self):
        ''' Alias to len(self) '''
        if self.inds is not None:
            return len(self.inds)
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
        '''

        # Create a new People object with the same properties as the original
        filtered = object.__new__(self.__class__)
        BasePeople.__init__(filtered)
        filtered.__dict__ = {k:v for k,v in self.__dict__.items()}

        # Perform the filtering
        if criteria is None:
            filtered.inds = None
            if inds is not None:
                filtered.inds = inds
        else:
            if len(criteria) == len(self):
                filtered.inds = criteria.nonzero()[0] # Criteria is already filtered
            elif len(criteria) == self.len_people:
                filtered.inds = criteria[self.inds].nonzero()[0] # Criteria is not filtered yet
            else:
                errormsg = f'"criteria" must be boolean array matching either current filter length ({self.len_inds}) or else the total number of people ({self.len_people}), not {len(criteria)}'
                raise ValueError(errormsg)

        return filtered


    def unfilter(self):
        '''
        An easy way of unfiltering the People object.
        '''
        unfiltered = self.filter(criteria=None)
        return unfiltered


    def binomial(self, prob, as_inds=False, as_filter=False):
        '''
        Return indices either by a single probability or by an array of probabilities.

        Args:
            prob (float/array): either a scalar probability, or an array of probabilities
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
    The BaseSim class handles the admin work of managing time in the simulation.
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










