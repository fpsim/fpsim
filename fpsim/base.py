'''
Base classes for loading parameters and for running simulations with FP model
'''

import numpy as np
import sciris as sc
from . import defaults as fpd


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
            return self.__dict__[key]
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


    def __add__(self, people2):
        ''' Combine two people arrays '''

        # Preliminaries
        newpeople = sc.dcp(self)
        keys      = self.keys()
        n_orig    = len(newpeople)
        max_uid   = newpeople.uid.max() + 1
        n_new     = len(people2)

        # Merge arrays
        for key in keys:
            npval = newpeople[key]
            p2val = people2[key]
            if sc.isarray(npval):
                newpeople[key] = np.concatenate([npval, p2val], axis=0)
            elif isinstance(npval, list):
                newpeople[key] = npval + p2val
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
        if hasattr(self, '_keys'):
            return sc.dcp(self._keys)
        else:
            return []
        return

    @property
    def is_female(self):
        return self.sex == 0

    @property
    def is_male(self):
        return self.sex == 1

    @property
    def int_ages(self):
        return np.array(self.ages, dtype=np.int64)

    def female_inds(self):
        return sc.findinds(self.is_female)

    def male_inds(self):
        return sc.findinds(self.is_male)

    @property
    def n(self):
        return self.alive.sum()



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









