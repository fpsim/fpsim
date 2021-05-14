'''
Base classes for loading parameters and for running simulations with FP model
'''

import numpy as np # Needed for a few things not provided by pl
import sciris as sc
# import numba as nb

__all__ = ['ParsObj', 'BaseSim']

mpy = 12 # Months per year, to avoid magic numbers

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
        # self._pars_to_attrs() # Convert parameters to attributes
        return


    # def _pars_to_attrs(self):
    #     ''' Convert from a dictionary to class attributes in order to avoid having to do dict lookups -- used by update_pars '''
    #     self.par_keys = self.pars.keys()
    #     for key,value in self.pars.items():
    #         setattr(self, key, value)
    #     return None

    # def _attrs_to_pars(self):
    #     ''' Convert back -- not used currently '''
    #     pars = dict()
    #     for key in self.par_keys:
    #         pars[key] = getattr(self, key)
    #     return pars


class BaseSim(ParsObj):
    '''
    The BaseSim class handles the admin work of managing time in the simulation.
    '''
    def __init__(self, pars=None):
        super().__init__(pars)
        self.year2ind(year = 0)
        self.ind2year(ind = 0)
        self.ind2calendar(ind = 0)

    def year2ind(self, year):
        index = int((year - self.pars['start_year']) * mpy / self.pars['timestep'])
        return index

    def ind2year(self, ind):
        year = ind * self.pars['timestep'] / mpy  # Months
        return year

    def ind2calendar(self, ind):
        year = self.ind2year(ind) + self.pars['start_year']
        return year

    @property
    def npts(self):
        ''' Count the number of points in timesteps between the starting year and the ending year.'''
        try:
            return int(mpy * (self.pars['end_year'] - self.pars['start_year']) / self.pars['timestep'] + 1)
        except:
            return 0

    @property
    def tvec(self):
        ''' Create a time vector array at intervals of the timestep in years '''
        try:
            return self.pars['start_year'] + np.arange(self.npts) * self.pars['timestep'] / mpy
        except:
            return np.array([])









