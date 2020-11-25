'''
File to help process parameters and calculate population attributes

These are all be static values for the population with no random numbers used that would indicate
an agent's characteristics

To be used with model.py for simulation
'''

import pylab as pl
from scipy import interpolate as si

# Specify all externally visible things this file defines
__all__ = ['make_age_sex_splines', 'lookup_fertility_mortality_splines']

resolution = 100 # For spline interpolation

def make_age_sex_splines(pars):
    """ Fit splines to the demographic pyramid data, and returns male fraction of population """
    pyramid = pars['age_pyramid']
    year = pyramid[:, 0]
    year = pl.append(year, pars['max_age'])
    m = pl.insert(pyramid[:, 1], 0, 0)
    f = pl.insert(pyramid[:, 2], 0, 0)
    male_pop_total = m.sum()
    female_pop_total = f.sum()
    total_pop = male_pop_total + female_pop_total
    male_fraction = male_pop_total / total_pop
    m = (pl.cumsum(m)) / male_pop_total  # Transform population into fraction of total population in each 5 yr age bin
    f = (pl.cumsum(f)) / female_pop_total
    m_pop_spline = si.splrep(x=m, y=year)  # Note how axes are swapped for inverse CDF
    f_pop_spline = si.splrep(x=f, y=year)
    return m_pop_spline, f_pop_spline, male_fraction


def lookup_mortality_splines(pars, bound=True):
    """
    Returns splines mortality data evaluated along arrays of ages and trends over years
    For use in calculating probability of death
    Not currently using, moved this code to senegal_parameters.py and took out trend spline and replaced with
    findnearest in model.py code.   Leaving here in case this code is useful.
    """
    data = pars['age_mortality']
    ages = pl.arange(resolution * pars['max_age'] + 1) / resolution
    years = pl.arange(resolution * (pars['end_year'] - pars['start_year']) + 1) / resolution + pars['start_year']
    male_mortality_spline = si.splrep(x=data['bins'],
                                 y=data['m'])  # Create a spline of mortality along known age bins
    female_mortality_spline = si.splrep(x=data['bins'], y = data['f'])
    trend_spline = si.splrep(x=data['years'],
                             y=data['trend'])  # Create a spline of the mortality trend over years we have data
    male_mortality_lookup = si.splev(ages,
                                male_mortality_spline) # Evaluate the spline along the range of ages in the model with resolution
    female_mortality_lookup = si.splev(ages,
                                               female_mortality_spline)
    trend_lookup = si.splev(years, trend_spline)  # Evaluate the spline along the range of years in the model
    if bound:
        male_mortality_lookup = pl.minimum(1, pl.maximum(0, male_mortality_lookup))
        female_mortality_lookup = pl.minimum(1, pl.maximum(0, female_mortality_lookup))

    return male_mortality_lookup, female_mortality_lookup, trend_lookup


















