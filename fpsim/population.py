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


def lookup_fertility_mortality_splines(pars, which = None, bound=True):
    """
    Returns splines of fertility or mortality data evaluated along arrays of ages and trends over years
    For use in calculating probability of pregnancy or death
    'which' argument refers to either fertility or mortality from parameters
    """
    data = pars[which]
    ages = pl.arange(resolution * pars['max_age'] + 1) / resolution
    years = pl.arange(resolution * (pars['end_year'] - pars['start_year']) + 1) / resolution + pars['start_year']
    male_fertility_mortality_spline = si.splrep(x=data['bins'],
                                 y=data['m'])  # Create a spline of fertility or mortality along known age bins
    female_fertility_mortality_spline = si.splrep(x=data['bins'], y = data['f'])
    trend_spline = si.splrep(x=data['years'],
                             y=data['trend'])  # Create a spline of the mortality trend over years we have data
    male_fertility_mortality_lookup = si.splev(ages,
                                male_fertility_mortality_spline) # Evaluate the spline along the range of ages in the model with resolution
    female_fertility_mortality_lookup = si.splev(ages,
                                               female_fertility_mortality_spline)
    trend_lookup = si.splev(years, trend_spline)  # Evaluate the spline along the range of years in the model
    if bound:
        male_fertility_mortality_lookup = pl.minimum(1, pl.maximum(0, male_fertility_mortality_lookup))
        female_fertility_mortality_lookup = pl.minimum(1, pl.maximum(0, female_fertility_mortality_lookup))

    return male_fertility_mortality_lookup, female_fertility_mortality_lookup, trend_lookup

def lookup_fecundity_splines(pars, bound = True ):

    ages = pl.arange(resolution * pars['max_age'] + 1) / resolution
    male_fecundity_interp_model = si.interp1d(x = pars['age_fertility']['bins'], y = pars['age_fertility']['m'])
    female_fecundity_interp_model = si.interp1d(x=pars['age_fertility']['bins'], y=pars['age_fertility']['f'])
    male_fecundity_interp = male_fecundity_interp_model(ages)
    female_fecundity_interp = female_fecundity_interp_model(ages)
    if bound:
        male_fecundity_interp = pl.minimum(1, pl.maximum(0, male_fecundity_interp))
        female_fecundity_interp = pl.minimum(1, pl.maximum(0, female_fecundity_interp))

    return male_fecundity_interp, female_fecundity_interp

















