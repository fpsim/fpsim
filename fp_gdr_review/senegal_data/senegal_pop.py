'''
Generate the population size data for Senegal.

Based on UN World Population Projections.
'''

import pylab as pl

years = pl.arange(1980,2021)

popsize = pl.array([5583, 5740, 5910, 6090, 6277, 6471, 6671, 6876, 7087, 7304, 7526, 7756, 7990, 8227, 8461, 8690, 8913, 9131, 9348, 9569, 9798, 10036, 10284, 10541, 10810, 11090, 11382, 11687, 12005, 12335, 12678, 13034, 13402, 13782, 14175, 14578, 14994, 15419, 15854, 16296, 16744])

popsize *= 1000/popsize[0]