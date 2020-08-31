'''
Calculate age-dependent mortality rates. From WPP, Senegal, 1990-1995.

WPP2019_MORT_F15_2_LIFE_TABLE_SURVIVORS_MALE.xlsx
WPP2019_MORT_F15_2_LIFE_TABLE_SURVIVORS_FEMALE.xlsx
'''

import pylab as pl
import sciris as sc

m_life_table = [100000, 92401, 85538, 84405, 83715, 82683, 81138, 79620, 77909, 75974, 73514, 70398, 66191, 60576, 52950, 43086, 30618, 17170, 6564, 1405, 145, 8]
f_life_table = [100000, 93297, 86904, 85771, 85081, 84065, 82763, 81308, 79788, 78091, 76145, 73879, 70825, 66697, 60429, 51306, 38433, 23232, 9731, 2430, 310, 21]

bins = 5*pl.arange(len(m_life_table))[:-2] # Remove last bin

mortality = sc.objdict()
mortality.bins = bins
mortality.m = pl.zeros(len(bins))
mortality.f = pl.zeros(len(bins))
for key in ['m','f']:
    for b in range(len(bins)):
        if key == 'm': data = m_life_table
        else:          data = f_life_table
        ratio = data[b+1]/data[b]
        rate = ratio**(1/5.)
        mortality[key][b] = 1-rate