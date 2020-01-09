'''
Calculate age-dependent mortality rates. From WPP, globally, 1990-1995, i.e.

WPP2019_MORT_F15_2_LIFE_TABLE_SURVIVORS_MALE.xlsx
WPP2019_MORT_F15_2_LIFE_TABLE_SURVIVORS_FEMALE.xlsx
'''

import pylab as pl
import sciris as sc

m_life_table = [100000, 93358, 90680, 89863, 89302, 88537, 87539, 86412, 85105, 83511, 81460, 78670, 74682, 69022, 61136, 50760, 38096, 24476, 12526, 4661, 1139, 167]
f_life_table = [100000, 94080, 91177, 90390, 89873, 89229, 88449, 87605, 86668, 85583, 84240, 82404, 79806, 76027, 70471, 62534, 51572, 37833, 23164, 10839, 3473, 679]

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