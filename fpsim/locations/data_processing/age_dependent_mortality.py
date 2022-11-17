'''
Calculate age-dependent mortality rates. From WPP 2022, Kenya, 2010.
https://population.un.org/wpp/Download/Standard/Mortality/
WPP2022_MORT_F04_3_LIFE_TABLE_SURVIVORS_MALE.xlsx
WPP2022_MORT_F04_3_LIFE_TABLE_SURVIVORS_FEMALE.xlsx
'''

import pylab as pl
import sciris as sc

m_life_table = [100000, 93580, 92832, 92279, 91359, 89961, 88210, 85989, 83094, 79588, 75346, 69948, 63442, 55031, 44424, 31830, 18656, 7683, 1741, 161, 9]
f_life_table = [100000, 86904, 85771, 85081, 84065, 82763, 81308, 79788, 78091, 76145, 73879, 70825, 66697, 60429, 51306, 38433, 23232, 9731, 2430, 310, 21]



bins = 5*pl.arange(len(m_life_table))[:-1] # Remove last bin

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

print(mortality.m)
print(mortality.f)