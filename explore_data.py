'''
Explore FP data
'''

import pandas as pd
import sciris as sc

#datafile = '/u/cliffk/idm/fp/data/Senegal/Baseline/SEN_base_wm_20160427.dta'
#datafile = '/u/cliffk/idm/fp/data/Senegal/Endline/SEN_end_wm_match_20160505.dta'
#datafile = '/u/cliffk/idm/fp/data/DHS/example/SNPR6RFL.DTA'
#datafile = '/u/cliffk/idm/fp/data/DHS/example3/SNIR61FL.DTA'
datafile = '/u/cliffk/idm/fp/data/test_load/NGIR6ADT/NGIR6AFL.DTA'
calfile = 'calendar.txt'

sc.tic()
df = pd.read_stata(datafile,convert_categoricals=False)
sc.toc()

calendar = df['vcal_1'].to_list()
calstring = '\n'.join(calendar)

with open(calfile,'w') as f:
    f.write(calstring)


print('Done')