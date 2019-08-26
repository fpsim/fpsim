'''
Explore FP data
'''

import pandas as pd
import sciris as sc

datafile1 = '/u/cliffk/idm/fp/data/Senegal/Baseline/SEN_base_wm_20160427.dta'
#datafile1 = '/u/cliffk/idm/fp/data/Senegal/Endline/SEN_end_wm_match_20160505.dta'
#datafile2 = '/u/cliffk/idm/fp/data/DHS/example/SNPR6RFL.DTA'
datafile2 = '/u/cliffk/idm/fp/data/DHS/example3/SNIR61FL.DTA'

sc.tic()
#df1 = pd.read_stata(datafile1,convert_categoricals=False)
df2 = pd.read_stata(datafile2,convert_categoricals=False)
sc.toc()

print('Done')