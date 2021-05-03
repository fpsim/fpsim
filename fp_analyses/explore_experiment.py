'''
Explorations of the Experiment object (renamed from calibration)
'''

import sciris as sc
import pandas as pd

pd.set_option('display.max_columns', 20) # Show full dataframe
pd.set_option('display.width', 999) # Show full dataframe

print('Loading saved experiment...')
exp = sc.loadobj('senegal_experiment.obj')

print('Computing comparison...')
df = exp.compare()
print(df)

print('Plotting...')
exp.plot()

print('Fitting...')
exp.compute_fit()
exp.fit.plot()

print('Mismatch: ', exp.fit.mismatch)

print('Done.')