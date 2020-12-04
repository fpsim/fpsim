'''
Explorations of the calibration object
'''

import sciris as sc
import pandas as pd

pd.set_option('display.max_columns', 20) # Show full dataframe
pd.set_option('display.width', 999) # Show full dataframe

print('Loading saved calibration...')
calib = sc.loadobj('senegal_calibration.obj')

print('Computing comparison...')
df = calib.compare()
print(df)

print('Plotting...')
calib.plot()

print('Fitting...')
calib.compute_fit()
calib.fit.plot()

print('Mismatch: ', calib.fit.mismatch)

print('Done.')