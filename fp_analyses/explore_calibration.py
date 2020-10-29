'''
Explorations of the calibration object
'''

import sciris as sc
import numpy as np
import pylab as pl
import pandas as pd

pd.set_option('display.max_columns', 20) # Show full dataframe
pd.set_option('display.width', 999) # Show full dataframe

calib = sc.loadobj('senegal_calibration.obj')

# Check that keys match
data_keys = calib.dhs_data.keys()

model_keys = calib.model_to_calib.keys()

assert set(data_keys) == set(model_keys), 'Data and model keys do not match'

# Compare the two
comparison = []
for key in data_keys:
    dv = calib.dhs_data[key]
    mv = calib.model_to_calib[key]
    cmp = sc.objdict(key=key,
                     d_type=type(dv),
                     m_type=type(mv),
                     d_shape=np.array(dv).shape,
                     m_shape=np.array(mv).shape,
                     d_val=None,
                     m_val=None)
    if sc.isnumber(dv):
        cmp.d_val = dv
    if sc.isnumber(mv):
        cmp.m_val = mv

    comparison.append(cmp)


df = pd.DataFrame.from_dict(comparison)
print(df)


print('Done.')