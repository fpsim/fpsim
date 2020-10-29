'''
Explorations of the calibration object
'''

import sciris as sc
import pylab as pl

calib = sc.loadobj('senegal_calibration.obj')

# Check that keys match
data_keys = calib.dhs_data.keys()

model_keys = calib.model_to_calib.keys()

assert set(data_keys) == set(model_keys), 'Data and model keys do not match'

print('Done.')