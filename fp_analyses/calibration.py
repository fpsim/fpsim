'''
File to extract outputs to calibration from model and compare to data
'''

import sciris as sc
import fpsim as fps # Need to rename
import senegal_parameters as sp

sc.tic()

'''
CALIBRATION TARGETS:
'''
# ALWAYS ON - Information for scoring needs to be extracted from DataFrame with cols 'Age', 'Parity', 'Currently pregnant'
# Dict key: 'pregnancy_parity'
# Overall age distribution and/or percent of population in each age bin
# Age distribution of agents currently pregnant
# Age distribution of agents in each parity group (can be used to make box plot)
# Percent of reproductive age female population in each parity group

do_save = True # Whether to save the completed calibration




pars = sp.make_pars()
calib = fps.Calibration()
calib.run()

if do_save:
    sc.saveobj('senegal_calibration.obj', calib)

sc.toc()

print('Done.')
