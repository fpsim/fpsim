import os
import pylab as pl
import pandas as pd
import sciris as sc

sc.heading('Setting parameters...')

uhri_file = os.path.join(os.pardir, 'UHRI', 'store.hdf')

#%% Loading data
sc.heading('Loading data...')

# UHRI data
store = pd.HDFStore(uhri_file)
uhri = store['women']

# DHS data



print('Done.')