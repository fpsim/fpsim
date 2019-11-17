import os
import pylab as pl
import pandas as pd
import sciris as sc
import fp_data.DHS.calobj as co

sc.heading('Setting parameters...')

uhri_file = os.path.join(os.pardir, 'UHRI', 'senegal_women.obj')
dhs_file = os.path.join(os.pardir, 'DHS', 'senegal.cal')


#%% Loading data
sc.heading('Loading data...')
uhri = sc.loadobj(uhri_file)
dhs = co.load(dhs_file) # Create from saved data
f1 = dhs.plot_transitions()
f2 = dhs.plot_transitions(projection='3d')
f3 = dhs.plot_slice('None')



print('Done.')