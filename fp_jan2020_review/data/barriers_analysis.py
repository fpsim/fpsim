import pylab as pl
import pandas as pd
import sciris as sc

#%% Load data
fn = '/home/idm_user/idm/Dropbox/FP Dynamic Modeling/Calibration/Senegal-DHS/DHSIndividualBarriers.csv'
d = pd.read_csv(fn, low_memory=False)
d = d.dropna()

keys = ['No need',
        'Opposition',
        'Knowledge',
        'Access',
        'Health',
        ]

n_rows = len(d)
n_keys = len(keys)

results = pl.zeros((n_rows, n_keys))

for k,key in enumerate(keys):
    results[:,k] = pl.minimum(d[key].to_numpy(), 1)
    
    
#%%
sc.heading('Proportions')
print('Propotion of people with:')
for k,key in enumerate(keys):
    print(f'  {key:10s}: {results[:,k].mean()*100:5.1f}%')


#%%
sc.heading('Counts')
summed = results.sum(axis=1)
print('Propotion of people with:')
for i in range(n_keys):
    print(f'  {i} barriers: {sum(summed==i)/n_rows*100:5.1f}%')

print(f'Mean number of barriers: {summed.mean():0.2f}%')


#%%
sc.heading('Two barriers')
two_barriers_inds = (summed==2)
two_barriers = results[two_barriers_inds,:]
matrix = pl.zeros((n_keys, n_keys))
for row in two_barriers:
    bb = sc.findinds(row) # Find both barriers
    assert len(bb) == 2 # There should always be two
    matrix[bb[0], bb[1]] += 1
matrix /= matrix.sum()/100

print(keys)
print(matrix)

pl.figure(figsize=(12,8))
pl.pcolor(matrix)
tick_locs = pl.arange(n_keys)+0.5
pl.gca().set_xticks(tick_locs)
pl.gca().set_yticks(tick_locs)
pl.gca().set_xticklabels(keys)
pl.gca().set_yticklabels(keys)
pl.title('Interaction between barriers')
pl.colorbar()
    
    
