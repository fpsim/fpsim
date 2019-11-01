import os
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt # for plt.show()
import pandas as pd

cachefn = 'store.hdf'
store = pd.HDFStore(cachefn)

force_read = False
normalize_by_from = True
write_codebooks = False
uid = None # '96.66.16' # Put a UID here to get see data for one individual across waves or None to disable.


if (not force_read) and os.path.isfile(cachefn) and 'women' in store:
    women = store['women']
else:
    filenames = [
        '/home/dklein/Dropbox (IDM)/URHI/Senegal/Baseline/SEN_base_wm_20160427.dta',
        '/home/dklein/Dropbox (IDM)/URHI/Senegal/Midline/SEN_mid_wm_match_20160609.dta',
        '/home/dklein/Dropbox (IDM)/URHI/Senegal/Endline/SEN_end_wm_match_20160505.dta'
    ]

    datalist = []
    datalist_UID = []
    for wave, filename in enumerate(filenames):
        print('-'*80)
        print(f'On wave {wave}, reading {filename}')
        data = pd.read_stata(filename, convert_categoricals=False)

        # NOTE: Overwriting values for each file, then using last below [BAD]
        values = pd.io.stata.StataReader(filename).value_labels()
        #print([k for k,v in values.items() if '338' in k])
        if write_codebooks:
            codebook = pd.io.stata.StataReader(filename).variable_labels()
            pd.DataFrame({'keys': list(codebook.keys()), 'values': list(codebook.values())}).set_index('keys').to_csv(f'codebook_{wave}.csv')


        data['UID'] = data['location_code'].astype(str) + '.' + data['hhnum'].astype(str) + '.' + data['line'].astype(str)
        data['Wave'] = wave

        # w102 is age, w208 is parity
        # 310 current use of contraception
        if wave == 0:
            cols = {'w102':'Age', 'w208':'Parity', 'method':'Method', 'wm_allcity_wt':'Weight', 'city':'City'}
        elif wave == 1:
            cols = {'mw102':'Age', 'mw208b':'Parity', 'mmethod':'Method', 'wm_allcity_wt':'Weight', 'mcity':'City'}
        elif wave == 2:
            cols = {'ew102':'Age', 'ew208':'Parity', 'emethod':'Method', 'ewoman_weight_6city':'Weight', 'ecity':'City'}

        for c in cols.keys():
            if c not in data.columns:
                raise Exception(f'Cannot find column {c} at wave {wave}!')

        data = data \
            .rename(columns=cols) \
            .replace({
                'Method': values['method'],
            })
            # 'City': values['City'],

        datalist.append( data[['UID', 'Wave', 'Age', 'Parity', 'Method', 'Weight', 'City']] ) # 'w102', 'w208', 'w310', 'w311x', 'w312', 'w339', 'w510c', 'method'

        if uid:
            datalist_UID.append(data[data['UID'] == uid])

    print('Done reading, concat now.')
    women = pd.concat(datalist)

    if uid:
        pd.concat(datalist_UID).transpose().to_csv(f'{uid}.csv')


    women = women. \
        set_index(['UID', 'Wave']). \
        sort_index()

    store['women'] = women

store.close()

# Keep only women seen in all three waves (reduces data to 3 of 6 cities)
nRecordsPerWoman = women.groupby(['UID']).size()
women3 = women.loc[nRecordsPerWoman[nRecordsPerWoman==3].index]


print(women.head())
bardat = women.reset_index()
bardat['Wave'] = bardat['Wave'].astype(str)
#sns.countplot(data=women.reset_index(), x='Method', hue='Wave')
g = sns.catplot(data=women.reset_index(), x='Method', hue='Wave', col='City', kind='count', height=4, aspect=0.7, legend_out=False, sharex=True, sharey=False) # , col_wrap=3
plt.suptitle('Method by Wave')
g.set_xticklabels(rotation=45, horizontalalignment='right') # , fontsize='x-large'
#g.legend(loc='upper right')
#plt.legend(loc='upper right')#,bbox_to_anchor=(1,0.5))
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

fig, ax = plt.subplots()
methods = women3['Method'].unique()

# Define switching matrix.  Rows are FROM, columns are TO
switching = pd.DataFrame(index=methods, columns=methods).fillna(0)

def extract_switches(w):
    uid = w.index.get_level_values('UID').values[0]
    waves = w.index.get_level_values('Wave').values
    for wave in range(3):
        if wave in waves and wave+1 in waves:
            frm = w.loc[(uid, wave), 'Method']
            to = w.loc[(uid, wave+1), 'Method']
            weight = w.loc[(uid, wave), 'Weight']
            switching.loc[frm, to] += weight # Use weights

women3.groupby('UID').apply(extract_switches) # Fills switching matrix

# Normalize by row-sum (FROM)
title = 'Senegal longitudinal switching'
if normalize_by_from:
    switching = switching.div(switching.sum(axis=1), axis=0)
    title += 'normalize by FROM'

sns.heatmap(switching, square=True, cmap='jet', xticklabels=methods, yticklabels=methods, ax=ax)
plt.xlabel('TO')
plt.ylabel('FROM')
plt.suptitle(title)
plt.xticks(rotation=45, horizontalalignment='right') # , fontsize='x-large'
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

exit()

import numpy as np
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

_age = np.linspace(women3['Age'].min(), women3['Age'].max())
_parity = np.linspace(women3['Parity'].min(), women3['Parity'].max())

print(_age)
print(_parity)
exit()
_xx, _yy = np.meshgrid(_x, _y)
x, y = _xx.ravel(), _yy.ravel()

tmp = women3.reset_index().groupby(['Age', 'Parity']).size().sort_index().reset_index()

#top = women3[]
bottom = np.zeros_like(top)
width = depth = 1

ax.bar3d(x, y, bottom, width, depth, top, shade=True)
ax.set_title('Shaded')


plt.show()
