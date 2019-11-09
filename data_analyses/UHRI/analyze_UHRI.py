import os
import seaborn as sns
sns.set()
import pylab as pl
import numpy as np
import pandas as pd
import sciris as sc

cachefn = 'store.hdf'
store = pd.HDFStore(cachefn)

force_read = True
normalize_by_from = True
exclude_missing_parity = True
write_codebooks = False
do_plot = False

if (not force_read) and os.path.isfile(cachefn) and 'women' in store:
    women = store['women']
else:
    username = os.path.split(os.path.expanduser('~'))[-1]
    folderdict = {'dklein': '/home/dklein/Dropbox (IDM)/URHI/Senegal',
                  'cliffk': '/home/cliffk/idm/fp/data/Senegal',
                 }
    try:
        basedir = folderdict[username]
    except:
        raise Exception(f'User {username} not found among users {list(folderdict.keys())}, cannot find data folder')

    filenames = [
        os.path.join(basedir, 'Baseline', 'SEN_base_wm_20160427.dta'),
        os.path.join(basedir, 'Midline', 'SEN_mid_wm_match_20160609.dta'),
        os.path.join(basedir, 'Endline', 'SEN_end_wm_match_20160505.dta')
    ]

    datalist = []
    datalist_UID = []
    for wave, filename in enumerate(filenames):
        print('-'*80)
        print(f'On wave {wave}, reading {filename}')
        data = pd.read_stata(filename, convert_categoricals=False)

        values = pd.io.stata.StataReader(filename).value_labels()
        if write_codebooks:
            codebook = pd.io.stata.StataReader(filename).variable_labels()
            pd.DataFrame({'keys': list(codebook.keys()), 'values': list(codebook.values())}).set_index('keys').to_csv(f'codebook_{wave}.csv')

        data['UID'] = data['location_code'].astype(str) + '.' + data['hhnum'].astype(str) + '.' + data['line'].astype(str)
        data['Wave'] = wave

        # w102 is age, w208 is parity
        # 310 current use of contraception
        if wave == 0:
            cols = {'w102':'Age', 'w208':'Parity', 'method':'Method', 'wm_allcity_wt':'Weight', 'city':'City', 'unmet_cmw': 'Unmet'}
            city_key = 'city'
        elif wave == 1:
            cols = {'mw102':'Age', 'mw208b':'Parity', 'mmethod':'Method', 'wm_allcity_wt':'Weight', 'mcity':'City', 'munmet_cmw': 'Unmet'}
            city_key = 'MCITY'
        elif wave == 2:
            cols = {'ew102':'Age', 'ew208':'Parity', 'emethod':'Method', 'ewoman_weight_6city':'Weight', 'ecity':'City', 'eunmet_cmw': 'Unmet'}
            city_key = 'ECITY'

        for c in cols.keys():
            if c not in data.columns:
                raise Exception(f'Cannot find column {c} at wave {wave}!')

        data = data \
            .rename(columns=cols) \
            .replace({
                'Method': values['method'],
                'City': values[city_key],
                'Unmet': values['unmet_cmw'],
            })

        datalist.append( data[['UID', 'Wave', 'Age', 'Parity', 'Method', 'Weight', 'City', 'Unmet']] ) # 'w102', 'w208', 'w310', 'w311x', 'w312', 'w339', 'w510c', 'method'

    print('Done reading, concat now.')
    women = pd.concat(datalist)

    women = women. \
        set_index(['UID', 'Wave']). \
        sort_index()
        
    # Fix inconsistencies
    women = women.replace({'City':'Guédiawaye'}, 'Guediawaye')
    women = women.replace({'Unmet':np.nan}, 'Missing') # Think this is right?
    if exclude_missing_parity:
        original_size = women.shape[0]
        women = women.dropna(subset=['Parity'])
        new_size= women.shape[0]
        print(f'Dropped {original_size-new_size} rows out of {original_size} due to missing parity data')
    print(women['Parity'].unique())

    store['women'] = women

store.close()

print(f'Women data contains {women.shape[0]:,} rows.')
print('\nDescription by city:')
print(women.groupby('City').describe().T)
print('\nCrosstab of Method and Unmet need, values are weight sums:')
print(pd.pivot_table(women, index='Method', columns='Unmet', values='Weight', aggfunc=sum))

###############################################################################
# Data cleaning and variations ################################################
###############################################################################

method_mapping = {
    'No method': 'No method', # Unmet?  Not quite...

    'Daily pill': 'Short-lasting',
    'Male condom': 'Short-lasting',
    'Female condom': 'Short-lasting',

    'Injectables': 'Injections',

    'Implants': 'Long-lasting',
    'iucd': 'Long-lasting',
    'Female sterilization': 'Long-lasting',

    # Some of these could be unmet?
    'sdm': 'Other',
    'Natural methods': 'Other',
    'Breastfeeding/LAM': 'Other',
    'Other traditional method': 'Other',
    'Emergency pill': 'Other',
    'Other modern method': 'Other',
}
women['MethodClass'] = women['Method'].replace(method_mapping)

age_edges = list(range(15,55,5)) + [99]
women['AgeBin'] = pd.cut(women['Age'], bins = age_edges, right=False)

parity_edges = list(range(6+1)) + [99]
women['ParityBin'] = pd.cut(women['Parity'], bins = parity_edges, right=False)

# Keep only women seen in all three waves (reduces data to 3 of 6 cities)
nRecordsPerWoman = women.groupby(['UID']).size()
women3 = women.loc[nRecordsPerWoman[nRecordsPerWoman==3].index]


###############################################################################
# PLOT: Method mix ############################################################
###############################################################################
data = women.copy(deep=True)
bardat = data.reset_index()
bardat['Wave'] = bardat['Wave'].astype(str)
#sns.countplot(data=data.reset_index(), x='Method', hue='Wave')
g = sns.catplot(data=data.reset_index(), x='Method', hue='Wave', col='City', kind='count', height=4, aspect=0.7, legend_out=False, sharex=True, sharey=False) # , col_wrap=3
pl.suptitle('Method by Wave')
g.set_xticklabels(rotation=45, horizontalalignment='right') # , fontsize='x-large'
#g.legend(loc='upper right')
#pl.legend(loc='upper right')#,bbox_to_anchor=(1,0.5))
pl.tight_layout(rect=[0, 0.03, 1, 0.95])


###############################################################################
# PLOT: Method switching ######################################################
###############################################################################
data = women3.copy(deep=True)
fig, ax = pl.subplots()
methods = data['Method'].unique()

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

data.groupby('UID').apply(extract_switches) # Fills switching matrix

# Normalize by row-sum (FROM)
print('WARNING, seems to be something wrong with the switching data...!')
title = 'Senegal longitudinal switching'
if normalize_by_from:
    switching = switching.div(switching.sum(axis=1), axis=0)
    title += ' normalize by FROM'

sns.heatmap(switching, square=True, cmap='jet', xticklabels=methods, yticklabels=methods, ax=ax)
pl.xlabel('TO')
pl.ylabel('FROM')
pl.suptitle(title)
pl.xticks(rotation=45, horizontalalignment='right') # , fontsize='x-large'
pl.tight_layout(rect=[0, 0.03, 1, 0.95])


###############################################################################
# PLOT: Stacked bar ###########################################################
###############################################################################
data = women.copy(deep=True)
pivot_by_age = data \
    .groupby(['AgeBin', 'MethodClass'])['Weight'].sum() \
    .reset_index() \
    .pivot(index='AgeBin', columns='MethodClass', values='Weight')
pivot_by_age.plot.bar(stacked=True, figsize=(10,10))

pivot_by_parity = data \
    .groupby(['ParityBin', 'MethodClass'])['Weight'].sum() \
    .reset_index() \
    .pivot(index='ParityBin', columns='MethodClass', values='Weight')
pivot_by_parity.plot.bar(stacked=True, figsize=(10,10))

###############################################################################
# PLOT: Skyscraper ############################################################
###############################################################################
def skyscraper(data, label, ax=None):
    if ax == None:
        fig = pl.figure(figsize=(10,10))
        ax = fig.add_subplot(projection='3d')
    ax.view_init(elev=37, azim=-31)

    data['Parity'] = data['Parity'].fillna(0).astype(int)

    data['AgeBinCode'] = data['AgeBin'].cat.codes
    data['ParityBinCode'] = data['ParityBin'].cat.codes
    age_bin_codes = np.array(sorted(list(data['AgeBinCode'].unique())))
    parity_bin_codes = np.array(sorted(list(data['ParityBinCode'].unique())))

    age_mesh, parity_mesh = np.meshgrid(age_bin_codes, parity_bin_codes)
    age_flat, parity_flat = age_mesh.ravel(), parity_mesh.ravel()

    age_parity = pd.DataFrame({'AgeBinCode': age_flat, 'ParityBinCode': parity_flat, 'Weight': np.zeros_like(age_flat)})
    age_parity = pd.concat([age_parity, data]).groupby(['AgeBinCode', 'ParityBinCode'])['Weight'].sum().sort_index().reset_index()

    bottom = 0
    width = depth = 0.75

    color_values = sc.vectocolor(age_parity['Weight'])

    ax.bar3d(age_parity['AgeBinCode'], age_parity['ParityBinCode'], bottom, width, depth, age_parity['Weight'], color=color_values) # , shade=True
    age_bin_labels = list(data['AgeBin'].cat.categories)
    age_bin_labels[-1] = f'{age_edges[-2]}+'
    ax.set_xlabel('Age')
    ax.set_xticks(age_bin_codes+0.5) # To center the tick marks
    ax.set_xticklabels(age_bin_labels)

    parity_bin_labels = parity_edges[:-1]
    parity_bin_labels[-1] = f'{parity_edges[-2]}+'
    ax.set_ylabel('Parity')
    ax.set_yticks(parity_bin_codes+0.5)
    ax.set_yticklabels(parity_bin_labels)

    ax.set_zlabel('Women (weighted)')
    ax.set_title(label)


sc.ax3d(silent=True) # Enable 3D plotting
nrows = 2
ncols = women['MethodClass'].nunique() // nrows +1
fig = pl.figure(figsize=(10,10))
idx = 0
for label, raw in women.groupby('MethodClass'):
    idx += 1
    data = raw.copy(deep=True) # Just to be safe
    ax = fig.add_subplot(nrows, ncols, idx, projection='3d')
    skyscraper(data, label, ax)

ax = fig.add_subplot(nrows, ncols, idx+1, projection='3d')
skyscraper(women, 'All women', ax)
pl.show()
