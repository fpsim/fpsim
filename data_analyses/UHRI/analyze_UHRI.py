import os
import seaborn as sns
import matplotlib.pyplot as plt # for plt.show()
import pandas as pd

cachefn = 'store.hdf'
store = pd.HDFStore(cachefn)

force_read = False

if (not force_read) and os.path.isfile(cachefn) and 'women' in store:
    women = store['women']
else:
    filenames = [
        '/home/dklein/Dropbox (IDM)/URHI/Senegal/Baseline/SEN_base_wm_20160427.dta',
        '/home/dklein/Dropbox (IDM)/URHI/Senegal/Midline/SEN_mid_wm_match_20160609.dta',
        '/home/dklein/Dropbox (IDM)/URHI/Senegal/Endline/SEN_end_wm_match_20160505.dta'
    ]

    datalist = []
    for wave, filename in enumerate(filenames):
        print('-'*80)
        print(f'On wave {wave}, reading {filename}')
        data = pd.read_stata(filename, convert_categoricals=False)

        values = pd.io.stata.StataReader(filename).value_labels()
        codebook = pd.io.stata.StataReader(filename).variable_labels()

        data['UID'] = data['location_code'].astype(str) + '.' + data['hhnum'].astype(str) + '.' + data['line'].astype(str)
        data['Wave'] = wave

        # w102 is age, w208 is parity
        datalist.append( data[['UID', 'Wave', 'w102', 'w208', 'w310', 'w311a', 'w312', 'method']] )

    print('Done reading, concat now.')
    women = pd.concat(datalist)
    nRecords = women.groupby(['UID']).size()

    women = women.set_index(['UID', 'Wave']).sort_index()

    women = women \
        .loc[nRecords[nRecords==3].index] \
        .replace({
            'method': values['method'],
            'w310': values['W310'],
            'w312': values['W312'],
        })
        #'w311a': values['W311A'],

    store['women'] = women

store.close()

print(women.head(100))

methods = women['method'].unique()

# Define switching matrix.  Rows are FROM, columns are TO
switching = pd.DataFrame(index=methods, columns=methods).fillna(0)

def extract_switches(w):
    uid = w.index.get_level_values('UID').values[0]
    waves = w.index.get_level_values('Wave').values
    for wave in range(3):
        if wave in waves and wave+1 in waves:
            frm = w.loc[(uid, wave), 'method']
            to = w.loc[(uid, wave+1), 'method']
            if frm != to:
                print(frm, to)
                print(w)
                exit()
            switching.loc[frm, to] += 1 # Use weights

women.groupby('UID').apply(extract_switches)

# Normalize by row-sum (FROM)
switching_normalized_from = switching.div(switching.sum(axis=1), axis=0)

sns.heatmap(switching_normalized_from, square=True, cmap='jet')
plt.xlabel('TO')
plt.ylabel('FROM')
plt.suptitle('Senegal longitudinal switching, normalized by FROM')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
