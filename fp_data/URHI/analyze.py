import os
import itertools
import argparse
from multiprocessing import Pool
from functools import partial
import numpy as np
import pandas as pd
from pathlib import Path
import unidecode
import seaborn as sns
import matplotlib.pyplot as plt

fs=(12,8)

username = os.path.split(os.path.expanduser('~'))[-1]
folderdict = {
    'dklein': '/home/dklein/sdb2/Dropbox (IDM)/URHI/Senegal',
    'cliffk': '/home/cliffk/idm/fp/data/Senegal'
}

try:
    foldername = folderdict[username]
except:
    raise Exception(f'User {username} not found among users {list(filedict.keys())}, cannot find data.')

yeardict = {
    'Baseline':     os.path.join(foldername, 'Baseline', 'SEN_base_wm_20160427.dta'),
    'Midline':     os.path.join(foldername, 'Midline', 'SEN_mid_wm_match_20160609.dta'),
    'Endline':     os.path.join(foldername, 'Endline', 'SEN_end_wm_match_20160505.dta'),
}

indicators = {
    'Baseline': {'iyear':'Year', 'imon':'Month', 'w102':'Age', 'w208':'Parity', 'method':'Method', 'methodtype':'MethodType', 'wm_allcity_wt':'Weight', 'city':'City', 'unmet_cmw': 'Unmet'},
    'Midline': {'mwiyear':'Year', 'mwimon':'Month', 'mw102':'Age', 'mw208b':'Parity', 'mmethod':'Method', 'mmethodtype':'MethodType', 'mwm_allcity_wt':'Weight', 'mcity':'City', 'munmet_cmw': 'Unmet'},
    'Endline': {'ewiyear':'Year', 'ewimon':'Month', 'ew102':'Age', 'ew208':'Parity', 'emethod':'Method', 'emethodtype':'MethodType', 'ewoman_weight_6city':'Weight', 'ecity':'City', 'eunmet_cmw': 'Unmet'},
}

Path("cache").mkdir(exist_ok=True)
cachefn = os.path.join('cache', 'urhi.hdf')

results_dir = 'results'
Path(results_dir).mkdir(exist_ok=True)


def load(x):
    year, path = x
    filename = os.path.join(foldername, path)
    print(f'Loading {year} from file {filename}')

    fn = Path(filename).resolve().stem
    print(f'File: {filename} ...')
    data = pd.read_stata(filename, convert_categoricals=False)

    data['SurveyYear'] = year
    found_keys = []
    for k in indicators[year].keys():
        if k not in data.columns:
            print(f'Survey {year} is missing {k}')
        else:
            found_keys.append(k)


    data = data[['SurveyYear'] + found_keys]

    values = pd.io.stata.StataReader(filename).value_labels()
    replace_dict = {k: values[k.upper()] if k.upper() in values else values[k] for k in found_keys if k in values or k.upper() in values}

    # Ugh
    if year == 'Midline':
        replace_dict['mmethodtype'] = values['methodtype'] # Doesn't appear to be an entry for mmethodtype?
    elif year == 'Endline':
        replace_dict['emethod'] = values['method'] # Doesn't appear to be an entry for emethod?
        replace_dict['emethodtype'] = values['methodtype'] # Doesn't appear to be an entry for emethodtype?

    for k in replace_dict.keys():
        if indicators[year][k] == 'Parity':
            print('Skipping parity')
            continue

        if 0 in replace_dict[k]:
            # zero-based
            data[k] = data[k].fillna(-1)
        else:
            # assume one-based
            data[k] = data[k].fillna(0) - 1

        try:
            data[k] = pd.Categorical.from_codes(data[k], categories = [unidecode.unidecode(v[1]) for v in sorted(replace_dict[k].items(), key = lambda x: x[0])] )
        except:
            print('Difficulty:', year, k, data[k].unique(), replace_dict[k])
            data[k] = data[k].replace(replace_dict[k]).map(str) #.astype('category')
            print(data[k])

    data.rename(columns=indicators[year], inplace=True)

    age_edges = list(range(15,55,5)) + [99]
    a,b = itertools.tee(age_edges)
    a = list(a)[:-1]
    next(b)
    labels = [f'{c}-{d}' for c,d in zip(a,b)]
    data['AgeBin'] = pd.cut(data['Age'], bins = age_edges, labels=labels, right=False)

    parity_edges = list(range(6+1)) + [99]
    a,b = itertools.tee(parity_edges)
    a = list(a)[:-1]
    next(b)
    labels = [f'{c}-{d}' for c,d in zip(a,b)]
    data['ParityBin'] = pd.cut(data['Parity'], bins = parity_edges, labels=labels, right=False)

    if True:
        values = pd.io.stata.StataReader(filename).value_labels()
        codebook = pd.io.stata.StataReader(filename).variable_labels()

        pd.DataFrame({'keys': list(codebook.keys()), 'values': list(codebook.values())}).set_index('keys').to_csv(f'codebook_{fn}.csv')

    return data

def read():
    with Pool(3) as p:
        data_list = p.map(load, yeardict.items())

    data = pd.concat(data_list)

    #data['Parity'] = data['Parity']].map(str) #Ugh
    data.to_hdf(cachefn, key='women', format='t')

    return data


def wmean(data, value, weight):
    if data[value].dtype.name == 'category':
        return  np.sum(data[weight] * data[value].cat.codes) / np.sum(data[weight])
    return  np.sum(data[weight] * data[value]) / np.sum(data[weight])


def main(force_read = False):

    if force_read:
        data = read()
    else:
        try:
            data = pd.read_hdf(cachefn, key='women')
        except:
            data = read()

    # Useful to see which methods are classified as modern / traditional
    print(pd.crosstab(index=data['Method'], columns=data['MethodType'], values=data['Weight'], aggfunc=sum))

    # Figure out the mean time point of each survey in years - this is remarkably slow!
    data['Date'] = data['Year'] + data['Month']/12
    year_to_date = data.groupby('SurveyYear').apply( partial(wmean, value='Date', weight='Weight') )
    year_to_date.name = 'Date'


    def boolean_plot(name, value, data=data, ax=None):
        gb = data.groupby(['SurveyYear'])
        weighted = 100 * gb.apply( partial(wmean, value=value, weight='Weight') )
        weighted.name = name
        weighted =  pd.merge(weighted.reset_index(), year_to_date, on='SurveyYear')
        if ax == None:
            fig, ax = plt.subplots(figsize=fs)
        sns.lineplot(data = weighted, x='Date', y=name, ax=ax)
        ax.set_title(f'{name} ({value})')

        fn = name.replace(" ","_")
        weighted.set_index('SurveyYear').to_csv(os.path.join(results_dir, f'{fn}_{value}.csv'))
        plt.savefig(os.path.join(results_dir, f'{fn}_{value}.png'))

    def boolean_plot_by(name, value, by, data=data, ax=None):
        gb = data.groupby(['SurveyYear', by])
        weighted = 100 * gb.apply( partial(wmean, value=value, weight='Weight') )
        weighted.name = name
        weighted =  pd.merge(weighted.reset_index(), year_to_date, on='SurveyYear')
        if ax == None:
            fig, ax = plt.subplots(figsize=fs)
        sns.lineplot(data = weighted, x='Date', y=name, hue=by, ax=ax)
        ax.set_title(f'{name} ({value} by {by})')

        fn = name.replace(" ","_")
        weighted.set_index(['SurveyYear', by]).to_csv(os.path.join(results_dir, f'{fn}_{value}_by_{by}.csv'))
        plt.savefig(os.path.join(results_dir, f'{fn}_{value}.png'))


    def multi_plot(name, value, data=data, ax=None):
        unstacked = data.groupby(['SurveyYear', value])['Weight'].sum().unstack(value)
        stacked = 100 * unstacked \
            .divide(unstacked.sum(axis=1), axis=0) \
            .stack()

        stacked.name = name

        stacked =  pd.merge(stacked.reset_index(), year_to_date, on='SurveyYear')
        if ax == None:
            fig, ax = plt.subplots(figsize=fs)
        sns.lineplot(data = stacked, x='Date', y=name, hue=value, ax=ax)
        ax.set_title(f'{name.replace(" ","_")} ({value})')

        fn = name.replace(" ","_")
        stacked.set_index('SurveyYear').to_csv(os.path.join(results_dir, f'{fn}_{value}.csv'))
        plt.savefig(os.path.join(results_dir, f'{fn}_{value}.png'))


    def age_pyramid_plot(name, data):
        age_pyramid = data.groupby(['SurveyYear', 'AgeBin'])['Weight'].sum()
        year_sum = data.groupby(['SurveyYear'])['Weight'].sum()
        age_pyramid = age_pyramid.divide(year_sum)
        age_pyramid.name = 'Percent'
        g = sns.catplot(x='Percent', y='AgeBin', hue='SurveyYear', data=age_pyramid.reset_index(), kind='point')
        for a in g.axes.flat:
            a.invert_yaxis()
        g.fig.set_size_inches(fs[0], fs[1], forward=True)
        g.fig.suptitle(f'{name}')

        fn = name.replace(" ","_")
        age_pyramid.to_csv(os.path.join(results_dir, f'{fn}.csv'))
        plt.savefig(os.path.join(results_dir, f'{fn}.png'))

    def skyscraper(data, name, savefig=True, savedata=True):
        age_parity = data.groupby(['SurveyYear', 'AgeBin', 'ParityBin'])['Weight'].sum()
        total = data.groupby(['SurveyYear'])['Weight'].sum()
        age_parity = 100 * age_parity.divide(total).fillna(0)
        age_parity.name = 'Percent'
        fig, ax_vec = plt.subplots(1, 2, figsize=fs)

        for i, (year, d) in enumerate(age_parity.groupby('SurveyYear')):
            ax = ax_vec[i]

            age_bins = d.index.get_level_values('AgeBin').unique().tolist()
            parity_bins = d.index.get_level_values('ParityBin').unique().tolist()

            X = d.loc[year].unstack('AgeBin').fillna(0)

            ax.imshow(X, aspect='auto', cmap='jet', interpolation='none', origin='lower')
            ax.set_title(year)
            ax.set_xticks(range(len(age_bins)))
            ax.set_xticklabels(age_bins, rotation=90)
            ax.set_yticks(range(len(parity_bins)))
            ax.set_yticklabels(parity_bins)
            ax.set_xlabel('Age Bin')
            ax.set_ylabel('Parity Bin')

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.suptitle(name)
        if savefig:
            plt.savefig(os.path.join(results_dir, f'Skyscrapers-{name}.png'))
        if savedata:
            age_parity.to_csv(os.path.join(results_dir, f'Skyscrapers-{name}.csv'))
        return fig

    def plot_pie(title, data):
        fig, ax = plt.subplots(1,3,figsize=(16,6))
        N = data['SurveyYear'].nunique()
        #for i, (sy, dat) in enumerate(tmp.groupby('SurveyYear')):
        for i, sy in enumerate(yeardict.keys()): # Keep in order
            dat = data.loc[data['SurveyYear']==sy]
            ans = dat.groupby('Method')['Weight'].sum()
            ax[i].pie(ans.values, labels=ans.index.tolist())
            ax[i].set_title(f'{title}: {sy}')
        plt.tight_layout()



    # CURRENTLY PREGNANT
    '''
    fig, ax = plt.subplots(1,1, figsize=fs)
    boolean_plot('Currently pregnant', 'v213', ax=ax[0])
    plt.tight_layout()

    boolean_plot_by('Currently pregnant', 'v213', 'v101')
    boolean_plot_by('Currently pregnant', 'v213', 'v102')

    multi_plot('Unmet need', 'v624')

    fig, ax = plt.subplots(1,4, sharey=True, figsize=fs)
    multi_plot('Method type', 'v313', ax=ax[0])
    plt.tight_layout()
    '''

    multi_plot('ByMethod', 'Method')
    plt.tight_layout()

    plot_pie('All women', data)
    plt.savefig(os.path.join(results_dir, f'Pie-All.png'))

    tmp = data.loc[data['MethodType']=='Modern']
    tmp['Method'] = tmp.Method.cat.remove_unused_categories()
    plot_pie('Modern', tmp)
    plt.savefig(os.path.join(results_dir, f'Pie-Modern.png'))

    # Age pyramids
    age_pyramid_plot('Population Pyramid - URHI', data)

    # Skyscraper images
    skyscraper(data.loc[data['SurveyYear']!='Midline'], 'URHI')

    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', default=False, action='store_true')
    args = parser.parse_args()

    main(force_read = args.force)


#print(pd.crosstab(data['SurveyYear'], data['v102'], data['v213']*data['Weight']/1e6, aggfunc=sum))
#with pd.option_context('display.precision', 1, 'display.max_rows', 1000): # 'display.precision',2, 
