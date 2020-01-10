import os
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
    #'dklein': os.path.join( os.getenv("HOME"), 'Dropbox (IDM)', 'FP Dynamic Modeling', 'DHS', 'Country data', 'Nigeria', '2013', 'NGIR6ADT', 'NGIR6AFL.DTA'),
    'dklein': '/home/dklein/Dropbox (IDM)/FP Dynamic Modeling/DHS/Country data/Senegal/',
    'cliffk': '/u/cliffk/idm/fp/data/DHS/NGIR6ADT/NGIR6AFL.DTA',
}

yeardict = {
    '1986':     os.path.join('1986',    'SNIR02DT', 'SNIR02FL.DTA'), # Missing v025
    '1992-93':  os.path.join('1992-93', 'SNIR21DT', 'SNIR21FL.DTA'),
    '1997':     os.path.join('1997',    'SNIR32DT', 'SNIR32FL.DTA'),
    #'1999':     os.path.join('1999',    'SNIQ41DT', 'SNIQ41FL.DTA'),
    '2005':     os.path.join('2005',    'SNIR4HDT', 'SNIR4HFL.dta'),
    '2010-11':  os.path.join('2010-11', 'SNIR61DT', 'SNIR61FL.DTA'),
    '2012-13':  os.path.join('2012-13', 'SNIR6DDT', 'SNIR6DFL.DTA'),
    '2014':     os.path.join('2014',    'SNIR70DT', 'SNIR70FL.DTA'), # SNIR6RFL
    '2015':     os.path.join('2015',    'SNIR7HDT', 'SNIR7HFL.DTA'),
    '2016':     os.path.join('2016',    'SNIR7IDT', 'SNIR7IFL.DTA'), # SNIRG0FL
    '2017':     os.path.join('2017',    'SNIR7ZDT', 'SNIR7ZFL.DTA'),
}

indicators = {
    'v005': "Weight",
    'v007': "Year of survey",
    #'v012': "Respondent's current age",
    'v013': "Age in 5-year groups",
    'v008': "Date of interview (cmc)",
    'v101': "De facto region of residence",
    'v102': "De facto type of place of residence",
    #'v106': "Highest educational level",
    'v190': "wealth index combined",
    'v201': "total children ever born",
    'v213': "Currently pregnant",
    'v225': "current pregnancy wanted",
    'v228': "ever had a terminated pregnancy",
    'v301': "knowledge of any method # Lists most sophisticate method type",
    #'v304 - v307 is the contraceptive table, interesting!",
    'v312': "current contraceptive method",
    'v313': "current use by method type",
    'v317': "date of start of use of method (cmc)",
    'v364': "contraceptive use and intention",
    'v367': "wanted last child",
    'v384a': "heard family planning on radio last few months",
    'v384b': "heard family planning on tv last few months",
    'v384c': "heard family planning in newspaper/magazine last few months",
        #'v384d': "heard family planning by text messages on mobile phone", # Missing in 1984!
    #'v393': "visited by fieldworker in last 12 months",
    #'v393a': "did fieldworker talk about family planning",
    #'v394': "visited health facility last 12 months",
    #'v395': "at health facility, told of family planning",
    'v3a08a': "reason not using: not married",
    'v3a08b': "reason not using: not having sex",
    'v3a08c': "reason not using: infrequent sex",
    'v3a08d': "reason not using: menopausal/hysterectomy",
    'v3a08e': "reason not using: subfecund/infecund",
    'v3a08f': "reason not using: postpartum amenorrheic",
    'v3a08g': "reason not using: breastfeeding",
    'v3a08h': "reason not using: fatalistic",
    'v3a08i': "reason not using: respondent opposed",
    'v3a08j': "reason not using: husband/partner opposed",
    'v3a08k': "reason not using: others opposed",
    'v3a08l': "reason not using: religious prohibition",
    'v3a08m': "reason not using: knows no method",
    'v3a08n': "reason not using: knows no source",
    'v3a08o': "na-reason not using: health concerns",
    'v3a08p': "reason not using: fear of side effects/health concerns",
    'v3a08q': "reason not using: lack of access/too far",
    'v3a08r': "reason not using: costs too much",
    'v3a08s': "reason not using: inconvenient to use",
    'v3a08t': "reason not using: interferes with body's processes",
    'v3a08u': "reason not using: preferred method not available",
    'v3a08v': "reason not using: no method available",
    'v3a08w': "na-reason not using: cs",
    'v3a08aa': "na-reason not using: cs",
    'v3a08ab': "na-reason not using: cs",
    'v3a08ac': "na-reason not using: cs",
    'v3a08ad': "na-reason not using: cs",
    'v3a08x': "reason not using: other",
    'v3a08z': "reason not using: don't know",

    'v404': "currently breastfeeding",
    'v405': "currently amenorrheic",
    'v406': "currently abstaining",

    'v501': "current marital status",
    'v525': "age at first sex",
    'v602': "fertility preference",
    #'v603': "preferred waiting time for birth of a/another child", # Requires special handling because some -1, some numeric, and some from dictionary {996: 'non-numeric', 997: 'inconsistent', 998: "don't know"}
    'v604': "preferred waiting time for birth of a/another child (grouped)",
    'v605': "desire for more children",
    #'v613': "ideal number of children", # Requires special handling because some numeric and others from {96: 'as god wills', 97: 'other', 98: 'don t know'}
    'v614': "ideal number of children (grouped)",

    'v623': "exposure",
    'v624': "unmet need",
    'v625': "exposure (definition 2)",
    'v626': "unmet need (definition 2)",
    'v625a': "exposure to need for contraception (definition 3)",
    'v626a': "unmet need for contraception (definition 3)",

    # vcol_1 vcol_2 vcol_3 vcol_4 vcol_5 vcol_6 vcol_7 vcol_8 vcol_9 vcal_1 vcal_2 vcal_3 vcal_4 vcal_5 vcal_6 vcal_7 vcal_8 vcal_9

    'b3_01': 'date of birth (cmc)',
    'b3_02': 'date of birth (cmc)',
    'b3_03': 'date of birth (cmc)',
    'b3_04': 'date of birth (cmc)',
    'b3_05': 'date of birth (cmc)',
    'b3_06': 'date of birth (cmc)',
    'b3_07': 'date of birth (cmc)',
    'b3_08': 'date of birth (cmc)',
    'b3_09': 'date of birth (cmc)',
    'b3_10': 'date of birth (cmc)',
    'b3_11': 'date of birth (cmc)',
    'b3_12': 'date of birth (cmc)',
    'b3_13': 'date of birth (cmc)',
    'b3_14': 'date of birth (cmc)',
    'b3_15': 'date of birth (cmc)',
    'b3_16': 'date of birth (cmc)',
    'b3_17': 'date of birth (cmc)',
    'b3_18': 'date of birth (cmc)',
    'b3_19': 'date of birth (cmc)',
    'b3_20': 'date of birth (cmc)',
}

Path("cache").mkdir(exist_ok=True)
cachefn = os.path.join('cache', 'senegal.hdf')

results_dir = 'results'
Path(results_dir).mkdir(exist_ok=True)

try:
    foldername = folderdict[username]
except:
    raise Exception(f'User {username} not found among users {list(filedict.keys())}, cannot find data.')


def load(x):
    year, path = x
    filename = os.path.join(foldername, path)
    print(f'Loading {year} from file {filename}')

    fn = Path(filename).resolve().stem
    print(f'File: {filename} ...')
    data = pd.read_stata(filename, convert_categoricals=False)

    data['SurveyYear'] = year
    found_keys = []
    for k in indicators.keys():
        if k not in data.columns:
            print(f'Survey {year} is missing {k}')
        else:
            found_keys.append(k)
    data = data[['SurveyYear'] + found_keys]

    values = pd.io.stata.StataReader(filename).value_labels()
    replace_dict = {k: values[k.upper()] if k.upper() in values else values[k] for k in found_keys if k in values or k.upper() in values}
    for k in replace_dict.keys():
        #data[k] = data[k].replace(replace_dict[k]).astype('category')
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

    if False: # TODO: Make argparse flag
        values = pd.io.stata.StataReader(filename).value_labels()
        codebook = pd.io.stata.StataReader(filename).variable_labels()

        pd.DataFrame({'keys': list(codebook.keys()), 'values': list(codebook.values())}).set_index('keys').to_csv(f'codebook_{fn}.csv')

    return data

def read():
    with Pool(16) as p:
        data_list = p.map(load, yeardict.items())

    data = pd.concat(data_list)
    data.to_hdf(cachefn, key='data', format='t')

    # Build barriers from reasons for non-use questions
    keys = [k for k in indicators.keys() if k[:5] == 'v3a08']
    barriers = pd.DataFrame(index=yeardict.keys(), columns=keys).fillna(0)
    print(data[keys].describe())
    for year, dat in data.groupby('SurveyYear'):
        ret = {}
        wsum = dat['v005'].sum()
        for k in keys:
            #print(year, k, dat[k].unique())  # nan, no, yes, not married

            tmp = dat[[k,'v005']].dropna()
            if tmp.shape[0] == 0:
                print('NaN', year, k, indicators[k], dat[k].unique())
                continue
            ans = 100 * np.dot(~(tmp[k] == 'no'), tmp['v005']) / tmp['v005'].sum()
            barriers.loc[year, k] = ans
            print('%.2f'%ans, year, k, indicators[k], dat[k].unique())


    new_colname_dict = {k: v.split(':')[1][1:] for k,v in indicators.items() if k in keys}
    barriers.rename(columns=new_colname_dict, inplace=True)
    barriers = barriers.stack()
    barriers.name = 'Percent'
    barriers = barriers.reset_index().rename(columns={'level_0':'SurveyYear','level_1':'Barrier'})

    barriers.to_hdf(cachefn, key='barriers', format='t')

    return data, barriers


import math
def cmc_to_year(data, cmc_col='v008'): # v008 is date of interview
    v007 = data['v007']
    cmc = data[cmc_col]
    if math.isnan(cmc):
        return np.nan
    if v007 < 100:
        year = 1900 + int((cmc-1)/12)
        month = cmc - (v007*12)
    elif v007 >= 2000:
        year = 1900 + int((cmc-1)/12)
        month = cmc - ((v007-1900)*12)
    else:
        raise Exception('Help!')
    return year + month/12


def wmean(data, value, weight):
    if data[value].dtype.name == 'category':
        return  np.sum(data[weight] * data[value].cat.codes) / np.sum(data[weight])
    return  np.sum(data[weight] * data[value]) / np.sum(data[weight])


def main(force_read = False):

    if force_read:
        data, barriers = read()
    else:
        try:
            data = pd.read_hdf(cachefn, key='data')
            barriers = pd.read_hdf(cachefn, key='barriers')
        except:
            data, barriers = read()

    ''''
    # Extracting birth spacing
    print(data.iloc[0])
    for i in range(1,20+1):
        # TODO: Use CMC directly as it's in months... maybe
        #data[f'b3_{i:02}y'] = data[['v007']].apply(partial(cmc_to_year, cmc_col = f'b3_{i:02}'), axis=1)
        year = 1900 + (data[f'b3_{i:02}']-1)//12
        data[f'b3_{i:02}y'] = year + data[f'b3_{i:02}']/12 - (year-1900)
        print(f'b3_{i:02}', data.iloc[0][[f'b3_{i:02}', f'b3_{i:02}y']])
    exit()
    '''

    # Shows method classification
    print(pd.crosstab(data['v312'], data['v313'], values=data['v005']/1e6, aggfunc=sum))

    # Figure out the mean time point of each survey in years - this is remarkably slow!
    data['Date'] = data.apply(cmc_to_year, axis=1)
    year_to_date = data.groupby('SurveyYear').apply( partial(wmean, value='Date', weight='v005') )
    year_to_date.name = 'Date'

    dakar_urban = data.loc[ (data['v101'].isin(['dakar'])) & (data['v102'] == 'urban') ]
    urhi_like = data.loc[ (data['v101'].isin(['west', 'dakar', 'kaolack', 'thies'])) & (data['v102'] == 'urban') ]

    urban = data.loc[ (data['v102'] == 'urban') ]
    rural = data.loc[ (data['v102'] == 'rural') ]



    def boolean_plot(name, value, data=data, ax=None):
        gb = data.groupby(['SurveyYear'])
        weighted = 100 * gb.apply( partial(wmean, value=value, weight='v005') )
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
        weighted = 100 * gb.apply( partial(wmean, value=value, weight='v005') )
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
        unstacked = data.groupby(['SurveyYear', value])['v005'].sum().unstack(value)
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


    def age_pyramid_plot(name, data, ax=None):
        age_pyramid = data.groupby(['SurveyYear', 'v013'])['v005'].sum()
        year_sum = data.groupby(['SurveyYear'])['v005'].sum()
        age_pyramid = age_pyramid.divide(year_sum)
        age_pyramid.name = 'Percent'
        if ax == None:
            fig, ax = plt.subplots(figsize=fs)
        g = sns.catplot(x='Percent', y='v013', hue='SurveyYear', data=age_pyramid.reset_index(), kind='point', ax=ax)
        for a in g.axes.flat:
            a.invert_yaxis()
        g.fig.set_size_inches(fs[0], fs[1], forward=True)
        g.fig.suptitle(f'{name}')

        fn = name.replace(" ","_")
        age_pyramid.to_csv(os.path.join(results_dir, f'{fn}.csv'))
        plt.savefig(os.path.join(results_dir, f'{fn}.png'))

    def skyscraper(data, name, savefig=True, savedata=True):
        age_parity = data.groupby(['SurveyYear', 'v013', 'v201'])['v005'].sum()
        total = data.groupby(['SurveyYear'])['v005'].sum()
        age_parity = 100 * age_parity.divide(total).fillna(0)
        age_parity.name = 'Percent'
        fig, ax = plt.subplots(2,5, figsize=fs)
        for i, (year, d) in enumerate(age_parity.groupby('SurveyYear')):
            row = i//5
            col = i%5
            age_bins = d.index.get_level_values('v013').unique().tolist()
            parity_bins = d.index.get_level_values('v201').unique().tolist()

            X = d.unstack('v013')

            ax[row,col].imshow(X, aspect='auto', cmap='jet', interpolation='none', origin='lower')
            ax[row,col].set_title(year)
            ax[row,col].set_xticks(range(len(age_bins)))
            if row == 1:
                ax[row,col].set_xticklabels(age_bins, rotation=90)
            else:
                ax[row,col].set_xticklabels([])

            ax[row,col].set_yticks(range(len(parity_bins)))
            if col == 0:
                ax[row,col].set_yticklabels(parity_bins)
            else:
                ax[row,col].set_yticklabels([])
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.suptitle(name)
        if savefig:
            plt.savefig(os.path.join(results_dir, f'Skyscrapers-{name}.png'))
        if savedata:
            age_parity.to_csv(os.path.join(results_dir, f'Skyscrapers-{name}.csv'))
        return fig

    def plot_pie(title, data):
        fig, ax = plt.subplots(1,len(yeardict),figsize=(16,6))
        N = data['SurveyYear'].nunique()
        #for i, (sy, dat) in enumerate(tmp.groupby('SurveyYear')):
        for i, sy in enumerate(yeardict.keys()): # Keep in order
            dat = data.loc[data['SurveyYear']==sy]
            ans = dat.groupby('v312')['v005'].sum()
            ax[i].pie(ans.values, labels=ans.index.tolist())
            ax[i].set_title(f'{title}: {sy}')
        plt.tight_layout()


    plot_pie('All women', data)
    plt.savefig(os.path.join(results_dir, f'Pie-All.png'))

    tmp = data.loc[data['v313']=='modern method']
    #tmp['v313'] = tmp.v313.cat.remove_unused_categories()
    print(tmp.groupby('v312')['v005'].sum())
    plot_pie('Modern', tmp)
    plt.savefig(os.path.join(results_dir, f'Pie-Modern.png'))


    # CURRENTLY PREGNANT
    fig, ax = plt.subplots(1,3, figsize=fs)
    boolean_plot('Currently pregnant', 'v213', ax=ax[0])
    boolean_plot('Currently pregnant (dakar-urban)', 'v213', data=dakar_urban, ax=ax[1])
    boolean_plot('Currently pregnant (URHI-like)', 'v213', data=urhi_like, ax=ax[2])
    plt.tight_layout()

    boolean_plot_by('Currently pregnant', 'v213', 'v101')
    boolean_plot_by('Currently pregnant', 'v213', 'v102')

    multi_plot('Unmet need', 'v624')

    fig, ax = plt.subplots(1,4, sharey=True, figsize=fs)
    multi_plot('Method type', 'v313', ax=ax[0])
    multi_plot('Method type (URHI-like)', 'v313', data=urhi_like, ax=ax[1])
    multi_plot('Method type (urban)', 'v313', data=urban, ax=ax[2])
    multi_plot('Method type (rural)', 'v313', data=rural, ax=ax[3])
    plt.tight_layout()

    fig, ax = plt.subplots(1,2, sharey=True, figsize=fs)
    multi_plot('Method', 'v312', ax=ax[0])
    multi_plot('Method (URHI-like)', 'v312', data=urhi_like, ax=ax[1])
    plt.tight_layout()

    # Age pyramids
    fig, ax = plt.subplots(1,4, sharey=True, figsize=fs)
    age_pyramid_plot('Population Pyramid - All', data, ax=ax[0])
    age_pyramid_plot('Population Pyramid - URHI-Like', urhi_like, ax=ax[1])
    age_pyramid_plot('Population Pyramid - Urban', urban, ax=ax[2])
    age_pyramid_plot('Population Pyramid - Rural', rural, ax=ax[3])

    # Skyscraper images
    skyscraper(data, 'All-DHS')
    skyscraper(urhi_like, 'URHI-like')
    skyscraper(urban, 'Urban')
    skyscraper(rural, 'Rural')

    fig, ax = plt.subplots(1,1, figsize=fs)
    tmp = barriers.merge(year_to_date, on='SurveyYear')
    tmp = tmp.loc[tmp['Date'] > 2003]
    sns.lineplot(data = tmp, x='Date', y='Percent', hue='Barrier', ax=ax)
    last_date = max(year_to_date)
    last_tmp = tmp[tmp['Date']==last_date]
    for idx, row in last_tmp.iterrows():
        plt.text(last_date, row['Percent'], row['Barrier'])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'Barriers.png'))
    tmp.to_csv(os.path.join(results_dir, 'Barriers.csv'))

    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', default=False, action='store_true')
    args = parser.parse_args()

    main(force_read = args.force)


#print(pd.crosstab(data['SurveyYear'], data['v102'], data['v213']*data['v005']/1e6, aggfunc=sum))
#with pd.option_context('display.precision', 1, 'display.max_rows', 1000): # 'display.precision',2, 
