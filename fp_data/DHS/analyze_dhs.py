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
    'dklein': '/home/dklein/sdb2/Dropbox (IDM)/FP Dynamic Modeling/DHS/Country data/Senegal/',
    'cliffk': '/u/cliffk/idm/fp/data/DHS/NGIR6ADT/NGIR6AFL.DTA',
}

yeardict = {
    '2017':     os.path.join('2017',    'SNIR7ZDT', 'SNIR7ZFL.DTA'),
    '2016':     os.path.join('2016',    'SNIR7IDT', 'SNIR7IFL.DTA'), # SNIRG0FL
    '2015':     os.path.join('2015',    'SNIR7HDT', 'SNIR7HFL.DTA'),
    '2014':     os.path.join('2014',    'SNIR70DT', 'SNIR70FL.DTA'), # SNIR6RFL
    '2012-13':  os.path.join('2012-13', 'SNIR6DDT', 'SNIR6DFL.DTA'),
    '2010-11':  os.path.join('2010-11', 'SNIR61DT', 'SNIR61FL.DTA'),
    '2005':     os.path.join('2005',    'SNIR4HDT', 'SNIR4HFL.dta'),
    #'1999':     os.path.join('1999',    'SNIQ41DT', 'SNIQ41FL.DTA'),
    '1997':     os.path.join('1997',    'SNIR32DT', 'SNIR32FL.DTA'),
    '1992-93':  os.path.join('1992-93', 'SNIR21DT', 'SNIR21FL.DTA'),
    '1986':     os.path.join('1986',    'SNIR02DT', 'SNIR02FL.DTA'), # Missing v025
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
    'v313': "Currently using a modern method",
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
    replace_dict = {k: values[k.upper()] if k.upper() in values else values[k] for k in indicators.keys() if k in values or k.upper() in values}
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

    if False:
        values = pd.io.stata.StataReader(filename).value_labels()
        codebook = pd.io.stata.StataReader(filename).variable_labels()

        pd.DataFrame({'keys': list(codebook.keys()), 'values': list(codebook.values())}).set_index('keys').to_csv(f'codebook_{fn}.csv')

    return data

def read():
    with Pool(4) as p:
        data_list = p.map(load, yeardict.items())

    data = pd.concat(data_list)
    data.to_hdf(cachefn, key='data', format='t')

    return data


def cmc_to_year(data):
    v007 = data['v007']
    cmc = data['v008']
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
        data = read()
    else:
        try:
            data = pd.read_hdf(cachefn, key='data')
        except:
            data = read()

    # Figure out the mean time point of each survey in years - this is remarkably slow!
    data['Date'] = data.apply(cmc_to_year, axis=1)
    year_to_date = data.groupby('SurveyYear').apply( partial(wmean, value='Date', weight='v005') )
    year_to_date.name = 'Date'

    urhi_like = data.loc[ (data['v101'].isin(['dakar', 'kaolack', u'thies'])) & (data['v102'] == 'urban') ]

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


    # CURRENTLY PREGNANT
    fig, ax = plt.subplots(1,2, figsize=fs)
    boolean_plot('Currently pregnant', 'v213', ax=ax[0])
    boolean_plot('Currently pregnant (URHI-like)', 'v213', data=urhi_like, ax=ax[1])
    plt.tight_layout()

    boolean_plot_by('Currently pregnant', 'v213', 'v101')
    boolean_plot_by('Currently pregnant', 'v213', 'v102')

    multi_plot('Unmet need', 'v624')

    #boolean_plot_by('Currently pregnant', 'v213', 'v013') # By age?!

    fig, ax = plt.subplots(1,2, figsize=fs)
    multi_plot('Method type', 'v313', ax=ax[0])
    multi_plot('Method type (URHI-like)', 'v313', data=urhi_like, ax=ax[1])
    plt.tight_layout()

    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', default=False, action='store_true')
    args = parser.parse_args()

    main(force_read = args.force)


#print(pd.crosstab(data['SurveyYear'], data['v102'], data['v213']*data['v005']/1e6, aggfunc=sum))
#with pd.option_context('display.precision', 1, 'display.max_rows', 1000): # 'display.precision',2, 
