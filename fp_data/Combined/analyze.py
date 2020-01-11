import os
import itertools
import argparse
import numpy as np
import pandas as pd
import unidecode
import seaborn as sns
import matplotlib.pyplot as plt

from dhs import DHS
from urhi import URHI

fs=(12,8)

username = os.path.split(os.path.expanduser('~'))[-1]
folderdict = {
    'dklein': {
        'DHS': '/home/dklein/sdb2/Dropbox (IDM)/FP Dynamic Modeling/DHS/Country data/Senegal/',
        'URHI': '/home/dklein/sdb2/Dropbox (IDM)/URHI/Senegal',
    },
    'cliffk': {
        'DHS': '/u/cliffk/idm/fp/data/DHS/NGIR6ADT/NGIR6AFL.DTA',
        'URHI': '/home/cliffk/idm/fp/data/Senegal'
    }
}


def plot_line(*args, **kwargs):
    data = kwargs.pop('data')
    by = kwargs.pop('by')
    tmp = data.groupby(['Survey', 'SurveyName', 'Date', by])['Weight'].sum().reset_index(by)
    weight_sum = data.groupby(['Survey', 'SurveyName', 'Date'])['Weight'].sum()
    tmp['Percent'] = 100*tmp['Weight'].divide(weight_sum)

    # Ugh, have to make all levels have a value to avoid color or linetype errors
    tmp = tmp.set_index(by, append=True).unstack(fill_value=-1).stack().reset_index()#.sort_values(['Survey', by, 'Date'])
    tmp.loc[tmp['Percent']<0,'Percent'] = np.NaN

    sns.lineplot(data=tmp, x='Date', y='Percent', hue=by, style='Survey', **kwargs)


def plot_pie(*args, **kwargs):
    data = kwargs.pop('data')
    by = kwargs.pop('by')
    ans = data.groupby(by)['Weight'].sum()
    plt.pie(ans.values, labels=ans.index.tolist(), labeldistance=None, explode=0.1*np.ones_like(ans.values))


def main(force_read = False):
    d = DHS(folderdict[username]['DHS'], force_read)
    print(d.data['MethodType'].unique())
    u = URHI(folderdict[username]['URHI'], force_read)
    print(u.data['MethodType'].unique())

    # Useful to see which methods are classified as modern / traditional
    print(pd.crosstab(index=d.data['Method'], columns=d.data['MethodType'], values=d.data['Weight']/1e6, aggfunc=sum))
    print(pd.crosstab(index=u.data['Method'], columns=u.data['MethodType'], values=u.data['Weight'], aggfunc=sum))


    cols = ['Survey', 'SurveyName', 'AgeBin', 'AgeBinCoarse', 'Method', 'MethodType', 'Weight', 'Date', 'ParityBin']
    #c = pd.concat((d.dakar_urban[cols], u.data[cols]))
    c = pd.concat((d.urhi_like[cols], u.data[cols]))
    #c = pd.concat((d.data[cols], u.data[cols]))

    plot_line(data=c, by='MethodType')

    # Take out Midline beacuse it's weird
    c_no_midline = c.loc[c['SurveyName']!='Midline']
    g = sns.FacetGrid(data=c_no_midline, col='AgeBinCoarse', height=5)
    g.map_dataframe(plot_line, by='MethodType').add_legend().set_xlabels('Year').set_ylabels('Percent')
    #g.savefig(os.path.join(results_dir, f'BirthSpacing_year_order.png'))

    g = sns.FacetGrid(data=c_no_midline, col='ParityBin', col_wrap=4, height=3)
    g.map_dataframe(plot_line, by='MethodType').add_legend().set_xlabels('Year').set_ylabels('Percent')
    #g.savefig(os.path.join(results_dir, f'BirthSpacing_year_order.png'))


    dat2011 = pd.concat( [
        #d.data[cols].loc[d.data['SurveyName']=='2010-11'],
        d.urhi_like[cols].loc[d.urhi_like['SurveyName']=='2010-11'], # URHI-like
        #d.dakar_urban[cols].loc[d.dakar_urban['SurveyName']=='2010-11'], # Dakar-urban
        u.data[cols].loc[u.data['SurveyName']=='Baseline']
    ])
    dat2011['Year'] = 2011

    dat2015 = pd.concat( [
        #d.data[cols].loc[d.data['SurveyName']=='2010-11'],
        d.urhi_like[cols].loc[d.urhi_like['SurveyName']=='2015'], # URHI-like
        #d.dakar_urban[cols].loc[d.dakar_urban['SurveyName']=='2010-11'], # Dakar-urban
        u.data[cols].loc[u.data['SurveyName']=='Endline']
    ])
    dat2015['Year'] = 2015

    dat = pd.concat([dat2011, dat2015])

    g = sns.FacetGrid(data=dat, col='Survey', row='Year', height=5)
    g.map_dataframe(plot_pie, by='MethodType').add_legend()#.set_xlabels('Year').set_ylabels('Percent')

    mod = dat.loc[dat['MethodType']=='Modern']
    g = sns.FacetGrid(data=mod, col='Survey', row='Year', height=5)
    g.map_dataframe(plot_pie, by='Method').add_legend()#.set_xlabels('Year').set_ylabels('Percent')

    plt.show()

    exit()



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
    skyscraper(data.loc[data['SurveyName']!='Midline'], 'URHI')

    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', default=False, action='store_true')
    args = parser.parse_args()

    main(force_read = args.force)


#print(pd.crosstab(data['SurveyName'], data['v102'], data['v213']*data['Weight']/1e6, aggfunc=sum))
#with pd.option_context('display.precision', 1, 'display.max_rows', 1000): # 'display.precision',2, 
