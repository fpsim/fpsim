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

from fp_utils.urhi import URHI
from fp_utils import plot_line, plot_pie, plot_pop_pyramid, plot_skyscraper

fs=(12,8)

username = os.path.split(os.path.expanduser('~'))[-1]
folderdict = {
    'dklein': '/home/dklein/sdb2/Dropbox (IDM)/URHI/Senegal',
    'cliffk': '/home/cliffk/idm/fp/data/Senegal'
}

def main(force_read = False):

    u = URHI(folderdict[username], force_read)


    # Useful to see which methods are classified as modern / traditional
    print(pd.crosstab(index=u.data['Method'], columns=u.data['MethodType'], values=u.data['Weight'], aggfunc=sum))

    g = sns.FacetGrid(data=u.data[['MethodType', 'Year', 'Weight', 'Survey', 'SurveyName', 'Date', 'Parity']], height=5)
    g.map_dataframe(plot_line, by='MethodType').add_legend().set_xlabels('Year').set_ylabels('Percent')
    g.savefig(os.path.join(u.results_dir, 'MethodType.png'))


    g = sns.FacetGrid(data=u.data, col='AgeBinCoarse', col_wrap=4, height=5)
    g.map_dataframe(plot_line, by='Method').add_legend().set_xlabels('Year').set_ylabels('Percent')
    g.savefig(os.path.join(u.results_dir, 'Method_by_AgeBinCoarse.png'))



    '''
    def boolean_plot(name, value, data=data, ax=None):
        gb = data.groupby(['SurveyName'])
        weighted = 100 * gb.apply( partial(wmean, value=value, weight='Weight') )
        weighted.name = name
        weighted =  pd.merge(weighted.reset_index(), year_to_date, on='SurveyName')
        if ax == None:
            fig, ax = plt.subplots(figsize=fs)
        sns.lineplot(data = weighted, x='Date', y=name, ax=ax)
        ax.set_title(f'{name} ({value})')

        fn = name.replace(" ","_")
        weighted.set_index('SurveyName').to_csv(os.path.join(results_dir, f'{fn}_{value}.csv'))
        plt.savefig(os.path.join(results_dir, f'{fn}_{value}.png'))

    def boolean_plot_by(name, value, by, data=data, ax=None):
        gb = data.groupby(['SurveyName', by])
        weighted = 100 * gb.apply( partial(wmean, value=value, weight='Weight') )
        weighted.name = name
        weighted =  pd.merge(weighted.reset_index(), year_to_date, on='SurveyName')
        if ax == None:
            fig, ax = plt.subplots(figsize=fs)
        sns.lineplot(data = weighted, x='Date', y=name, hue=by, ax=ax)
        ax.set_title(f'{name} ({value} by {by})')

        fn = name.replace(" ","_")
        weighted.set_index(['SurveyName', by]).to_csv(os.path.join(results_dir, f'{fn}_{value}_by_{by}.csv'))
        plt.savefig(os.path.join(results_dir, f'{fn}_{value}.png'))


    def multi_plot(name, value, data=data, ax=None):
        unstacked = data.groupby(['SurveyName', value])['Weight'].sum().unstack(value)
        stacked = 100 * unstacked \
            .divide(unstacked.sum(axis=1), axis=0) \
            .stack()

        stacked.name = name

        stacked =  pd.merge(stacked.reset_index(), year_to_date, on='SurveyName')
        if ax == None:
            fig, ax = plt.subplots(figsize=fs)
        sns.lineplot(data = stacked, x='Date', y=name, hue=value, ax=ax)
        ax.set_title(f'{name.replace(" ","_")} ({value})')

        fn = name.replace(" ","_")
        stacked.set_index('SurveyName').to_csv(os.path.join(results_dir, f'{fn}_{value}.csv'))
        plt.savefig(os.path.join(results_dir, f'{fn}_{value}.png'))


    def age_pyramid_plot(name, data):
        age_pyramid = data.groupby(['SurveyName', 'AgeBin'])['Weight'].sum()
        year_sum = data.groupby(['SurveyName'])['Weight'].sum()
        age_pyramid = age_pyramid.divide(year_sum)
        age_pyramid.name = 'Percent'
        g = sns.catplot(x='Percent', y='AgeBin', hue='SurveyName', data=age_pyramid.reset_index(), kind='point')
        for a in g.axes.flat:
            a.invert_yaxis()
        g.fig.set_size_inches(fs[0], fs[1], forward=True)
        g.fig.suptitle(f'{name}')

        fn = name.replace(" ","_")
        age_pyramid.to_csv(os.path.join(results_dir, f'{fn}.csv'))
        plt.savefig(os.path.join(results_dir, f'{fn}.png'))

    def skyscraper(data, name, savefig=True, savedata=True):
        age_parity = data.groupby(['SurveyName', 'AgeBin', 'ParityBin'])['Weight'].sum()
        total = data.groupby(['SurveyName'])['Weight'].sum()
        age_parity = 100 * age_parity.divide(total).fillna(0)
        age_parity.name = 'Percent'
        fig, ax_vec = plt.subplots(1, 2, figsize=fs)

        for i, (year, d) in enumerate(age_parity.groupby('SurveyName')):
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
        N = data['SurveyName'].nunique()
        #for i, (sy, dat) in enumerate(tmp.groupby('SurveyName')):
        for i, sy in enumerate(yeardict.keys()): # Keep in order
            dat = data.loc[data['SurveyName']==sy]
            ans = dat.groupby('Method')['Weight'].sum()
            ax[i].pie(ans.values, labels=ans.index.tolist())
            ax[i].set_title(f'{title}: {sy}')
        plt.tight_layout()
    '''


    g = sns.FacetGrid(data=u.data, col='SurveyName', height=5)
    g.map_dataframe(plot_pie, by='MethodType').add_legend()#.set_xlabels('Year').set_ylabels('Percent')
    g.savefig(os.path.join(u.results_dir, 'MethodTypePies_by_Year.png'))

    g = sns.FacetGrid(data=u.data[u.data['MethodType']=='Modern'], col='SurveyName', height=5)
    g.map_dataframe(plot_pie, by='Method').add_legend()#.set_xlabels('Year').set_ylabels('Percent')
    g.savefig(os.path.join(u.results_dir, 'ModernMethodPies_by_Year.png'))


    # Age pyramid
    g = sns.FacetGrid(data=u.data, hue='SurveyName', height=5)
    g.map_dataframe(plot_pop_pyramid).add_legend().set_xlabels('Percent').set_ylabels('Age Bin')
    g.savefig(os.path.join(u.results_dir, 'PopulationPyramid.png'))


    # Skyscraper images
    g = sns.FacetGrid(data=u.data.loc[u.data['SurveyName']!='Midline'], col='SurveyName', height=5)
    g.map_dataframe(plot_skyscraper, age='AgeBin', parity='ParityBin', vmax=20).set_xlabels('Parity').set_ylabels('Age')
    g.savefig(os.path.join(u.results_dir, 'Skyscraper.png'))

    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', default=False, action='store_true')
    args = parser.parse_args()

    main(force_read = args.force)


#print(pd.crosstab(data['SurveyName'], data['v102'], data['v213']*data['Weight']/1e6, aggfunc=sum))
#with pd.option_context('display.precision', 1, 'display.max_rows', 1000): # 'display.precision',2, 
