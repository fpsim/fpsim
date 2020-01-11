from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def wmean(data, value, weight):
    if data[value].dtype.name == 'category':
        return  np.sum(data[weight] * data[value].cat.codes) / np.sum(data[weight])
    return np.sum(data[weight] * data[value]) / np.sum(data[weight])


def weighted_lineplot(*args, **kwargs):
    data = kwargs.pop('data')
    gb = data.groupby(['SurveyName', 'Date', 'Survey'])
    weighted = 100 * gb.apply( partial(wmean, value=kwargs['value'], weight='Weight') )
    weighted.name = 'Weighted'
    weighted = weighted.reset_index()

    weighted.sort_values(by=['Date', 'Survey'], inplace=True)

    #weighted = pd.merge(weighted.reset_index(), year_to_date, on='SurveyName')
    #plt.plot(weighted['Date'], weighted['Weighted'], color=kwargs['color'], label=kwargs['label'])
    sns.lineplot(data=weighted, x='Date', y='Weighted', style='Survey', color=kwargs['color'], label=kwargs['label'])


def boolean_plot(name, value, data=None, ax=None):
    gb = data.groupby(['SurveyName'])
    weighted = 100 * gb.apply( partial(wmean, data=data, value=value, weight='Weight') )
    weighted.name = name
    #weighted =  pd.merge(weighted.reset_index(), year_to_date, on='SurveyName')
    if ax == None:
        fig, ax = plt.subplots(figsize=fs)
    sns.lineplot(data = weighted, x='Date', y=name, ax=ax)
    ax.set_title(f'{name} ({value})')

    fn = name.replace(" ","_")
    weighted.set_index('SurveyName').to_csv(os.path.join(results_dir, f'{fn}_{value}.csv'))
    plt.savefig(os.path.join(results_dir, f'{fn}_{value}.png'))


def boolean_plot_by(name, value, by, data=None, ax=None):
    gb = data.groupby(['SurveyName', by])
    weighted = 100 * gb.apply( partial(wmean, data=data, value=value, weight='Weight') )
    weighted.name = name
    #weighted =  pd.merge(weighted.reset_index(), year_to_date, on='SurveyName')
    if ax == None:
        fig, ax = plt.subplots(figsize=fs)
    sns.lineplot(data = weighted, x='Date', y=name, hue=by, ax=ax)
    ax.set_title(f'{name} ({value} by {by})')

    fn = name.replace(" ","_")
    weighted.set_index(['SurveyName', by]).to_csv(os.path.join(results_dir, f'{fn}_{value}_by_{by}.csv'))
    plt.savefig(os.path.join(results_dir, f'{fn}_{value}.png'))


def multi_plot(name, value, data=None, ax=None):
    unstacked = data.groupby(['SurveyName', value])['Weight'].sum().unstack(value)
    stacked = 100 * unstacked \
        .divide(unstacked.sum(axis=1), axis=0) \
        .stack()

    stacked.name = name

    #stacked =  pd.merge(stacked.reset_index(), year_to_date, on='SurveyName')
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

def plot_pie(title, data, yeardict):
    fig, ax = plt.subplots(1,len(yeardict),figsize=(16,6))
    N = data['SurveyName'].nunique()
    #for i, (sy, dat) in enumerate(tmp.groupby('SurveyName')):
    for i, sy in enumerate(yeardict.keys()): # Keep in order
        dat = data.loc[data['SurveyName']==sy]
        ans = dat.groupby('Method')['Weight'].sum()
        ax[i].pie(ans.values, labels=ans.index.tolist())
        ax[i].set_title(f'{title}: {sy}')
    plt.tight_layout()
