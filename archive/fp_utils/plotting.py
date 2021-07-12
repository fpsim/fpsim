import os
import seaborn as sns
import pandas as pd
import numpy as np
import argparse
import sciris as sc
import pylab as pl

fs=(12,8)

def plot_line_percent(*args, **kwargs):
    data = kwargs.pop('data')
    by = kwargs.pop('by')

    tmp = data.groupby(['Survey', 'SurveyName', 'Date', by])['Weight'].sum().reset_index(by)
    weight_sum = data.groupby(['Survey', 'SurveyName', 'Date'])['Weight'].sum()
    tmp['Percent'] = 100*tmp['Weight'].divide(weight_sum)

    # Ugh, have to make all levels have a value to avoid color or linetype errors
    tmp = tmp.set_index(by, append=True).unstack(fill_value=0).stack().reset_index()#.sort_values(['Survey', by, 'Date']) # 0 or -1???
    tmp.loc[tmp['Percent']<0,'Percent'] = np.NaN

    if 'values' in kwargs:
        values = kwargs.pop('values')
        tmp = tmp.loc[tmp[by].isin(values)]

        if len(values) > 1:
            kwargs['hue'] = by
    else:
        kwargs['hue'] = by

    if 'Survey' in tmp and tmp['Survey'].nunique() > 1:
        kwargs['style'] = 'Survey'

    sns.lineplot(data=tmp, x='Date', y='Percent', **kwargs) # hue=by, style='Survey',


def plot_stack(*args, **kwargs):
    data = kwargs.pop('data')
    by = kwargs.pop('by')

    tmp = data.groupby(['Survey', 'SurveyName', 'Date', by])['Weight'].sum().reset_index(by)
    weight_sum = data.groupby(['Survey', 'SurveyName', 'Date'])['Weight'].sum()
    tmp['Percent'] = 100*tmp['Weight'].divide(weight_sum)

    # Ugh, have to make all levels have a value to avoid color or linetype errors
    tmp = tmp.set_index(by, append=True).unstack(fill_value=0).stack().reset_index()#.sort_values(['Survey', by, 'Date']) # 0 or -1???
    tmp.loc[tmp['Percent']<0,'Percent'] = np.NaN

    if 'order' in kwargs:
        order = kwargs.pop('order')
    else:
        order = tmp[by].unique()

    tmp.set_index(['Date', by])['Percent'].unstack(by)[order].plot.area(ax=pl.gca(), linewidth=0) #x='Date', y='Percent'


def plot_pie(*args, **kwargs):
    data = kwargs.pop('data')
    by = kwargs.pop('by')
    ans = data.groupby(by)['Weight'].sum()
    print(ans)
    kwargs.pop('color')
    pl.pie(ans.values, labels=ans.index.tolist(), labeldistance=1.1, explode=0.1*np.ones_like(ans.values), autopct='%1.0f%%', **kwargs)


def plot_pop_pyramid(*args, **kwargs):
    data = kwargs.pop('data')
    ap = data.groupby(['SurveyName', 'AgeBin'])['Weight'].sum()
    year_sum = data.groupby(['SurveyName'])['Weight'].sum()
    ap = ap.divide(year_sum)
    ap.name = 'Percent'

    pl.plot(ap.values, ap.index.get_level_values('AgeBin'), **kwargs)


###############################################################################
# PLOT: Skyscraper ############################################################
###############################################################################
def plot_skyscraper(*args, **kwargs):
    data = kwargs.pop('data')

    age = 'Age'
    if 'age' in kwargs:
        age = kwargs.pop('age')

    parity = 'Parity'
    if 'parity' in kwargs:
        parity = kwargs.pop('parity')

    vmax = None
    if 'vmax' in kwargs:
        vmax = kwargs.pop('vmax')

    age_parity_data = data \
        .groupby([age, parity])['Weight'] \
        .sum() \
        .unstack(parity) \
        .fillna(0)
        #.drop(0, axis=0)

    age_parity_data = 100 * age_parity_data / age_parity_data.sum().sum() # Normalize

    g = sns.heatmap(age_parity_data, vmax=vmax, cmap='jet')
    pl.gca().invert_yaxis()

    #pl.gca().imshow(age_parity_data, vmax=vmax, aspect='auto', cmap='jet', interpolation='none', origin='lower')

    # 3D axes not compatible with FacetGrid?
    '''
    nrows = ncols = idx = 1

    axkwargs = dict(elev=37, azim=-31, nrows=nrows, ncols=ncols, index=idx)
    ax = sc.bar3d(fig=pl.gcf(), data=age_parity_data.values, cmap='jet', axkwargs=axkwargs)
    age_bin_labels = age_parity_data.index.values
    ax.set_xlabel('Age')
    ax.set_xticks(np.arange(len(age_bin_labels))+0.5) # To center the tick marks
    ax.set_xticklabels(age_bin_labels)

    parity_bin_labels = age_parity_data.columns
    ax.set_ylabel('Parity')
    ax.set_yticks(np.arange(len(parity_bin_labels))+0.5) # To center the tick marks
    ax.set_yticklabels(parity_bin_labels)
    '''


###############################################################################
# PLOT: Stacked bar ###########################################################
###############################################################################
def bars(data): # TODO - make compatible with FacetGrid like others above
    sc.heading('Plotting stacked bars')
    pivot_by_age = data \
        .groupby(['Age', 'IP_Key:CurrentStatus'])['Population'].sum() \
        .reset_index() \
        .pivot(index='Age', columns='IP_Key:CurrentStatus', values='Population')
    pivot_by_age.plot.bar(stacked=True, figsize=(10,10))

    pivot_by_parity = data \
        .groupby(['Parity', 'IP_Key:CurrentStatus'])['Population'].sum() \
        .reset_index() \
        .pivot(index='Parity', columns='IP_Key:CurrentStatus', values='Population')
    pivot_by_parity.plot.bar(stacked=True, figsize=(10,10))
