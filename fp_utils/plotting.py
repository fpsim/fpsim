import os
import pandas as pd
import numpy as np
import argparse
import sciris as sc
import pylab as pl

def load(exp_id):
    data = pd.read_csv(os.path.join(exp_id, 'CSVAnalyzer.csv'), skipinitialspace=True) \
        .set_index('SimId')
    tags = pd.read_csv(os.path.join(exp_id, 'tags.csv'), skipinitialspace=True) \
        .set_index('SimId')

    data = tags.join(data)

    return data


###############################################################################
# PLOT: Skyscraper ############################################################
###############################################################################
def skyscraper(data, label=None, fig=None, nrows=None, ncols=None, idx=None, figkwargs=None, axkwargs=None):
    age_parity_data = data \
        .groupby(['Age', 'Parity'])['Population'] \
        .sum() \
        .unstack('Parity') \
        .drop(0, axis=0)

    # WIP - want to merge with skyscraper plotting from data_analysis
    #import seaborn as sns
    #sns.heatmap(age_parity_data)

    parity_bins = age_parity_data.columns.values

    fig = pl.figure()

    nrows = ncols = idx = 1

    axkwargs = dict(elev=37, azim=-31, nrows=nrows, ncols=ncols, index=idx)
    ax = sc.bar3d(fig=fig, data=age_parity_data.values, cmap='jet', axkwargs=axkwargs)
    age_bin_labels = age_parity_data.index.values
    ax.set_xlabel('Age')
    ax.set_xticks(np.arange(len(age_bin_labels))+0.5) # To center the tick marks
    ax.set_xticklabels(age_bin_labels)

    parity_bin_labels = age_parity_data.columns
    ax.set_ylabel('Parity')
    ax.set_yticks(np.arange(len(parity_bin_labels))+0.5) # To center the tick marks
    ax.set_yticklabels(parity_bin_labels)


###############################################################################
# PLOT: Stacked bar ###########################################################
###############################################################################
def bars(data):
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
