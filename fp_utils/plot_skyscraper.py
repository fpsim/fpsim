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
        .unstack('Parity')

    # WIP - want to merge with skyscraper plotting from data_analysis
    import seaborn as sns
    sns.heatmap(age_parity_data)
    pl.show()

    '''
    axkwargs = dict(elev=37, azim=-31, nrows=nrows, ncols=ncols, index=idx)
    ax = sc.bar3d(fig=fig, data=age_parity_data, cmap='jet', axkwargs=axkwargs)
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
    '''
