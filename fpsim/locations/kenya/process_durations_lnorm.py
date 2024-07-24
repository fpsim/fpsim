"""
Investigating durations
"""

import numpy as np
import sciris as sc
import fpsim as fp
import pandas as pd
import pylab as pl
import scipy.stats as sps


def process_data():

    methods = sc.dcp(fp.Methods)
    del methods['btl']

    # Read in duration estimates
    dur_raw = pd.read_csv('data/method_time_coefficients_all.csv', keep_default_na=False, na_values=['NaN'])

    for method in methods.values():
        mlabel = method.csv_name

        thisdf = dur_raw.loc[(dur_raw.method == mlabel) & (dur_raw.functionform == 'lnorm')]
        method.age_bin_vals = thisdf.coefficient.values[2:]
        method.age_bin_edges = [18, 20, 25, 35, 50]
        if mlabel != 'Injectable':
            method.dur_use['dist'] = 'lognormal'
            try:
                method.dur_use['par1'] = thisdf.coefficient[thisdf.parameter == 'meanlog'].values[0]
            except:
                print('hi')
            method.dur_use['par2'] = thisdf.coefficient[thisdf.parameter == 'sdlog'].values[0]
        else:
            thisdf = dur_raw.loc[(dur_raw.method == mlabel) & (dur_raw.functionform == 'gamma')]
            method.age_bin_vals = thisdf.coefficient.values[2:]
            method.dur_use['dist'] = 'gamma'
            method.dur_use['par1'] = thisdf.coefficient[thisdf.parameter == 'shape'].values[0]
            method.dur_use['par2'] = thisdf.coefficient[thisdf.parameter == 'rate'].values[0]

    return methods


if __name__ == '__main__':

    methods = process_data()

    fig, axes = pl.subplots(3, 3, figsize=(12, 10))
    axes = axes.ravel()
    x = np.arange(0, 15*12+0.5, 0.5) # x axis
    age_bins = [18, 20, 25, 35, 50]
    age_bin_labels = ['<18', '18-20', '20-25', '25-35', '35-50']
    colors = sc.vectocolor(age_bins)

    for pn, method in enumerate(methods.values()):
        ax = axes[pn]
        for ai, ab in enumerate(method.age_bin_edges):
            if method.dur_use['dist'] == 'lognormal':
                par1 = method.dur_use['par1'] + method.age_bin_vals[ai]
                par2 = np.exp(method.dur_use['par2'])
                rv = sps.lognorm(par2, 0, np.exp(par1))
            elif method.dur_use['dist'] == 'gamma':
                par1 = np.exp(method.dur_use['par1'])
                par2 = 1 / np.exp(method.dur_use['par2'] + method.age_bin_vals[ai])
                rv = sps.gamma(par1, scale=par2)

            ax.plot(x/12, rv.pdf(x), color=colors[ai], lw=2, label=age_bin_labels[ai])

        if pn == 6: ax.legend(loc='best', frameon=False)

        ax.set_xlabel('Duration of use')
        ax.set_ylabel('Density')
        ax.set_title(method.label+' - ' + method.dur_use['dist'])

    sc.figlayout()
    sc.savefig('durations_lnorm.png')


