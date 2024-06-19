"""
Investigating durations
"""

import numpy as np
import sciris as sc
import fpsim as fp
import pandas as pd
import pylab as pl
import scipy.stats as sps


def lognorm_params(par1, par2):
    """
    Given the mean and std. dev. of the log-normal distribution, this function
    returns the shape and scale parameters for scipy's parameterization of the
    distribution.
    """
    # mean = np.log(par1 ** 2 / np.sqrt(par2 ** 2 + par1 ** 2))  # Computes the mean of the underlying normal distribution
    # sigma = np.sqrt(np.log(par2 ** 2 / par1 ** 2 + 1))  # Computes sigma for the underlying normal distribution
    mean = par1
    sigma = par2
    scale = np.exp(mean)
    shape = sigma
    return shape, scale


def process_data():

    methods = sc.dcp(fp.Methods)
    del methods['btl']

    # Read in duration estimates
    dur_raw = pd.read_csv('data/method_time_coefficients.csv', keep_default_na=False, na_values=['NaN'])

    for method in methods.values():
        mlabel = method.csv_name

        thisdf = dur_raw.loc[(dur_raw.method == mlabel)]
        dist = thisdf.functionform.iloc[0]

        method.age_bin_vals = thisdf.coef.values[2:]
        method.age_bin_edges = [18, 20, 25, 35, 50]

        if dist == 'lognormal':
            method.dur_use['dist'] = dist
            method.dur_use['par1'] = thisdf.coef[thisdf.estimate=='meanlog'].values[0]
            method.dur_use['par2'] = thisdf.coef[thisdf.estimate=='sdlog'].values[0]
        elif dist in ['gamma', 'gompertz']:
            method.dur_use['dist'] = dist
            method.dur_use['par1'] = thisdf.coef[thisdf.estimate=='shape'].values[0]
            method.dur_use['par2'] = thisdf.coef[thisdf.estimate=='rate'].values[0]
        elif dist == 'llogis':
            method.dur_use['dist'] = dist
            method.dur_use['par1'] = thisdf.coef[thisdf.estimate=='shape'].values[0]
            method.dur_use['par2'] = thisdf.coef[thisdf.estimate=='scale'].values[0]

    return methods


if __name__ == '__main__':

    methods = process_data()

    fig, axes = pl.subplots(3, 3, figsize=(12,10))
    axes = axes.ravel()
    x = np.arange(0,15*12+0.5,0.5) # x axis
    age_bins = [18, 20, 25, 35, 50]
    age_bin_labels = ['<18', '18-20', '20-25', '25-35', '35-50']
    colors = sc.vectocolor(age_bins)

    for pn, method in enumerate(methods.values()):
        ax = axes[pn]
        for ai, ab in enumerate(method.age_bin_edges):
            par1 = np.exp(method.dur_use['par1'] + method.age_bin_vals[ai])
            par2 = np.exp(method.dur_use['par2'])

            if method.dur_use['dist'] == 'lognormal':
                par1 = method.dur_use['par1'] + method.age_bin_vals[ai]
                par2 = method.dur_use['par2']
                sigma, scale = lognorm_params(par1, par2)
                rv = sps.lognorm(sigma, 0, scale)
            if method.dur_use['dist'] == 'gamma':
                rv = sps.gamma(par1, scale=1/par2)
            # if method.dur_use['dist'] == 'gompertz':
            #     par1 = method.dur_use['par1'] + method.age_bin_vals[ai]
            #     rv = sps.gompertz(par1, scale=1/par2)
            if method.dur_use['dist'] == 'llogis':
                rv = sps.fisk(c=par1, scale=par2)

            ax.plot(x/12, rv.pdf(x), color=colors[ai], lw=2, label=age_bin_labels[ai])

        if pn==6: ax.legend(loc='best', frameon=False)

        ax.set_xlabel('Duration of use')
        ax.set_ylabel('Density')
        ax.set_title(method.label+' - ' + method.dur_use['dist'])

    sc.figlayout()
    sc.savefig('duration_dists.png')


