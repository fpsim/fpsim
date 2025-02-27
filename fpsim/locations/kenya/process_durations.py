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
        try:
            dist = thisdf.functionform.iloc[0]
        except:
            print(f'No functionform for {method.label}')

        age_ind = sc.findfirst(thisdf.coef.values, 'age_grp_fact(0,18]')
        method.age_bin_vals = thisdf.estimate.values[age_ind:]
        if len(method.age_bin_vals) != 5:
            errormsg = f'Error: {method.label} has {len(method.age_bin_vals)} age bins, expected 5'
            raise ValueError(errormsg)
        method.age_bin_edges = [18, 20, 25, 35, 50]

        if dist in ['lognormal', 'lnorm']:
            method.dur_use['dist'] = 'lognormal'
            method.dur_use['par1'] = thisdf.estimate[thisdf.coef == 'meanlog'].values[0]
            method.dur_use['par2'] = thisdf.estimate[thisdf.coef == 'sdlog'].values[0]
        elif dist in ['gamma']:
            method.dur_use['dist'] = dist
            method.dur_use['par1'] = thisdf.estimate[thisdf.coef == 'shape'].values[0]
            method.dur_use['par2'] = thisdf.estimate[thisdf.coef == 'rate'].values[0]
        elif dist == 'llogis':
            method.dur_use['dist'] = dist
            method.dur_use['par1'] = thisdf.estimate[thisdf.coef == 'shape'].values[0]
            method.dur_use['par2'] = thisdf.estimate[thisdf.coef == 'scale'].values[0]
        elif dist == 'weibull':
            method.dur_use['dist'] = dist
            method.dur_use['par1'] = thisdf.estimate[thisdf.coef == 'shape'].values[0]
            method.dur_use['par2'] = thisdf.estimate[thisdf.coef == 'scale'].values[0]
        elif dist == 'exponential':
            method.dur_use['dist'] = dist
            method.dur_use['par1'] = thisdf.estimate[thisdf.coef == 'rate'].values[0]
            method.dur_use['par2'] = None

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
            # par1 = np.exp(method.dur_use['par1'] + method.age_bin_vals[ai])
            # par2 = np.exp(method.dur_use['par2'])

            if method.dur_use['dist'] == 'lognormal':
                par1 = np.exp(method.dur_use['par1'])
                par2 = np.exp(method.dur_use['par2'] + method.age_bin_vals[ai])
                rv = sps.lognorm(s=par2, loc=0, scale=par1)
            elif method.dur_use['dist'] == 'gamma':
                par1 = np.exp(method.dur_use['par1'])
                par2 = np.exp(method.dur_use['par2'] + method.age_bin_vals[ai])
                rv = sps.gamma(par1, scale=1/par2)
            elif method.dur_use['dist'] == 'llogis':
                par1 = np.exp(method.dur_use['par1'])
                par2 = np.exp(method.dur_use['par2'] + method.age_bin_vals[ai])
                rv = sps.fisk(c=par1, scale=par2)
            elif method.dur_use['dist'] == 'weibull':
                par1 = method.dur_use['par1']
                par2 = method.dur_use['par2'] + method.age_bin_vals[ai]
                rv = sps.weibull_min(c=par1, scale=par2)
            elif method.dur_use['dist'] == 'exponential':
                par1 = np.exp(method.dur_use['par1'] + method.age_bin_vals[ai])
                par2 = None
                rv = sps.expon(scale=1/par1)
            else:
                raise NotImplementedError(f'Distribution {method.dur_use["dist"]} not implemented')

            if par2 is not None:
                print(f'{method.label} - {age_bin_labels[ai]}: {par1:.2f}, {par2:.2f}: {rv.cdf(12*700)}')
            else:
                print(f'{method.label} - {age_bin_labels[ai]}: {par1:.2f}: {rv.cdf(12*700)}')
            ax.plot(x/12, rv.pdf(x), color=colors[ai], lw=2, label=age_bin_labels[ai])

        if pn == 6: ax.legend(loc='best', frameon=False)

        ax.set_xlabel('Duration of use')
        ax.set_ylabel('Density')
        ax.set_title(method.label+' - ' + method.dur_use['dist'])

    sc.figlayout()
    sc.savefig('duration_dists.png')


