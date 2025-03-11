"""
Investigating durations
"""

import numpy as np
import sciris as sc
import fpsim as fp
import fpsim.locations as fplocs
import pylab as pl
import pandas as pd
import scipy.stats as sps


if __name__ == '__main__':

    location = 'ethiopia'
    cm = fp.SimpleChoice(location=location)
    methods = cm.methods

    del methods['btl']

    fig, axes = pl.subplots(3, 3, figsize=(12,10))
    axes = axes.ravel()
    x = np.arange(0,15*12+0.5) # x axis
    age_bins = [18, 20, 25, 35, 50]
    age_bin_labels = ['<18', '18-20', '20-25', '25-35', '35-50']
    colors = sc.vectocolor(age_bins)

    for pn, method in enumerate(methods.values()):
        ax = axes[pn]

        dist_pars_fun, make_dist_dict = cm._get_dist_funs(methods[pn].dur_use['dist'])
        par1, par2 = dist_pars_fun(methods[pn].dur_use, np.arange(5))

        for ai, ab in enumerate(age_bins):
            if method.dur_use['dist'] == 'lognormal_sps':
                rv = sps.lognorm(s=par2[ai], scale=par1, loc=0)  # NOTE ORDERING
            elif method.dur_use['dist'] == 'gamma':
                rv = sps.gamma(a=par1, scale=par2[ai])
            elif method.dur_use['dist'] == 'llogis':
                rv = sps.fisk(c=par1, scale=par2[ai])
            elif method.dur_use['dist'] == 'weibull':
                rv = sps.weibull_min(c=par1, scale=par2[ai])
            elif method.dur_use['dist'] == 'exponential':
                rv = sps.expon(scale=par1[ai])
            else:
                raise NotImplementedError(f'Distribution {method.dur_use["dist"]} not implemented')

            if par2 is not None:
                print(f'{method.label} - {age_bin_labels[ai]}: {par1:.2f}, {par2[ai]:.2f}: {rv.cdf(12*700)}')
            else:
                print(f'{method.label} - {age_bin_labels[ai]}: {par1[ai]:.2f}: {rv.cdf(12*700)}')

            ax.plot(x, rv.pdf(x), color=colors[ai], lw=2, label=age_bin_labels[ai])

        if pn == 6: ax.legend(loc='best', frameon=False)

        ax.set_xlabel('Duration of use')
        ax.set_ylabel('Density')
        ax.set_title(method.label+' - ' + method.dur_use['dist'])

    sc.figlayout()
    sc.savefig(f'duration_dists_{location}.png')


