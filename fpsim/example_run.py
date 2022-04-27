'''
Run example simulation with model output and compare to sample calibration from Senegal
'''

import pylab as pl
import sciris as sc
import fpsim as fp
from fpsim import defaults as fpd
from fp_analyses import senegal_parameters as sp
import pandas as pd

# Housekeeping
sc.tic()
pl.rcParams['font.size'] = 10

# import data files
popsize_file = 'data/senegal-popsize.csv'
skyscrapers_file = 'data/Skyscrapers-All-DHS.csv'

do_run = 1
do_plot_popsize = 1
do_plot_asfr = 1
do_plot_age_parity_heatmap = 1
do_save = 0

if do_run:
    pars = sp.make_pars()
    sim = fp.Sim(pars=pars)

    sim.run()
    sim.plot()

    res = sim.results

if do_plot_popsize:

    # Load data
    popsize = pd.read_csv(popsize_file, header=None)

    # Handle population size and mcpr from data
    pop_years_data = popsize.iloc[0, :].to_numpy()
    popsize_data = popsize.iloc[1, :].to_numpy() / (popsize.iloc[
                                                            1, 0] / sim.pars['n'])

    # Handle population size and mcpr from model
    pop_years_model = res['tfr_years']
    popsize_model = res['pop_size']

    fig = pl.figure(figsize=(16, 16))

    # Population size plot
    pl.plot(pop_years_model, popsize_model, color='cornflowerblue', label='FPsim')
    pl.scatter(pop_years_data, popsize_data, color='black', label='Data', zorder=1000)
    pl.title('Population growth')
    pl.xlabel('Years')
    pl.ylabel('Population')
    pl.legend()
    pl.show()

    if do_save:
        pl.savefig('senegal_popsize.png')

if do_plot_asfr:  # Plots ASFR for the last year of the sim in comparison to Senegal 2019 ASFR

    x = pl.arange(1,9)
    asfr_data = [1, 71, 185, 228, 195, 171, 74, 21] # From DHS Stat Compiler Senegal 2019 ASFR
    x_labels = []
    asfr_model = []

    for key in fpd.age_bin_mapping.keys():
        x_labels.append(key)
        asfr_model.append(res['asfr'][key][-1])

    fig, ax = pl.subplots()

    kw = dict(lw=3, alpha=0.7, markersize=10)
    ax.plot(x, asfr_data, marker='^', color='black', label="DHS data", **kw)
    ax.plot(x, asfr_model, marker='*', color='cornflowerblue', label="FPsim", **kw)
    pl.xticks(x, x_labels)
    pl.ylim(bottom=-10)
    ax.set_title('Age specific fertility rate per 1000 woman years')
    ax.set_xlabel('Age')
    ax.set_ylabel('ASFR in 2019')
    ax.legend(frameon=False)
    sc.boxoff()
    pl.show()

    if do_save:
        pl.savefig('ASFR_last_year.png')

if do_plot_age_parity_heatmap:

    # Set up
    min_age = 15
    max_age = 50
    bin_size = 5
    year_str = '2017'
    age_bins = pl.arange(min_age, max_age, bin_size)
    parity_bins = pl.arange(0, 8)
    n_age = len(age_bins)
    n_parity = len(parity_bins)
    x_age = pl.arange(n_age)
    x_parity = pl.arange(n_parity)  # Should be the same

    # Load data
    data_parity_bins = pl.arange(0, 18)
    sky_raw_data = pd.read_csv(skyscrapers_file, header=None)
    sky_raw_data = sky_raw_data[sky_raw_data[0] == year_str]
    sky_parity = sky_raw_data[2].to_numpy()
    sky_props = sky_raw_data[3].to_numpy()
    sky_arr = sc.odict()

    sky_arr['Data'] = pl.zeros((len(age_bins), len(parity_bins)))
    count = -1
    for age_bin in x_age:
        for dpb in data_parity_bins:
            count += 1
            parity_bin = min(n_parity - 1, dpb)
            sky_arr['Data'][age_bin, parity_bin] += sky_props[count]
    assert count == len(sky_props) - 1  # Ensure they're the right length

    # Extract from model
    sky_arr['Model'] = pl.zeros((len(age_bins), len(parity_bins)))
    ppl = sim.people
    for i in range(len(ppl)):
        if ppl.alive[i] and not ppl.sex[i] and ppl.age[i] >= min_age and ppl.age[i] < max_age:
            age_bin = sc.findinds(age_bins <= ppl.age[i])[-1]
            parity_bin = sc.findinds(parity_bins <= ppl.parity[i])[-1]
            sky_arr['Model'][age_bin, parity_bin] += 1

    # Normalize
    for key in ['Data', 'Model']:
        sky_arr[key] /= sky_arr[key].sum() / 100

    # Plot heatmaps
    ages = pl.arange(len(age_bins))
    parities = pl.arange(len(parity_bins))

    age_labels = [f'{5*x+10}-{5*(x+1)+10-1}' for x in ages]
    parity_labels = [f'{x}' for x in parities]

    sc.options(dpi=150)
    fig, axs = pl.subplots(figsize=(7,8), nrows=2)
    axdhs, axfps = axs
    cax = pl.axes([0.87, 0.07, 0.03, 0.88])

    cmap = 'viridis'
    vmax = max(sky_arr['Data'].max(), sky_arr['Model'].max())
    kw = dict(vmin=0, vmax=vmax, cmap=cmap, origin='lower', aspect='auto')

    pc1 = axdhs.imshow(sky_arr['Data'], **kw)
    pc2 = axfps.imshow(sky_arr['Model'], **kw)
    cb = fig.colorbar(pc1, cax=cax)
    cb.set_label('% of women', rotation=270, labelpad=15)

    # Configure labels
    kw = dict(fontweight='bold')
    axdhs.set_title('DHS data', **kw)
    axfps.set_title('FPsim', **kw)
    axdhs.set_xticklabels([])
    axfps.set_xlabel('Age')
    axfps.set_xticklabels(age_labels)
    for ax in axs:
        ax.set_xticks(ages)
        ax.set_yticks(parities)
        ax.set_ylabel('Parity')
        ax.set_yticklabels(parity_labels)

    # Adjust layout and show
    sc.figlayout(right=0.8)
    pl.show()
    if do_save:
        pl.savefig('senegal_heatmap.png')
