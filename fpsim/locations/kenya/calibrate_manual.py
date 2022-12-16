'''
The very start of a script for running plotting to compare the model to data
'''


import sciris as sc
import fpsim as fp
import pylab as pl

# Set options
do_plot = True
pars = fp.pars(location='kenya')
pars['n_agents'] = 50000 # Small population size
pars['end_year'] = 2020 # 1961 - 2020 is the normal date range
pars['exposure_correction'] = 1.0 # Overall scale factor on probability of becoming pregnant

sc.tic()
sim = fp.Sim(pars=pars)
sim.run()

if do_plot:
    sim.plot()

age_bin_map = {
        '10-14': [10, 15],
        '15-19': [15, 20],
        '20-24': [20, 25],
        '25-29': [25, 30],
        '30-34': [30, 35],
        '35-39': [35, 40],
        '40-44': [40, 45],
        '45-49': [45, 50]
}

res = sim.results

for key in age_bin_map.keys():
            print(f'ASFR (annual) for age bin {key} in the last year of the sim: {res["asfr"][key][-1]}')

x = [1, 2, 3, 4, 5, 6, 7, 8]
asfr_data = [3.03, 64.94, 172.12, 174.12, 136.10, 80.51, 34.88, 13.12]  # From UN Data Kenya 2020
x_labels = []
asfr_model = []

for key in age_bin_map.keys():
        x_labels.append(key)
        asfr_model.append(res['asfr'][key][-1])

fig, ax = pl.subplots()
kw = dict(lw=3, alpha=0.7, markersize=10)
ax.plot(x, asfr_data, marker='^', color='black', label="UN data", **kw)
ax.plot(x, asfr_model, marker='*', color='cornflowerblue', label="FPsim", **kw)
pl.xticks(x, x_labels)
pl.ylim(bottom=-10)
ax.set_title('Age specific fertility rate per 1000 woman years')
ax.set_xlabel('Age')
ax.set_ylabel('ASFR in 2019')
ax.legend(frameon=False)
sc.boxoff()
pl.show()

sc.toc()
print('Done.')
