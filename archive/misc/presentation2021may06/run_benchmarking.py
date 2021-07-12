'''
Create figure for benchmarking the model
'''

# Flag to set which version to benchmark -- need to run separately
use_orig = True

import numpy   as np
import pylab   as pl
import sciris  as sc
import matplotlib as mpl
if use_orig:
    import fpsim_orig as fp
    import fp_analyses_orig.senegal_parameters as sp
else:
    import fpsim as fp
    import fp_analyses.senegal_parameters as sp

pop_sizes_n = np.array([1e2, 2e2, 5e2, 1e3, 2e3, 5e3, 10e3, 20e3, 50e3, 100e3])
pop_sizes_o = np.array([1e2, 2e2, 5e2, 1e3, 2e3, 5e3, 10e3])

if use_orig:
    pop_sizes = pop_sizes_n
    n_sizes = len(pop_sizes_o)
else:
    pop_sizes = pop_sizes_o
    n_sizes = len(pop_sizes_n)

#%% Test speed

# Set options
default_pars = sp.make_pars()

def run(n):
    pars = sc.dcp(default_pars)
    pars['n'] = n # Small population size
    sim = fp.Sim(pars=pars)
    sim.run()
    return

rerun_speed = False

if rerun_speed:

    results = {k:[] for k in pop_sizes}
    T0 = sc.tic()
    repeats = 1
    for p in pop_sizes:
        for r in range(repeats):
            T = sc.tic()
            run(int(p))
            elapsed = sc.toc(T, output=True)
            results[p].append(elapsed)
            print(f'Ran {p}: {elapsed:0.2f} s ({sc.toc(T0, output=True):0.2f} s)')

    sc.toc(T0)

else:
    results_n = {
        100.0: [1.1809098720550537],
         200.0: [1.1524598598480225],
         500.0: [1.3407106399536133],
         1000.0: [1.3936264514923096],
         2000.0: [1.764389991760254],
         5000.0: [2.365990161895752],
         10000.0: [3.8409616947174072],
         20000.0: [5.7647130489349365],
         50000.0: [14.671123027801514],
         100000.0: [25.951420783996582],
         }

    results_o = {
        100.0: [2.306666135787964],
         200.0: [4.368374824523926],
         500.0: [8.88023066520691],
         1000.0: [18.26348328590393],
         2000.0: [33.023337602615356],
         5000.0: [90.1453378200531],
         10000.0: [176.40770077705383],
         }

# Analyze results

r = sc.objdict()

for k,pop_sizes,results in zip('no', [pop_sizes_n, pop_sizes_o], [results_n, results_o]):
    r[k] = np.zeros(len(pop_sizes))
    for i,p in enumerate(pop_sizes):
        r[k][i] = np.median(np.array(results[p]))



#%% Plot results

do_plot = True

if do_plot:

    fig = pl.figure(figsize=(22,10))
    axis_args = dict(left=0.10, bottom=0.10, right=0.95, top=0.97, wspace=0.25, hspace=0.25)
    pl.subplots_adjust(**axis_args)
    pl.rcParams['font.family'] = 'Proxima Nova'
    pl.rcParams['font.size'] = 18
    msize = 12

    ax = pl.subplot(1,1,1)
    f_n = 3500
    f_o = 50
    ny = 2020 - 1960
    pl.plot(pop_sizes_o, r.o, 'o', markersize=msize, label='Old (loop-based)', zorder=100)
    pl.plot(pop_sizes_n, r.n, 'o', markersize=msize, label='New (array-based)', zorder=100)
    pl.plot(pop_sizes_o, pop_sizes_o/f_o, c=[0.1,0.1,0.9], label=f'Old constant time ({f_o*ny/1e3:0.0f},000 person-years / second)')
    pl.plot(pop_sizes_n, pop_sizes_n/f_n, c=[0.6,0.6,0.1], label=f'New constant time ({f_n*ny/1e3:0.0f},000 person-years / second)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim([1, 200])
    sc.commaticks(axis='x')
    ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    pl.xlabel('Population size')
    pl.ylabel('CPU time (s)')
    pl.legend()

    pl.show()
    pl.savefig('fpsim-performance.png', dpi=150)

