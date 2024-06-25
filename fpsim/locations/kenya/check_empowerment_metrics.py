'''
Short simulation to compare input empowerment data to what's propagated inside
the simulation.
'''

import numpy as np
import matplotlib.pyplot as plt

import sciris as sc
import fpsim as fp

# Set options
do_plot = True
empwr = fp.Empowerment()
pars = fp.pars(location='kenya')
pars['n_agents'] = 10_000  # Smallish population size
pars['end_year'] = 1980  # Very short simulation


sc.tic()
#age_bin_edges = np.concatenate((np.array([0]), np.linspace(15, 50, 36, endpoint=True), np.array([100])))
age_bin_edges = np.arange(15, 51)  # for simplicity track data in 1yr bins, in the relevant range

sim = fp.Sim(pars=pars, empowerment_module=empwr, analyzers=[fp.empowerment_recorder(bins=age_bin_edges)])
sim.run()

sc.toc()
print('Done.')

empwr_analyzer = sim.get_analyzers()[0]
if do_plot:
    empwr_analyzer.plot_snapshot(ti=0)
    plt.show()

def plot_sim_vs_data(emp_data, sim_data, bins, nbins, ti=0,
                     to_plot=None, fig_args=None, pl_args=None):

    fig_args = sc.mergedicts(fig_args)
    pl_args = sc.mergedicts(pl_args)
    fig = plt.figure(**fig_args)

    if to_plot is None:
        to_plot = emp_data["avail_metrics"]

    nkeys = len(to_plot)
    rows, cols = sc.get_rows_cols(nkeys)

    axs = []
    for k, key in enumerate(to_plot):
        axs.append(fig.add_subplot(rows, cols, k+1))

        simdata = np.array(sim_data[key], dtype=float)[:, ti]  # (nbinx x ntpts)
        simage = [bins[i] for i in range(nbins)]
        empdata = emp_data[key]
        empage = emp_data["avail_ages"]
        ttl = f"metric: {key} at timestep {ti}"

        ylabel = f"proportion of women who\n{key}"
        empcolor = "powderblue"
        simcolor = "gray"
        ymin, ymax = 0, 1.05

        axs[k].bar(empage, empdata, label="empirical", color=empcolor, ec=empcolor, **pl_args)
        axs[k].bar(simage, simdata, label="fpsim", color='none', ec=simcolor, hatch="/", **pl_args)

        # Label plots
        axs[k].set_title(ttl)
        axs[k].set_xlabel('Age group (years)')
        axs[k].set_ylabel(ylabel)
        axs[k].set_ylim([ymin, ymax])
        axs[k].legend()

    return fig


# Plot data from analyzer and  in empowerment pars
empwr_data = sim.empowerment_module.empowerment_pars
anlzr_data = empwr_analyzer.data
bins = empwr_analyzer.bins
nbins = empwr_analyzer.nbins

plot_sim_vs_data(empwr_data, anlzr_data, bins, nbins, ti=240)
plt.show()

