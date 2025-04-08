"""
Test initation intervention with simple contraception module
- https://github.com/fpsim/fpsim/tree/rc2.0-methodtime-calibration
"""
import sys
import numpy as np

sys.path.append('..')

import matplotlib.pyplot as plt
import fpsim as fp
import sciris as sc


start_intv = 2040
end_intv = 2050

ax1_ylims = [0, 1800]
ax2_ylims = [0.015, 0.065]
ax3_ylims = [10_000, 15_000]
ax_mix_ylims = [0, 150]


def make_sims():
    location = 'kenya'
    pars = fp.pars(location=location)

    # Settings
    pars['n_agents'] = 20_000
    pars['start_year'] = 2000
    pars['end_year'] = 2060
    pars['timestep'] = 1

    # Free parameters for calibration
    pars['fecundity_var_low'] = 0.5
    pars['fecundity_var_high'] = 2
    pars['exposure_factor'] = 0.0575

    p1 = 0.03
    p2 = 0.03
    intv_1 = fp.change_initiation(years=[start_intv, end_intv], annual=True, perc=p1, force_theoretical=False)  #
    intv_2 = fp.change_initiation(years=[start_intv, end_intv], annual=True, perc=p2, force_theoretical=True)   # force_theoretical == True --> Force the theoretical exponential increase defined vy the initial value at the start of the intervention and the annual increase

    sim0 = fp.Sim(
            pars=pars,
            contraception_module=fp.RandomChoice(pars={"p_use": 0.05, "force_choose": False}),
            analyzers=[fp.state_tracker(state_name="on_contra"), fp.method_mix_over_time()],
            label="Baseline - No intervention"
        )

    sim1 = fp.Sim(
            pars=pars,
            contraception_module=fp.RandomChoice(pars={"p_use": 0.05, "force_choose": False}),
            analyzers=[fp.state_tracker(state_name="on_contra"), fp.method_mix_over_time()],
            interventions=intv_1,
            label=f"Intervention: {p1*100}%/year more women on contra. "
        )

    sim2 = fp.Sim(
            pars=pars,
            contraception_module=fp.RandomChoice(pars={"p_use": 0.05, "force_choose": False}),
            analyzers=[fp.state_tracker(state_name="on_contra"), fp.method_mix_over_time()],
            interventions=intv_2,
            label=f"Intervention: {p1*100}%/year more women on contra. Force theoretical value."
        )
    return sim0, sim1, sim2


# Workaround https://github.com/fpsim/fpsim/issues/449
def make_sim_list(sim, n=10):
    sim_list = sc.autolist()
    for seed in range(n):
        sim_list.append(sc.dcp(sim))
        sim_list[seed].pars["seed"] = seed
    return sim_list


def run_single_replicate_multisim():
    sim0, sim1, sim2 = make_sims()
    simlist = sc.autolist([sim0, sim1, sim2])
    msim = fp.MultiSim(sims=simlist)
    msim.run(serial=True, compute_stats=False)

    fig0 = msim.sims[0].get_analyzers()[0].plot()
    fig1 = msim.sims[1].get_analyzers()[0].plot()
    fig2 = msim.sims[2].get_analyzers()[0].plot()

    fig0a = msim.sims[0].get_analyzers()[1].plot()
    fig1a = msim.sims[1].get_analyzers()[1].plot()
    fig2a = msim.sims[2].get_analyzers()[1].plot()

    for fig, sim in zip([fig0, fig1, fig2], msim.sims):
        intv = sim.get_interventions()
        if len(intv):
            fig.axes[0].plot(sim.tvec[::12], intv[0].init_women_oncontra*np.ones(len(sim.tvec[::12])),
                             linestyle="none", marker="^", markersize=5, color="steelblue", zorder=0)
            fig.axes[0].plot(sim.tvec[::12], intv[0].expected_women_oncontra*np.ones(len(sim.tvec[::12])),
                             linestyle="none", marker="v", markersize=5, color="steelblue", zorder=0)

        fig.axes[2].vlines([start_intv], ax3_ylims[0], ax3_ylims[1],
                           linestyle="--", color="dimgray", zorder=0)
        fig.axes[2].vlines([end_intv], ax3_ylims[0], ax3_ylims[1],
                           linestyle="--", color="dimgray", zorder=0)
        fig.axes[0].set_ylim(ax1_ylims)
        fig.axes[1].set_ylim(ax2_ylims)
        fig.axes[2].set_ylim(ax3_ylims)
        fig.axes[0].set_xlim([sim.tvec[0], sim.tvec[-1]])
        fig.suptitle(f"Sim: {sim.label}")

        # Plot baseline simulation on the the figures with interventions
        baseline_num = msim.sims[0].get_analyzers()[0].data_num
        fig1.axes[0].plot(msim.sims[0].tvec, baseline_num, linestyle="-",
                          color="tomato", zorder=0)
        fig2.axes[0].plot(msim.sims[0].tvec, baseline_num, linestyle="-",
                          color="tomato", zorder=0)


        for fig, sim in zip([fig0a, fig1a, fig2a], msim.sims):
            fig.axes[0].vlines([start_intv], ax_mix_ylims[0], ax_mix_ylims[1],
                               linestyle="--", color="dimgray", zorder=0)
            fig.axes[0].vlines([end_intv], ax_mix_ylims[0], ax_mix_ylims[1],
                               linestyle="--", color="dimgray", zorder=0)
            fig.axes[0].set_ylim(ax_mix_ylims)
            fig.axes[0].set_xlim([sim.tvec[0], sim.tvec[-1]])
            fig.suptitle(f"Sim: {sim.label}")

    msim.plot()
    plt.show()
    return msim


def run_multi_replicate_multisim(n_runs=10):
    sim0, sim1, sim2 = make_sims()
    # Run multiple replicates of the same sim, for each of the different base simulations
    sims0 = make_sim_list(sim0, n=n_runs)
    sims1 = make_sim_list(sim1, n=n_runs)
    sims2 = make_sim_list(sim2, n=n_runs)

    msim0 = fp.MultiSim(sims=sims0)
    msim1 = fp.MultiSim(sims=sims1)
    msim2 = fp.MultiSim(sims=sims2)

    msim0.run()
    msim1.run()
    msim2.run()

    msim0.compute_stats()
    msim1.compute_stats()
    msim2.compute_stats()

    fig0 = plot_multi_replicates(msim0)
    fig1 = plot_multi_replicates(msim1)
    fig2 = plot_multi_replicates(msim2)

    for fig, sim in zip([fig0, fig1, fig2], [msim0.sims[0], msim1.sims[0], msim2.sims[0]]):
        fig.axes[2].vlines([start_intv], ax3_ylims[0], ax3_ylims[1],
                           linestyle="--", color="dimgray", zorder=0)
        fig.axes[2].vlines([end_intv], ax3_ylims[0], ax3_ylims[1],
                           linestyle="--", color="dimgray", zorder=0)
        fig.axes[0].set_ylim(ax1_ylims)
        fig.axes[1].set_ylim(ax2_ylims)
        fig.axes[2].set_ylim(ax3_ylims)
        fig.suptitle(f"Sim: {sim.label}")
        fig.tight_layout()

    msims = fp.MultiSim.merge([msim0, msim1, msim2], base=False)
    msims.plot()

    return fig0, fig1, fig2


def plot_multi_replicates(msimx):
    tvec = msimx.sims[0].tvec
    data_num = np.zeros((len(tvec), len(msimx.sims)))
    data_perc = np.zeros_like(data_num)
    data_n_female = np.zeros_like(data_num)

    for idx, sim in enumerate(msimx.sims):
        anlz = sim.get_analyzers()[0]
        data_num[:, idx] = anlz.data_num
        data_perc[:, idx] = anlz.data_perc
        data_n_female[:, idx] = anlz.data_n_female

    data_num = np.quantile(data_num, q=0.5, axis=1)
    data_perc = np.quantile(data_perc, q=0.5, axis=1)
    data_n_female = np.quantile(data_n_female, q=0.5, axis=1)

    fig = make_fig(tvec, data_num, data_perc, data_n_female)
    return fig


def make_fig(tvec, data_num, data_perc, data_n_female, state_name="on_contra"):
    colors = ["steelblue", "slategray", "black"]
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax2 = ax1.twinx()
    ax3 = ax1.twinx()

    ax1.spines["left"].set_color(colors[0])
    ax1.tick_params(axis="y", labelcolor=colors[0])

    ax2.spines["right"].set_color(colors[1])
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.tick_params(axis="y", labelcolor=colors[1])

    ax3.yaxis.tick_left()
    ax3.spines["left"].set_position(('outward', 70))
    ax3.spines["left"].set_color(colors[2])
    ax3.yaxis.set_label_position("left")
    ax3.tick_params(axis="y", labelcolor=colors[2])

    ax1.plot(tvec, data_num, color=colors[0])
    ax2.plot(tvec, data_perc, color=colors[1])
    ax3.plot(tvec, data_n_female, color=colors[2])

    ax1.set_xlabel('Year')
    ax1.set_ylabel(f'Number of women who are {state_name}',
                   color=colors[0])
    ax2.set_ylabel(f'percentage (%) of women who are {state_name} \n (denominator=num living women all ages)', color=colors[1])
    ax3.set_ylabel(f'Number of women alive, all ages', color=colors[2])
    fig.tight_layout()
    return fig


# Run a single repetition of each sim, plots results
msim = run_single_replicate_multisim()

# Run n_runs of each sim, makes figs
#run_multi_replicate_multisim(n_runs=10)
plt.show()
