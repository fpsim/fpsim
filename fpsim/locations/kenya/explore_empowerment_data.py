'''
A script to visualise empowerment data after it's been loaded and processed
by data functions in kenya.py
'''

import numpy as np
import matplotlib.pyplot as plt

import sciris as sc

import fpsim.locations.kenya as kny
import fpsim.utils as fpu


empowerment_pars, empowerment_data = kny.make_empowerment_pars(return_data=True)

# How many empowerment metrics do we have
n_base_metrics = len(empowerment_pars["avail_metrics"])
comp_metrics = ['financial_autonomy', 'decision_making']

dm_cols = ["buy_decision_major", "buy_decision_daily",
           "buy_decision_clothes", "decision_health"]

fa_cols = ["has_savings", "has_fin_knowl", "has_fin_goals"]

n_plots = n_base_metrics + len(comp_metrics)

# Plotting stuff
nr, nc = sc.get_rows_cols(n_plots, ncols=2)
fig, axs = plt.subplots(nr, nc)
axes = [ax for ax in axs.flat]

n_draws = 100
n_ages = 35  # number of represented ages in the data [15-49]
data_draws = np.zeros((n_draws, n_plots, n_ages))
fa_draws = np.zeros((n_draws, n_ages))
dm_draws = np.zeros((n_draws, n_ages))
n_agents = 1000

def generate_composites(empwr_pars):
    age_inds = np.random.randint(0, high=35, size=n_agents)
    fa = np.zeros_like(age_inds, dtype=np.float64)
    dm = np.zeros_like(age_inds, dtype=np.float64)

    for metric in fa_cols:
        probs = empwr_pars[metric][age_inds]
        vals  = fpu.binomial_arr(probs).astype(float)
        temp = vals * empwr_pars["loadings"][metric]
        fa += temp

    for metric in dm_cols:
        probs = empwr_pars[metric][age_inds]
        vals = fpu.binomial_arr(probs).astype(float)
        dm += vals * empwr_pars["loadings"][metric]
    return fa, dm, age_inds + 15


def get_average_composites(fa, dm, ages):
    unique_ages = np.unique(ages)
    fa_by_age = np.array([np.median(fa[ages == age]) for age in unique_ages])
    dm_by_age = np.array([np.median(dm[ages == age]) for age in unique_ages])
    return fa_by_age, dm_by_age, unique_ages


# Showcase how the baseline empowerment probabilities are slightly different
# for multiple seeds.
for draw_i, seed in enumerate(np.random.randint(21, size=n_draws)):
    empowerment_pars = kny.make_empowerment_pars(seed=seed)
    m_i = -1
    c_i = -1
    fa_draw, dm_draw, ages = generate_composites(empowerment_pars)
    fa, dm, age_c = get_average_composites(fa_draw, dm_draw, ages)

    fa_draws[draw_i, :] = fa
    dm_draws[draw_i, :] = dm

    for ax, metric in zip(axes, empowerment_pars["avail_metrics"]+comp_metrics):
        m_i += 1
        if metric not in comp_metrics:
            data = empowerment_pars[metric]
            data_draws[draw_i, m_i, :] = data
        else:
            c_i += 1

        if draw_i == n_draws-1:
            # Dummy plot to get the label
            if metric not in comp_metrics:
                ax.plot(empowerment_pars["avail_ages"], data, 'o', ms=2, c='b',
                        alpha=0.4, label=f"Draws with {n_draws} different seeds")
            # Pick axes background color
            if metric in dm_cols:
                ax.set_facecolor('oldlace')
            if metric in fa_cols:
                ax.set_facecolor('aliceblue')

            if metric in comp_metrics:
                mean_estimates = np.nan*np.ones(n_ages)
                se_vals = np.nan*np.ones(n_ages)
                ax.set_title(f"Composite: {metric}")
                if metric in ["financial_autonomy"]:
                    ax.set_facecolor('powderblue')
                if metric in ["decision_making"]:
                    ax.set_facecolor('peachpuff')
            else:
                # Empirical data from csv file
                mean_estimates = empowerment_data[f"{metric}.mean"].to_numpy()
                se_vals = empowerment_data[f"{metric}.se"].to_numpy()
                ax.set_title(f"Metric: {metric}")
                #ax.set_title(f"Prob({metric}|age).\n  Draws from ~N({metric}.mean, {metric}.se).")


            # Plot empirical mean estimates and use SE for error val
            ax.errorbar(empowerment_data["age"], mean_estimates, yerr=se_vals,
                        ls='',
                        marker='s', ms=5, c='k',
                        label="Empirical data (estimate +/- se)")
            # Plot mean estimate from multiple draws
            data_mean = np.squeeze(data_draws[:, m_i, :])

            if metric not in comp_metrics:
                ax.errorbar(empowerment_pars["avail_ages"],
                            data_mean.mean(axis=0),
                            yerr=data_mean.std(axis=0) / np.sqrt(n_draws),
                            ls='', marker='s', ms=3, c='r',
                            alpha=0.7,
                            label=f"Mean and SE estimated from {n_draws} draws.")
            else:
                if metric in ["financial_autonomy"]:
                    me = np.median(fa_draws, axis=0)
                    se = fa_draws.std(axis=0) / np.sqrt(n_draws)

                if metric in ["decision_making"]:
                    me = np.median(dm_draws, axis=0)
                    se = dm_draws.std(axis=0) / np.sqrt(n_draws)

                ax.errorbar(age_c,
                            me,
                            yerr=se,
                            ls='', marker='s', ms=3, c='r',
                            alpha=0.7,
                            label=f"Mean and SE estimated from {n_draws} draws.")

            if m_i > 11:
                ax.set_xlabel("Age [years]")
            ax.set_ylabel(f"p(empowerment|age)")
            #ax.legend()

        else:
            if metric in ["financial_autonomy"]:
                ax.plot(age_c, fa, '.', zorder=-1,  ms=2, c='b', alpha=0.1)
                ax.set_ylim([0, 4])
            elif metric in ["decision_making"]:
                ax.plot(age_c, dm, '.', zorder=-1, ms=2, c='b', alpha=0.1)
                ax.set_ylim([0, 4])
            else:
                ax.plot(empowerment_pars["avail_ages"], data, '.', zorder=-1, ms=2, c='b', alpha=0.1)
                ax.set_ylim([0, 1])
#fig.tight_layout()
plt.show()
