"""
Test running sims
"""

import fpsim as fp
import sciris as sc
import pylab as pl

# par_kwargs = dict(n_agents=1000, start_year=1960, end_year=2020, seed=1, verbose=1)
par_kwargs = dict(n_agents=500, start=2000, stop=2010, rand_seed=1, verbose=-1)


def test_simple():
    sc.heading('Test simplest possible FPsim run')
    sim = fp.Sim(location='kenya')  # NB it should be possible to run without any arguments
    sim.run()
    sim.plot()
    return sim


def test_simple_choice(location='kenya'):
    sc.heading('Method choice is based on age & previous method')

    # Make & run sim
    import numpy as np
    pars = fp.pars(location=location, **par_kwargs)
    cm_pars = dict(
        prob_use_trend_par=0.1,
        force_choose=False,
        method_weights=np.array([0.1, 2, 0.5, 0.5, 2, 1, 1.5, 0.5, 5])
    )
    method_choice = fp.SimpleChoice(pars=cm_pars, location=location, methods=sc.dcp(fp.Methods))
    sim = fp.Sim(sim_pars=par_kwargs, fp_pars=pars, contraception_module=method_choice, analyzers=fp.cpr_by_age())
    sim.run()

    # Plots
    fig, axes = pl.subplots(2, 2, figsize=(10, 7))
    axes = axes.ravel()
    age_bins = [18, 20, 25, 35, 50, 99] # Added 99 to capture the Total CPR
    colors = sc.vectocolor(age_bins)
    cind = 0

    # mCPR
    ax = axes[0]
    ax.plot(sim.results.cpr.timevec, sim.results.cpr.values)
    ax.set_ylim([0, 1])
    ax.set_ylabel('CPR')
    ax.set_title('CPR')

    # mCPR by age
    ax = axes[1]
    for result in sim.results['cpr_by_age'].all_results:
        ax.plot(result.timevec, result.values, label=result.full_label, color=colors[cind])
        cind += 1
    ax.legend(loc='best', frameon=False)
    ax.set_ylim([0, 1])
    ax.set_ylabel('CPR')
    ax.set_title('CPR')

    # Plot method mix
    ax = axes[2]
    oc = sim.people.on_contra.uids
    method_props = [len(oc[sim.people.method[oc] == i])/len(oc) for i in range(1, 10)]
    method_labels = [m.name for m in sim.fp_pars['contraception_module'].methods.values() if m.label != 'None']
    ax.bar(method_labels, method_props)
    ax.set_ylabel('Proportion among all users')
    ax.set_title('Contraceptive mix')

    sc.figlayout()
    pl.show()

    return sim


def test_mid_choice(location='kenya'):
    sc.heading('Test sim with default contraceptive choice module')

    # Define new modules
    ms = fp.StandardChoice(location=location, methods=sc.dcp(fp.Methods))
    edu = fp.Education(location=location)

    # Define pars
    pars = fp.pars(location=location, **par_kwargs)

    # Make and run sim
    s = fp.Sim(pars, contraception_module=ms, education_module=edu)

    return s


if __name__ == '__main__':

    s0 = test_simple()
    s1 = test_simple_choice('kenya')    # TODO: check with senegal and ethiopia as well
    s2 = test_mid_choice('kenya')  # TODO: check with senegal and ethiopia as well
    print('Done.')
