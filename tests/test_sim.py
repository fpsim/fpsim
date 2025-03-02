"""
Test running sims
"""

import fpsim as fp
import sciris as sc
import pylab as pl
import numpy as np

do_plot=False

# par_kwargs = dict(n_agents=1000, start_year=1960, end_year=2020, seed=1, verbose=1)
par_kwargs = dict(n_agents=500, start_year=2000, end_year=2010, seed=1, verbose=-1)


def test_simple(do_plot=do_plot):
    sc.heading('Test simplest possible FPsim run')
    sim = fp.Sim()
    sim.run()
    if do_plot:
        sim.plot()
    return sim


def test_simple_choice(do_plot=do_plot):
    sc.heading('Method choice is based on age & previous method')

    # Make & run sim
    sims = sc.autolist()
    for location in ['kenya', 'ethiopia', 'senegal']:
        pars = fp.pars(location=location, **par_kwargs)
        method_choice = fp.SimpleChoice(location=location, methods=sc.dcp(fp.Methods))
        sim = fp.Sim(pars, contraception_module=method_choice, analyzers=fp.cpr_by_age())
        sims += sim

    for sim in sims:
        sim.run()
        print(f'✓ (successfully ran SimpleChoice for {sim.location})')

        if do_plot:

            # Plots
            fig, axes = pl.subplots(2, 2, figsize=(10, 7))
            axes = axes.ravel()
            age_bins = [18, 20, 25, 35, 50]
            colors = sc.vectocolor(age_bins)
            cind = 0

            # mCPR
            ax = axes[0]
            ax.plot(sim.results.t, sim.results.cpr)
            ax.set_ylim([0, 1])
            ax.set_ylabel('CPR')
            ax.set_title('CPR')

            # mCPR by age
            ax = axes[1]
            for alabel, ares in sim['analyzers'].results.items():
                ax.plot(sim.results.t, ares, label=alabel, color=colors[cind])
                cind += 1
            ax.legend(loc='best', frameon=False)
            ax.set_ylim([0, 1])
            ax.set_ylabel('CPR')
            ax.set_title('CPR')

            # Plot method mix
            ax = axes[2]
            oc = sim.people.filter(sim.people.on_contra)
            method_props = [sc.safedivide(len(oc.filter(oc.method == i)), len(oc)) for i in range(1, 10)]
            method_labels = [m.name for m in sim.contraception_module.methods.values() if m.label != 'None']
            ax.bar(method_labels, method_props)
            ax.set_ylabel('Proportion among all users')
            ax.set_title('Contraceptive mix')

            sc.figlayout()
            pl.show()

    return sims


def test_mid_choice():
    sc.heading('Test sims with default contraceptive choice module')

    sims = sc.autolist()
    for location in ['kenya', 'ethiopia', 'senegal']:
        ms = fp.StandardChoice(location=location, methods=sc.dcp(fp.Methods))
        edu = fp.Education(location=location)
        pars = fp.pars(location=location, **par_kwargs)
        s = fp.Sim(pars, contraception_module=ms, education_module=edu)
        sims += s

    for sim in sims:
        sim.run()
        print(f'✓ (successfully ran StandardChoice for {sim.location})')

    return sims


if __name__ == '__main__':

    do_plot = True

    s0 = test_simple(do_plot)
    sims1 = test_simple_choice(do_plot)
    sims2 = test_mid_choice()
    print('Done.')
