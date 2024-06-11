"""
Test running sims
"""

import fpsim as fp
import sciris as sc
import pylab as pl

par_kwargs = dict(n_agents=100, start_year=2000, end_year=2010, seed=1, verbose=1)


def test_simple():
    sc.heading('Test simplest possible FPsim run')
    sim = fp.Sim(location='test')
    sim.run()
    sim.plot()
    return sim


def test_simple_choice(location='kenya'):
    sc.heading('Method choice is based on age & previous method')

    # Make & run sim
    pars = fp.pars(location=location, **par_kwargs)
    method_choice = fp.SimpleChoice(pars=dict(prob_use_trend_par=0.1), location=location, methods=sc.dcp(fp.Methods))
    sim = fp.Sim(pars, contraception_module=method_choice, analyzers=fp.cpr_by_age())
    sim.run()

    # Plots
    fig, axes = pl.subplots(2, 1, figsize=(5, 7))
    axes = axes.ravel()
    age_bins = [18, 20, 25, 35, 50]
    colors = sc.vectocolor(age_bins)
    cind = 0

    # mCPR
    ax = axes[0]
    for alabel, ares in sim['analyzers'].results.items():
        ax.plot(sim.results.t, ares, label=alabel, color=colors[cind])
        cind += 1
    ax.legend(loc='best', frameon=False)

    ax.set_ylim([0, 1])
    ax.set_ylabel('CPR')
    ax.set_title('CPR')

    # Plot method mix
    ax = axes[1]
    oc = sim.people.filter(sim.people.on_contra)
    method_props = [len(oc.filter(oc.method == i))/len(oc) for i in range(1, 10)]
    method_labels = [m.name for m in sim.contraception_module.methods.values() if m.label != 'None']
    ax.bar(method_labels, method_props)
    ax.set_ylabel('Proportion among all users')
    ax.set_title('Contraceptive mix')

    sc.figlayout()
    pl.show()

    return sim


def test_empowered_choice():
    sc.heading('Test sim with empowerment')

    # Define new modules
    ms = fp.EmpoweredChoice(location='kenya')
    emp = fp.Empowerment(location='kenya')
    edu = fp.Education(location='kenya')

    # Define pars
    pars = fp.pars(location='kenya', **par_kwargs)

    # Make and run sim
    s = fp.Sim(pars, contraception_module=ms, empowerment_module=emp, education_module=edu)
    s.run()

    return s


if __name__ == '__main__':

    s0 = test_simple()
    s1 = test_simple_choice()
    s2 = test_empowered_choice()
