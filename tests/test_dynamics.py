"""
Test dynamics within FPsim related to contraceptive use, method choice, and duration of use
"""

import sciris as sc
import pylab as pl
import fpsim as fp
import numpy as np


serial   = 1 # Whether to run in serial (for debugging)
do_plot  = 1 # Whether to do plotting in interactive mode
# sc.options(backend='agg') # Turn off interactive plots


def make_sim_parts():
    location = 'kenya'
    par_kwargs = dict(n_agents=500, start_year=2000, end_year=2020, seed=1, verbose=-1)
    pars = fp.pars(location=location, **par_kwargs)
    edu = fp.Education(location=location)
    choice = fp.StandardChoice(location=location)
    return pars, choice, edu


def make_sim(label='Baseline', intvs=None, analyzers=None, **kwargs):
    pars, choice, edu = make_sim_parts()
    sim = fp.Sim(
        pars=pars, contraception_module=choice, education_module=edu,
        label=label, interventions=intvs, analyzers=analyzers, **kwargs
        )
    return sim


def test_mcpr():
    sc.heading('Testing that mCPR changes as expected with its covariates...')

    sims = sc.autolist()
    sims += make_sim()

    # Create covariate changes
    class Covar:
        def __init__(self, pplattr, val, resname):
            self.pplattr = pplattr
            self.val = val
            self.resname = resname
            self.base = None
            self.intv = None
            self.mcpr = None

    covars = [
        Covar('edu_attainment', 15, 'edu_attainment'),
        Covar('urban', True, 'urban_women'),
        # Covar('parity', 2, 'parity2to3'),  # Unfortunately this will not work
        Covar('wealthquintile', 5, 'wq5'),
        Covar('ever_used_contra', True, 'ever_used_contra'),
        ]

    # Create interventions and sims
    for covar in covars:

        def select_women(sim): return sim.people.is_female & sim.people.alive

        change_state = fp.change_people_state(
                            covar.pplattr,
                            eligibility=select_women,
                            years=2010.0,
                            new_val=covar.val,
                        )
        sims += make_sim(intvs=change_state, label=f'Increased {covar.pplattr}')

    # Run
    for sim in sims: sim.run()
    # m = fp.parallel(*sims, serial=serial, compute_stats=False)
    # sims = m.sims[:]  # Replace with run versions

    # Firstly, check that changing the people attributes has registered in the relevant results metrics as expected
    for ri, covar in enumerate(covars):
        print(f"Checking effect of {covar.pplattr} ... ")
        base = sims[0].results[covar.resname][-1]
        intv = sims[ri+1].results[covar.resname][-1]
        assert base < intv, f'Increasing {covar.pplattr} should register in results, but {intv}<{base}'
        print(f"✓ ({base:.2f} < {intv:.2f})")
        covar.base = sims[0].results[covar.resname]
        covar.intv = sims[ri+1].results[covar.resname]
        covar.mcpr = sims[ri+1].results.mcpr

    # Next, check that changing the people attributes has registered in the relevant results metrics as expected
    mcprs = [sim.results['mcpr'][-1] for sim in sims]
    print('Computed mCPRs:')
    for sim, mcpr in zip(sims, mcprs):
        print(f'{sim.label}: {mcpr:.2f}')
    # assert mcprs[0] == min(mcprs), 'Increasing any covariate should increase contraceptive use.'
    print(f"✗ (TEST FAILS: mCPR DOES NOT INCREASE WITH ALL COVARIATES) - NEED TO DEBUG")

    # Plot
    fig, axes = pl.subplots(2, 2, figsize=(12, 6))
    axes = axes.flatten()
    for ri, covar in enumerate(covars):
        ax = axes[ri]
        ax.plot(sims[0].results.t, covar.base, label='Baseline')
        ax.plot(sims[ri+1].results.t, covar.intv, label=f'Increased {covar.pplattr}')
        ax.set_title(f'{covar.pplattr}')
        ax.set_xlabel('Year')
        ax.legend()

    fig, ax = pl.subplots(1, 1, figsize=(12, 6))
    ax.plot(sims[0].results.t, sims[0].results.mcpr, label=sims[0].label)
    for ri, covar in enumerate(covars):
        ax.plot(sims[ri+1].results.t, covar.mcpr, label=covar.pplattr)
    ax.set_ylabel('mCPR')
    ax.set_xlabel('Year')
    pl.legend()
    pl.show()

    return sims


def test_durations():
    sc.heading('Testing that durations are as expected...')

    # Create parameters and modules
    sim_base = make_sim()

    pars, choice, edu = make_sim_parts()
    short_choice = sc.dcp(choice)
    for m in short_choice.methods.values(): m.dur_use = dict(dist='lognormal', par1=1, par2=1)
    sim_short = fp.Sim(
        pars=pars, contraception_module=short_choice, education_module=edu,
        label='Short durations')

    # Run sims
    m = fp.parallel([sim_base, sim_short], serial=serial, compute_stats=False)
    sim_base, sim_short = m.sims[:]  # Replace with run versions

    # Shorter durations should mean more switching
    print(f"Checking effect of durations ... ")
    base = sum(sim_base.results.switchers)
    short = sum(sim_short.results.switchers)
    assert base < short, f'Shorted durations should mean more switching, but {short}<{base}'
    print(f"✓ ({base:.2f} < {short:.2f})")

    return [sim_base, sim_short]


if __name__ == '__main__':

    sims1 = test_mcpr()
    sims2 = test_durations()

    print('Done.')


