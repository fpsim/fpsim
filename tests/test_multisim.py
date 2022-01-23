"""
Run tests on the multisim object.
"""

import sciris as sc
import fpsim as fp
import fp_analyses as fa

do_plot = 1


def test_multisim(do_plot=False):
    ''' Try running a multisim '''
    sc.heading('Testing multisim...')

    # Generate sims in a loop
    sims = []
    for i in range(3):
        exposure = 0.5 + 0.5*i # Run a sweep over exposure
        pars = fa.senegal_parameters.make_pars()
        pars['n'] = 500
        pars['verbose'] = 0.1
        pars['exposure_correction'] = exposure
        sim = fp.Sim(pars=pars, label=f'Exposure {exposure}')
        sims.append(sim)

    msim = fp.MultiSim(sims)
    msim.run() # Run sims in parallel
    msim.to_df() # Test to_df

    births = msim.results.births
    assert sum(births.low) < sum(births.high), 'Expecting the higher bound of births to be higher than the lower bound'

    if do_plot:
        msim.plot(plot_sims=True)
        msim.plot(plot_sims=False)

    return msim


if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    T = sc.tic()

    msim = test_multisim(do_plot=do_plot)

    print('\n'*2)
    sc.toc(T)
    print('Done.')