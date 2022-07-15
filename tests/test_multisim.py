"""
Run tests on the multisim object.
"""

import sciris as sc
import fpsim as fp

serial  = 0
do_plot = 1
sc.options(backend='agg') # Turn off interactive plots


def test_multisim(do_plot=do_plot):
    ''' Try running a multisim '''
    sc.heading('Testing multisim...')

    # Generate sims in a loop
    sims = []
    for i in range(3):
        exposure = 0.5 + 0.5*i # Run a sweep over exposure
        pars = fp.pars('test', exposure_factor=exposure)
        sim = fp.Sim(pars=pars, label=f'Exposure {exposure}')
        sims.append(sim)

    msim = fp.MultiSim(sims)
    msim.run(serial=serial) # Run sims in parallel
    msim.to_df() # Test to_df

    births = msim.results.births
    assert sum(births.low) < sum(births.high), 'Expecting the higher bound of births to be higher than the lower bound'

    if do_plot:
        msim.plot(plot_sims=True)
        msim.plot(plot_sims=False)
        msim.plot_age_first_birth()

    return msim


if __name__ == '__main__':
    sc.options(backend=None) # Turn on interactive plots
    with sc.timer(): # Start timing
        msim = test_multisim()
