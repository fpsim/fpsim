import fpsim as fp
import fp_analyses as fa
import pylab as pl


kwargs = dict(
    analyzers = [
        # fp.snapshot(timesteps=[0, 10, 100, 200]),
        # fp.timeseries_recorder(),
        fp.age_pyramids(),
        ],
)

pars = fa.senegal_parameters.make_pars()
pars['n'] = 10000
pars['verbose'] = 0.1
pars['end_year'] = 2100
pars['exposure_correction'] = 1.0
pars.update(kwargs)
sim = fp.Sim(pars=pars)
sim.run()
sim.plot()
sim.people.plot(hist_args=dict(bins=50))
ap = sim.pars['analyzers'][0]
ap.plot()
pl.figure()
pl.plot(ap.data[-1,:])