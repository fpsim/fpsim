import fpsim as fp
import fp_analyses as fa


kwargs = dict(
    analyzers = [
        fp.snapshot(timesteps=[0, 10, 100, 200]),
        fp.timeseries_recorder()
        ],
)

pars = fa.senegal_parameters.make_pars()
pars['n'] = 1000
pars['verbose'] = 0.1
pars['end_year'] = 2000
pars['exposure_correction'] = 3.0
pars.update(kwargs)
sim = fp.Sim(pars=pars)
sim.run()
sim.plot()
sim.people.plot(hist_args=dict(bins=50))
an = sim.pars['analyzers'][1]
an.plot()
