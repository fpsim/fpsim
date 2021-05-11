import fpsim as fp
import fp_analyses as fa
import numpy as np
import pylab as pl

class rec(fp.Analyzer):

    def __init__(self):
        super().__init__()
        self.t = []
        self.y = []

    def apply(self, sim):
        self.t.append(sim.t)
        self.y.append(np.mean(sim.people.sexually_active))


kwargs = dict(
    analyzers = [fp.snapshot(timesteps=[0, 10, 100, 200]), rec()],
)

pars = fa.senegal_parameters.make_pars()
pars['n'] = 10000
pars['verbose'] = 0.1
pars['end_year'] = 2100
pars['exposure_correction'] = 3.0
pars.update(kwargs)
sim = fp.Sim(pars=pars)
sim.run()
sim.plot()
sim.people.plot(hist_args=dict(bins=50))
an = sim.pars['analyzers'][1]
pl.figure(); pl.plot(an.t, an.y)