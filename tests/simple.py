import fpsim as fp
import fp_analyses as fa


n = 1000

pars = fa.senegal_parameters.make_pars()
pars['n'] = n
sim = fp.Sim(pars)
sim.run()
sim.plot()