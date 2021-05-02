# Simple example usage for FPsim

import sciris as sc
import fpsim as fp
import fp_analyses.senegal_parameters as sp

doplot = True
dosave = False

sc.tic()
pars = sp.make_pars()
pars['n'] = 5000 # Only do a partial run
sim = fp.Sim(pars=pars)
sim.run()
if doplot:
    sim.plot(dosave=dosave)

sc.toc()
print('Done.')

