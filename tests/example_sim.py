# Simple example usage for FPsim

import fpsim as fp
import fp_analyses.senegal_parameters as sp

doplot = True
dosave = False

pars = sp.make_pars()
pars['n'] = 100 # Only do a partial run
sim = fp.Sim(pars=pars)
sim.run()
if doplot:
    sim.plot(dosave=dosave)

print('Done.')

