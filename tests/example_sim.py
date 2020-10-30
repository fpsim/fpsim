# Simple example usage for FPsim

import fpsim as lfp
import fp_analyses.senegal_parameters as sp

doplot = True
dosave = False

pars = sp.make_pars()
pars['start_year'] = 2010 # Only do a partial run
sim = lfp.Sim(pars=pars)
sim.run()
if doplot:
    sim.plot(dosave=dosave)

print('Done.')

