# Simple example usage for LEMOD-FP

import lemod_fp as lfp

doplot = True
dosave = False

sim = lfp.Sim()
sim.run()
if doplot:
    sim.plot(dosave=dosave)

print('Done.')

