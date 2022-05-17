'''
Simplest possible FPsim run.
'''

import fpsim as fp

sim = fp.Sim(n=1000, start=1950, end=2050)
sim.run()
sim.plot()