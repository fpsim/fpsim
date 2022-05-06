'''
Simplest possible FPsim run.
'''

import fpsim as fp

sim = fp.Sim(n=1000)
sim.run()
sim.plot()