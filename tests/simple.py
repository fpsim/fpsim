'''
Simplest possible FPsim run.
'''

import fpsim as fp

sim = fp.Sim(n=1000, start_year=1920, end_year=2080)
sim.run()
sim.plot()