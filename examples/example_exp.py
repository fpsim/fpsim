'''
Simple example usage for FPsim
'''

import fpsim as fp

# Set options
do_plot = True

# Run
exp = fp.Experiment()
exp.run()

if do_plot:
    exp.plot()
    exp.fit.plot()

print('Done.')


