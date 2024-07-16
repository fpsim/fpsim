'''
Test births, conceptions, etc.
'''

import numpy as np
import sciris as sc
import fpsim as fp
import pytest



# Parameters
do_plot  = 1 # Whether to do plotting in interactive mode
sc.options(backend='agg') # Turn off interactive plots


def ok(string):
    ''' Print out a successful test nicely '''
    return sc.printgreen(f'✓ {string}\n')

def test_plot():
    ''' Test Experiment plotting '''
    sc.heading('Testing Experiment plotting...')
    if do_plot:
        pars = fp.pars(location='kenya')
        exp = fp.Experiment(pars)
        exp.run()
        exp.plot()
        ok('Plotting succeeded')
    return exp

def test_regional_exp():
    ''' Test Experiment using a region as location'''
    sc.heading('Testing regional Experiment')
    if do_plot:
        pars = fp.pars(location='amhara')
        exp = fp.Experiment(pars)
        exp.run()
        exp.plot()
        ok('Regional experiment succeeded')
    return exp

if __name__ == '__main__':

    sc.options(backend=None) # Turn on interactive plots

    with sc.timer():
        exp = test_plot()
