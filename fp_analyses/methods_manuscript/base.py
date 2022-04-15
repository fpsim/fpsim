'''
Helper functions for manuscript output repo
'''

import os

def abspath(path, *args):
    '''
    Turn a relative path into an absolute path. Accepts a
    list of arguments and joins them into a path.

    Example:

        import senegal_parameters as sp
        figpath = sp.abspath('figs', 'myfig.png')
    '''
    cwd = os.path.abspath(os.path.dirname(__file__))
    output = os.path.join(cwd, path, *args)

    return output

