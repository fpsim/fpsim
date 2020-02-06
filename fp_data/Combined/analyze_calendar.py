'''
Simple script to run analyses using the heavy-lifting CalObj class.
'''

import os
import pylab as pl
import fp_utils as fpu
import sciris as sc

# Comment/uncomment these to run different analyses
torun = [
    #'loadsave',
    'plot_matrix',
    #'plot_slice',
]

# Choose a case to analyze
case = 'Senegal-URHI'
#case = 'Senegal-DHS2010-11'
#case = 'Senegal-DHS2017'


sc.tic()

username = os.path.split(os.path.expanduser('~'))[-1]

scenarios = {
    'Senegal-URHI': {
        'filedict': {
            'dklein': os.path.join( os.getenv("HOME"), 'Dropbox (IDM)', 'URHI', 'Senegal', 'Endline', 'SEN_end_wm_match_20160505.dta'),
            #'cliffk': '/u/cliffk/idm/fp/data/DHS/NGIR6ADT/NGIR6AFL.DTA',
        },
        'cache' : os.path.join('data', f'{case}.cal'),
        'which' : 'URHI-brief',
        'key'   : 'cal_1',
    },
    'Senegal-DHS2017': {
        'filedict': {
            'dklein': os.path.join( os.getenv("HOME"), 'Dropbox (IDM)', 'FP Dynamic Modeling', 'DHS', 'Country data', 'Senegal', '2017', 'SNIR7ZDT', 'SNIR7ZFL.DTA'),
            #'cliffk': '/u/cliffk/idm/fp/data/DHS/NGIR6ADT/NGIR6AFL.DTA',
        },
        'cache': os.path.join('data', f'{case}.cal'),
        'which': 'DHS7',
        'key'  : 'vcal_1',
    },
    'Senegal-DHS2010-11': {
        'filedict': {
            'dklein': os.path.join( os.getenv("HOME"), 'Dropbox (IDM)', 'FP Dynamic Modeling', 'DHS', 'Country data', 'Senegal', '2010-11', 'SNIR61DT', 'SNIR61FL.DTA'),
            #'cliffk': '/u/cliffk/idm/fp/data/DHS/NGIR6ADT/NGIR6AFL.DTA',
        },
        'cache': os.path.join('data', f'{case}.cal'),
        'which': 'DHS7', # DHS6
        'key'  : 'vcal_1',
    },
}

scenario = scenarios[case]
cache = scenario['cache']
filedict = scenario['filedict']

try:
    filename = filedict[username]
except:
    raise Exception(f'User {username} not found among users {list(filedict.keys())}, cannot find data.')

def load_from_file():
    print(f'Loading calobj from file {filename}')
    calobj = fpu.CalObj(filename, which=scenario['which'], key=scenario['key']) # Create from saved data
    print(f'Saving to cache {cache}')
    calobj.save(cache)

    return calobj


if os.path.isfile(cache):
    try:
        print('Trying to load from cache {cache}...')
        calobj = fpu.CalObj(cache) # Load saved object
        print('Loaded presaved object...')
    except:
        print(f'Unable to load from cache {cache}, falling back to loading from file {filename}')
        calobj = load_from_file()
else:
    calobj = load_from_file()

if 'plot_matrix' in torun:
    #calobj.plot_transitions(projection='2d', figsize=(16,10))
    calobj.plot_prop(projection='2d', figsize=(8,8))
    basename = os.path.splitext(cache)[0]
    filename = basename + '.png'
    pl.savefig(filename)


if 'plot_slice' in torun:
    calobj.plot_slice(key='None')


sc.toc()
print('Done')

pl.show()
