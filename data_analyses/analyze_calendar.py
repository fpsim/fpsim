'''
Simple script to run analyses using the heavy-lifting CalObj class.
'''

import calobj as co
import pylab as pl
import sciris as sc

pl.rcParams['xtick.top'] = pl.rcParams['xtick.labeltop'] = True
pl.rcParams['ytick.right'] = pl.rcParams['ytick.labelright'] = True

torun = [
#        'loadsave',
        'plot',
        ]

filename = '/u/cliffk/idm/fp/fp_analyses/data_analyses/data/nigeria_cal.obj'

if 'loadsave' in torun:
    print('Loading original data and saving...')
    filename = '/u/cliffk/idm/fp/data/DHS/NGIR6ADT/NGIR6AFL.DTA' # Nigeria 2013 DHS
    calobj = co.CalObj(filename)
    calobj.save('data/nigeria_cal.obj')
else:
    print('Loading presaved object...')
    #calobj = co.load(filename) # Load saved object
    calobj = co.CalObj(filename) # Create from saved data
    
if 'plot' in torun:
#    calobj.plot_transitions()
    
    # Calculate
    offset = -calobj.numkeys.min() # Remove the minimal offset
    counts = pl.zeros((calobj.nmethods, calobj.nmethods))
    for pp,person in enumerate(calobj.cal):
        sc.percentcomplete(pp, len(calobj.cal))
        for month in range(len(person)-1):
            previous = person[month]
            current = person[month+1]
#            if previous != current: # Transition occurred!
            counts[previous+offset, current+offset] += 1
#    for m in range(calobj.nmethods):
#        counts[m,:] *= 
    
    # Plot
    data = pl.log10(counts)
    figsize = (16,14)
    fig = pl.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    pl.imshow(data) # , edgecolors=[0.8]*3
    ax.set_xticks(calobj.numkeys+offset) # +0.5
    ax.set_xticklabels(calobj.shortkeys)
    ax.set_yticks(calobj.numkeys+offset)
    ax.set_yticklabels(calobj.shortkeys)
    pl.colorbar()
    
    

print('Done')