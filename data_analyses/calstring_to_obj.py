'''
Convert the calendar text file to an object and save.


'''

import pylab as pl
import sciris as sc
import calobj as co

dataname = 'NGIR6A'
infile = f'{dataname}_calendar.txt'
outfile = f'{dataname}_calendar.obj'

calobj = co.CalObj()

# Load the string
with open(infile) as f:
    rawlines = f.readlines()

# Parse the string
data = []
for l,line in enumerate(rawlines):
    sc.percentcomplete(l, len(rawlines))
    data.append([])
    for char in line:
        try:
            number = calobj.mapping.DHS6[char][0]
            data[-1].append(number)
        except Exception as E:
            if char not in ['\n']: # Skip space and newline, we know to ignore those
                raise Exception(f'Could not parse character "{char}" on line {l} ({str(E)})')


sc.saveobj(outfile, pl.array(data))

print('Done.')
