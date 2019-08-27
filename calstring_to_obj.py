'''
Convert the calendar text file to an object and save.


'''

import sciris as sc

dataname = 'NGIR6A'
infile = f'{dataname}_calendar.txt'
outfile = f'{dataname}_calendar.obj'

# Map DHS entries to numbers and descriptions
mapping = {
         '0': [ 0,''],
         '1': [ 1,''],
         '2': [ 2,''],
         '3': [ 3,''],
         '4': [ 4,''],
         '5': [ 5,''],
         '6': [ 6,''],
         '7': [ 7,''],
         '8': [ 8,''],
         '9': [ 9,''],
         'B': [10,''],
         'C': [11,''],
         'F': [12,''],
         'K': [13,''],
         'L': [14,''],
         'M': [15,''],
         'N': [16,''],
         'P': [17,''],
         'T': [18,''],
         'W': [19,'']}
        }

# Load the string
with open(infile) as f:
    rawstring = f.readlines()

print('Done.')
