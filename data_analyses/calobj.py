'''
Class for contraceptive calendar data/methods
'''

import sciris as sc
import pandas as pd

class CalObj(object):
    
    def __init__(self, filename=None):
        '''
        Create object with the mapping, and load the data if supplied
        '''
        self.set_mapping()
        if filename is not None:
            self.load(filename)

    @classmethod
    def set_mapping(self, which):
        '''
        Map DHS entries to numbers and descriptions, see 
        https://dhsprogram.com/data/calendar-tutorial/ module 2
        '''
        self.mapping = sc.objdict()
        self.mapping.DHS6 = sc.odict({
                 ' ': [-1,'Missing'],
                 '0': [ 0,'Non-use'],
                 '1': [ 1,'Pill'],
                 '2': [ 2,'IUD'],
                 '3': [ 3,'Injectables'],
                 '4': [ 4,'Diaphragm'],
                 '5': [ 5,'Condom'],
                 '6': [ 6,'Female sterilization'],
                 '7': [ 7,'Male sterilization'],
                 '8': [ 8,'Rhythm'],
                 '9': [ 9,'Withdrawal'],
                 'B': [10,'Birth'],
                 'C': [11,'Female condom'],
                 'F': [12,'Foam and jelly'],
                 'K': [13,'Unknown'],
                 'L': [14,'Lactational amenorrhea'],
                 'M': [15,'Other modern'],
                 'N': [16,'Implants'],
                 'P': [17,'Pregnancy'],
                 'T': [18,'Termination'],
                 'W': [19,'Other traditional']
                })
        return None
    
    
    def load(self, filename='', rawlines=None, which='DHS6'):
        ''' Load a contraceptive calendar from the original STATA file or one of several processed forms '''
        # Load a preprocessed object file
        if filename.endswith('obj'):
            self.data = sc.loadobj(filename)
            
        # Load a calendar that has been exported directly to a text array
        elif filename.endswith('txt') or rawlines is not None:
            
            # Load the string
            if rawlines is not None:
                with open(filename) as f:
                    rawlines = f.readlines()
            
            # Parse the string
            data = []
            for l,line in enumerate(rawlines):
                sc.percentcomplete(l, len(rawlines))
                data.append([])
                for char in line:
                    try:
                        number = self.mapping[which][char][0] 
                        data[-1].append(number)
                    except Exception as E:
                        if char not in ['\n']: # Skip space and newline, we know to ignore those
                            raise Exception(f'Could not parse character "{char}" on line {l} ({str(E)})')
        
        # Load directly from the STATA file
        elif filename.endswith('dta') or filename.endswith('DTA'):
            df = pd.read_stata(filename, convert_categoricals=False)
            rawlines = df['vcal_1'].to_list()
            self.load(rawlines=rawlines) # Call this function again, loading as a string this time
        
        # Handle exceptions
        else:
            raise Exception(f'File must end in obj, txt, or dta: {filename} not recognized')
            