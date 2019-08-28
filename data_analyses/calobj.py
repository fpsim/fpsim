import os
import pylab as pl
import pandas as pd
import sciris as sc


def load(*args, **kwargs):
    ''' Tiny alias to sc.loadobj() to load saved calendar objects '''
    cal = sc.loadobj(*args, **kwargs)
    assert isinstance(cal, CalObj)
    return cal


class CalObj(sc.prettyobj):
    '''
    Class for contraceptive calendar data/methods.
    
    Examples
    -------
    # Load data
    >>> calobj = CalObj(filename='data/DHS/NGIR6ADT/NGIR6AFL.DTA', which='DHS6')
    
    # Plot data
    >>> calobj.plot_transitions()
    '''
    
    def __init__(self, filename=None, which='DHS6'):
        '''
        Create object with the mapping, and load the data if supplied
        '''
        self.which = which
        self.originaldatafile = None
        self.filename = None
        self.mapping = None
        self.nmethods = -1
        self._set_mapping() # Could be here, but cleaner to separate out
        if filename is not None:
            self.load(filename)
        return

    def _set_mapping(self):
        '''
        Map DHS entries to numbers and descriptions, see 
        https://dhsprogram.com/data/calendar-tutorial/ module 2
        
        Entries represent: original (e.g. 'C'), numeric (11), short ('FCond'), full ('Female condom')
        '''
        if self.which == 'DHS6': # DHS6 or DHS7
            self.mapping = sc.odict({
                     ' ': [-1, 'Miss',  'Missing'],
                     '0': [ 0, 'None',  'Non-use'],
                     '1': [ 1, 'Pill',  'Pill'],
                     '2': [ 2, 'IUD',   'IUD'],
                     '3': [ 3, 'Injct', 'Injectables'],
                     '4': [ 4, 'Diaph', 'Diaphragm'],
                     '5': [ 5, 'Cond',  'Condom'],
                     '6': [ 6, 'FSter', 'Female sterilization'],
                     '7': [ 7, 'MSter', 'Male sterilization'],
                     '8': [ 8, 'Rhyth', 'Rhythm'],
                     '9': [ 9, 'Withd', 'Withdrawal'],
                     'B': [10, 'Birth', 'Birth'],
                     'C': [11, 'FCond', 'Female condom'],
                     'F': [12, 'Foam',  'Foam and jelly'],
                     'K': [13, 'Unknw', 'Unknown'],
                     'L': [14, 'Lact',  'Lactational amenorrhea'],
                     'M': [15, 'OModr', 'Other modern'],
                     'N': [16, 'Impla', 'Implants'],
                     'P': [17, 'Preg',  'Pregnancy'],
                     'T': [18, 'Term',  'Termination'],
                     'W': [19, 'OTrad', 'Other traditional']
                    })
        self.nmethods = len(self.mapping)
        return
    
    @property
    def numkeys(self):
        return pl.array([val[0] for val in self.mapping.values()])
    
    @property
    def shortkeys(self):
        return [val[1] for val in self.mapping.values()]
    
    @property
    def longkeys(self):
        return [val[2] for val in self.mapping.values()]
    
    
    def load(self, filename='', rawlines=None, which='DHS6'):
        ''' Load a contraceptive calendar from the original STATA file or one of several processed forms '''
        # Load a preprocessed object file
        if filename.endswith('cal'):
            print('Loading saved CalObj...')
            obj = sc.loadobj(filename)
            if not isinstance(obj, CalObj):
                raise Exception(f'Not sure what to do with an object of type {type(obj)}, expecting CalObj')
            for attr in ['cal', 'mapping', 'nmethods']:
                value = getattr(obj, attr)
                setattr(self, attr, value)
        
        elif filename.endswith('obj'):
            print('Loading calendar data from object...')
            obj = sc.loadobj(filename)
            if sc.checktype(obj, 'array'):
                self.cal = obj # The loaded object is just the calendar
            else:
                raise Exception(f'Not sure what to do with an object of type {type(obj)}, expecting array')
            
        # Load a calendar that has been exported directly to a text array
        elif filename.endswith('txt') or rawlines is not None:
            print('Loading calendar data from text...')
            
            # Load the string
            if rawlines is None:
                with open(filename) as f:
                    rawlines = f.readlines()
            
            # Parse the string
            cal = []
            for l,line in enumerate(rawlines):
                sc.percentcomplete(l, len(rawlines))
                cal.append([])
                for char in line:
                    try:
                        number = self.mapping[char][0] 
                        cal[-1].append(number)
                    except Exception as E:
                        if char not in ['\n']: # Skip space and newline, we know to ignore those
                            raise Exception(f'Could not parse character "{char}" on line {l} ({str(E)})')
            self.cal = pl.array(cal)
        
        # Load directly from the STATA file
        elif filename.endswith('dta') or filename.endswith('DTA'):
            print('Loading calendar data from Stata...')
            df = pd.read_stata(filename, convert_categoricals=False)
            rawlines = df['vcal_1'].to_list()
            self.load(rawlines=rawlines) # Call this function again, loading as a string this time
        
        # Handle exceptions
        else:
            raise Exception(f'File must end in cal, obj, txt, or dta: {filename} not recognized')
        
        print(f'Data loaded from {filename}.')
        self.originaldatafile = filename
        return
    
    
    def save(self, filename=None):
        if filename is None:
            basename = os.path.splitext(self.filename)
            filename = basename + '.cal'
        self.filename = filename
        sc.saveobj(filename, self, verbose=True)
        return filename
    
    
    def plot_transitions(self, figsize=None):
        if figsize is None: figsize = (16,14)
        fig = pl.figure(figsize=figsize)
        return fig
            