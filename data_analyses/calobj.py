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
    
    def __init__(self, filename=None, which='DHS6', skipmissing=True):
        '''
        Create object with the mapping, and load the data if supplied
        '''
        self.which = which
        self.originaldatafile = None
        self.filename = None
        self.mapping = None
        self.nmethods = None
        self.skipmissing = skipmissing
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
        self.nmethods = len(self.mapping) - self.skipmissing
        return
    
    def _parse_keys(self, ind):
        startind = 1 if self.skipmissing else 0
        output = [val[ind] for val in list(self.mapping.values())[startind:]]
        return output
    
    @property
    def numkeys(self):
        return pl.array(self._parse_keys(0))
    
    @property
    def shortkeys(self):
        return self._parse_keys(1)
    
    @property
    def longkeys(self):
        return self._parse_keys(2)
    
    
    def load(self, filename='', rawlines=None, which='DHS6'):
        ''' Load a contraceptive calendar from the original Stata file or one of several processed forms '''
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
        ''' Save object to disk '''
        if filename is None:
            basename = os.path.splitext(self.filename)
            filename = basename + '.cal'
        self.filename = filename
        sc.saveobj(filename, self, verbose=True)
        return filename
    
    
    def plot_transitions(self, figsize=None):
        ''' Plot all transitions in the contraception calendar '''
        zero_to_nan = False # Set zero rows
        always_log = False # Use log for both plots
        if figsize is None: figsize = (30,14)
        
        # Calculate
        counts = pl.zeros((self.nmethods, self.nmethods))
        for pp,person in enumerate(self.cal):
            sc.percentcomplete(pp, len(self.cal))
            for month in range(len(person)-1):
                previous = person[month]
                current = person[month+1]
                if previous != -1 and current != -1: # Transition occurred, data aren't missing
                    counts[current, previous] += 1
        
        # Calculate log counts for panel A
        log_counts = pl.log10(counts)
        
        # Calculate relative proportions for panel B
        rel_props = sc.dcp(counts)
        for m in range(self.nmethods):
            rel_props[m,m] = 0 # Remove diagonal
            totalcount = rel_props[m,:].sum()
            if totalcount:
                rel_props[m,:] /= totalcount # Normalize to percentage
            else:
                if zero_to_nan:
                    rel_props[m,:] = pl.nan
                    print(f'No count for {self.shortkeys[m]}')
        if always_log:
            rel_props = pl.log10(rel_props)
        else:
            rel_props *= 100 # Convert to percentage
            
        # Create figure and set tick marks on top
        fig = pl.figure(figsize=figsize)
        pl.rcParams['xtick.top'] = pl.rcParams['xtick.labeltop'] = True
        pl.rcParams['ytick.right'] = pl.rcParams['ytick.labelright'] = True
        
        
        # Plot total counts
        ax1 = fig.add_subplot(121)
        im1 = pl.imshow(log_counts, cmap=sc.parulacolormap()) # , edgecolors=[0.8]*3
        ax1.set_xticks(self.numkeys) # +0.5
        ax1.set_xticklabels(self.shortkeys)
        ax1.set_yticks(self.numkeys)
        ax1.set_yticklabels(self.shortkeys)
        ax1.set_title('Total number of transitions in calendar (log scale, white=0)', fontweight='bold')
        ca1 = fig.add_axes([0.05, 0.11, 0.03, 0.75])
        fig.colorbar(im1, cax=ca1)
        
        # Plot relative counts
        ax2 = fig.add_subplot(122)
        im2 = pl.imshow(rel_props, cmap='jet') # , edgecolors=[0.8]*3
        ax2.set_xticks(self.numkeys) # +0.5
        ax2.set_xticklabels(self.shortkeys)
        ax2.set_yticks(self.numkeys)
        ax2.set_yticklabels(self.shortkeys)
        ax2.set_title('Relative proportion of each transition, diagonal removed (%)', fontweight='bold')
        ca2 = fig.add_axes([0.95, 0.11, 0.03, 0.75])
        fig.colorbar(im2, cax=ca2)
        return fig
            