import os
import pylab as pl
import pandas as pd
import sciris as sc


def load(*args, **kwargs):
    ''' Tiny alias to sc.loadobj() to load saved calendar objects '''

    import sys
    import types
    import fp_data

    # First, add any placeholder modules that have been subsequently removed
    fp_data.DHS = types.ModuleType('DHS')
    fp_data.DHS.calobj = types.ModuleType('calobj')
    sys.modules['fp_data.DHS'] = fp_data.DHS
    sys.modules['fp_data.DHS.calobj'] = fp_data.DHS.calobj
    fp_data.DHS.calobj.CalObj = CalObj

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

    def __init__(self, filename=None, which='DHS6', key='vcal_1', skipmissing=True):
        '''
        Create object with the mapping, and load the data if supplied
        '''
        self.which = which
        self.key = key
        self.skipmissing = skipmissing
        self.originaldatafile = None
        self.filename = None
        self.mapping = None
        self.nmethods = None
        self.results = None
        self._set_mapping() # Could be here, but cleaner to separate out
        if filename is not None:
            self.load(filename)
            self.make_results()
        return

    def _set_mapping(self):
        '''
        Map DHS entries to numbers and descriptions, see
        https://dhsprogram.com/data/calendar-tutorial/ module 2

        Entries represent: original (e.g. 'C'), numeric (11), short ('FCond'), full ('Female condom')
        '''
        if self.which == 'DHS6':
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
        elif self.which == 'DHS7': # See recode version -- contraceptive calendar tutorial PDF p. 25, table 3
            self.mapping = sc.odict({
                ' ': [-1, 'Miss',  'Missing'],
                'B': [ 0, 'Birth', 'Birth'],
                'T': [ 1, 'Term',  'Termination'],
                'P': [ 2, 'Preg',  'Pregnancy'],
                '0': [ 3, 'None',  'Non-use'],
                '1': [ 4, 'Pill',  'Pill'],
                '2': [ 5, 'IUD',   'IUD'],
                '3': [ 6, 'Injct', 'Injectables'],
                '4': [ 7, 'Diaph', 'Diaphragm'],
                '5': [ 8, 'Cond',  'Condom'],
                '6': [ 9, 'FSter', 'Female sterilization'],
                '7': [10, 'MSter', 'Male sterilization'],
                '8': [11, 'Rhyth', 'Rhythm'],
                '9': [12, 'Withd', 'Withdrawal'],
                'W': [13, 'OTrad', 'Other traditional'],
                'N': [14, 'Impla', 'Implants'],
                'A': [15, 'Abst',  'Abstinence'],
                'L': [16, 'Lact',  'Lactational amenorrhea'],
                'C': [17, 'FCond', 'Female condom'],
                'F': [18, 'Foam',  'Foam and jelly'],
                'E': [19, 'Emerg', 'Emergency contraception'],
                'S': [20, 'SDays', 'Standard days'],
                'M': [21, 'OModr', 'Other modern'],
#                     '?': [22, 'Unknw', 'Unknown'], # Seems to be zero?
            })
        elif self.which == 'URHI-brief':
            self.mapping = sc.odict({
                ' ': [-1, 'Miss',  'Missing'],
                'N': [ 0, 'Birth', 'Birth'],
                'F': [ 1, 'Term',  'Termination'],
                'M': [ 1, 'Term',  'Termination'],
                'A': [ 1, 'Term',  'Termination'],
                'G': [ 2, 'Preg',  'Pregnancy'],
                '0': [ 3, 'None',  'No method'],
                '6': [ 4, 'Pill',  'Pill'],
                '4': [ 5, 'IUD',   'IUD'],
                '5': [ 6, 'Injct', 'Injectables'],
                '?': [ 7, 'Diaph', 'Diaphragm'],
                '8': [ 8, 'Cond',  'Condom'],
                '1': [ 9, 'FSter', 'Female sterilization'],
                '2': [10, 'MSter', 'Male sterilization'],
                'R': [11, 'Rhyth', 'Rhythm method'],
                'W': [12, 'Withd', 'Withdrawl'],
                'Y': [13, 'OTrad', 'Other traditional method'],
                '3': [14, 'Impla', 'Implant'],
                '?': [15, 'Abst',  'Abstinence'],
                'L': [16, 'Lact',  'Lactational amenorrhea method (LAM)'],
                '9': [17, 'FCond',  'Female condom'],
                '?': [18, 'Foam',  'Foam and jelly'],
                '7': [19, 'Emerg', 'Emergency contraception'],
                'H': [20, 'SDays', 'Standard days method'],
                'X': [21, 'OModr', 'Other modern method'],

            })
        elif self.which == 'URHI':
            self.mapping = sc.odict({
                ' ': [-1, 'Miss',  'Missing'],
                'N': [ 0, 'Birth', 'Birth'],
                'F': [ 1, 'Mis',   'Miscarriage'],
                'G': [ 2, 'Preg',  'Pregnancy'],
                'M': [ 3, 'Still', 'Stillbirth'],
                'A': [ 4, 'Abort', 'Abortion'],

                '0': [ 5, 'None',  'No method'],
                '1': [ 6, 'FSter', 'Female sterilization'],
                '2': [ 7, 'MSter', 'Male sterilization'],
                '3': [ 8, 'Impla', 'Implant'],
                '4': [ 9, 'IUD',   'IUD'],
                '5': [10, 'Injct', 'Injectables'],
                '6': [11, 'Pill',  'Pill'],
                '7': [12, 'Emerg', 'Emergency contraception'],
                '8': [13, 'MCond', 'Male condom'],
                '9': [14, 'FCond', 'Female condom'],

                'H': [15, 'SDays', 'Standard days method'],
                'L': [16, 'Lact',  'Lactational amenorrhea method (LAM)'],
                'X': [17, 'OModr', 'Other modern method'],

                'R': [18, 'Rhyth', 'Rhythm method'],
                'W': [19, 'Withd', 'Withdrawl'],
                'Y': [20, 'OTrad', 'Other traditional method'],
            })

        self.nmethods = len(self.mapping) - self.skipmissing
        return

    def _parse_keys(self, ind):
        startind = 1 if self.skipmissing else 0
        output = [val[ind] for val in list(self.mapping.values())[startind:]]
        return output

    @property
    def numkeys(self):
        indices = pl.array(self._parse_keys(0))
        return indices

    @property
    def shortkeys(self):
        keys = self._parse_keys(1)
        return keys

    @property
    def longkeys(self):
        keys = self._parse_keys(2)
        return keys

    def keytoind(self, key):
        ind = self.shortkeys.index(key)
        return ind


    def load(self, filename='', rawlines=None, which='DHS6'):
        ''' Load a contraceptive calendar from the original Stata file or one of several processed forms '''
        print(filename)
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
            failed = sc.odict()
            for l,line in enumerate(rawlines):
                sc.percentcomplete(l, len(rawlines))
                cal.append([])
                for char in line:
                    if char in self.mapping:
                        number = self.mapping[char][0]
                        cal[-1].append(number)
                    else:
                        if char not in ['\n']: # Skip space and newline, we know to ignore those
                            if char not in failed:
                                failed[char] = []
                            else:
                                failed[char].append(l)
            if failed:
                raise Exception(f'Failed to parse keys: {failed.keys()}')
            self.cal = pl.array(cal)

        # Load directly from the STATA file
        elif filename.endswith('dta') or filename.endswith('DTA'):
            print('Loading calendar data from Stata...')
            df = pd.read_stata(filename, convert_categoricals=False)
            rawlines = df[self.key].to_list()
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


    def make_results(self, always_log=False, zero_to_nan=False):
        print('Generating results...')
        self.results = sc.objdict()

        # Calculate counts
        counts = pl.zeros((self.nmethods, self.nmethods))
        for pp,person in enumerate(self.cal):
            sc.percentcomplete(pp, len(self.cal))
            for month in range(len(person)-1):
                previous = person[month]
                current = person[month+1]
                if previous != -1 and current != -1: # Transition occurred, data aren't missing
                    counts[current, previous] += 1
        self.results.counts = counts # Store

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
        self.results.rel_props = rel_props # Store
        return self.results

    def _set_axis_labels(self, ax, which=None, offset=0.0, xrotation=90.0, yrotation=0.0):
        if which is None: which = ['x', 'y']
        which = sc.promotetolist(which)
        if 'x' in which:
            ax.set_xticks(self.numkeys+offset)
            ax.set_xticklabels(self.shortkeys, rotation=xrotation)
        if 'y' in which:
            ax.set_yticks(self.numkeys+offset)
            ax.set_yticklabels(self.shortkeys, rotation=yrotation)
        return


    def plot_prop(self, projection='2d', figsize=None):
        ''' Plot all transitions in the contraception calendar '''
        if figsize is None: figsize = (36,16)

        offset = 0.5
        labeloffset = 0.0
        yrotation = 90 if projection == '3d' else 0

        # Create figure and set tick marks on top
        #fig = pl.figure(figsize=figsize)
        fig, ax1 = pl.subplots(1,1,figsize=figsize)
        if projection != '3d':
            pl.rcParams['xtick.top'] = pl.rcParams['xtick.labeltop'] = True
            pl.rcParams['ytick.right'] = pl.rcParams['ytick.labelright'] = True
        else:
            pl.rcParams['xtick.top'] = pl.rcParams['xtick.labeltop'] = False
            pl.rcParams['ytick.right'] = pl.rcParams['ytick.labelright'] = False

        # Plot relative counts
        data = self.results.rel_props
        if projection != '3d':
            #ax1 = fig.add_subplot(122)
            im2 = pl.imshow(data, cmap='jet', vmin=0, vmax=100) # , edgecolors=[0.8]*3
            ca2 = fig.add_axes([0.91, 0.11, 0.03, 0.75])
            fig.colorbar(im2, cax=ca2)
        else:
            ax1 = sc.bar3d(data=data, fig=fig, cmap='jet', axkwargs={'nrows':1, 'ncols':2, 'index':2})
        ax1.set_title('Relative proportion of each transition, diagonal removed (%)', fontweight='bold')

        self._set_axis_labels(ax=ax1, offset=labeloffset, yrotation=yrotation)
        ax1.set_xlim([-offset, self.nmethods-offset])
        ax1.set_ylim([-offset, self.nmethods-offset])

        return fig

    def plot_transitions(self, projection='2d', figsize=None):
        ''' Plot all transitions in the contraception calendar '''
        if figsize is None: figsize = (36,16)

        offset = 0.5
        labeloffset = 0.0
        yrotation = 90 if projection == '3d' else 0

        # Create figure and set tick marks on top
        fig = pl.figure(figsize=figsize)
        if projection != '3d':
            pl.rcParams['xtick.top'] = pl.rcParams['xtick.labeltop'] = True
            pl.rcParams['ytick.right'] = pl.rcParams['ytick.labelright'] = True
        else:
            pl.rcParams['xtick.top'] = pl.rcParams['xtick.labeltop'] = False
            pl.rcParams['ytick.right'] = pl.rcParams['ytick.labelright'] = False

        # Plot total counts
        if projection != '3d':
            data = pl.log10(self.results.counts)
            ax1 = fig.add_subplot(121)
            im1 = pl.imshow(data, cmap=sc.parulacolormap()) # , edgecolors=[0.8]*3
            ca1 = fig.add_axes([0.05, 0.11, 0.03, 0.75])
            fig.colorbar(im1, cax=ca1)
        else:
            data = pl.log10(self.results.counts+1)
            ax1 = sc.bar3d(data=data, fig=fig, axkwargs={'nrows':1, 'ncols':2, 'index':1})
        ax1.set_title('Total number of transitions in calendar (log scale, white=0)', fontweight='bold')

        # Plot relative counts
        data = self.results.rel_props
        if projection != '3d':
            ax2 = fig.add_subplot(122)
            im2 = pl.imshow(data, cmap='jet') # , edgecolors=[0.8]*3
            ca2 = fig.add_axes([0.95, 0.11, 0.03, 0.75])
            fig.colorbar(im2, cax=ca2)
        else:
            ax2 = sc.bar3d(data=data, fig=fig, cmap='jet', axkwargs={'nrows':1, 'ncols':2, 'index':2})
        ax2.set_title('Relative proportion of each transition, diagonal removed (%)', fontweight='bold')

        for ax in [ax1,ax2]:
            self._set_axis_labels(ax=ax, offset=labeloffset, yrotation=yrotation)
            ax.set_xlim([-offset, self.nmethods-offset])
            ax.set_ylim([-offset, self.nmethods-offset])

        return fig


    def plot_slice(self, key, orientation='row', stacking='ontop', figsize=None):
        ''' Plot a single slice through the matrix '''
        if stacking == 'ontop':
            nrows = 2
            ncols = 1
            default_figsize = (20,16)
        else:
            nrows = 1
            ncols = 2
            default_figsize = (30,10)
        if figsize is None: figsize = default_figsize
        fig = pl.figure(figsize=figsize)

        for use_log in [False, True]:
            ax = fig.add_subplot(nrows,ncols,use_log+1)
            if orientation == 'row':
                data = self.results.counts[self.keytoind(key),:]
                preposition = 'from' # For plotting, set the word in the title
            elif orientation == 'col':
                data = self.results.counts[:,self.keytoind(key)]
                preposition = 'to'
            if use_log: data = pl.log10(data)
            pl.bar(self.numkeys, data, edgecolor='none')
            self._set_axis_labels(ax=ax, which='x')
            pl.xlabel('Method', fontweight='bold')
            if use_log: pl.ylabel('log10(Transition count)', fontweight='bold')
            else:       pl.ylabel('Transition count', fontweight='bold')
            pl.title(f'Number of transitions {preposition} method "{key}"', fontweight='bold')
        return fig

