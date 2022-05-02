'''
Specify the core interventions available in FPsim. Other interventions can be
defined by the user by inheriting from these classes.
'''

import numpy as np
import pylab as pl
import sciris as sc
import inspect
from . import utils as fpu


#%% Generic intervention classes

__all__ = ['Intervention', 'Analyzer', 'snapshot', 'timeseries_recorder', 'age_pyramids', 'update_methods']



class Intervention:
    '''
    Base class for interventions. By default, interventions are printed using a
    dict format, which they can be recreated from. To display all the attributes
    of the intervention, use disp() instead.
    
    To retrieve a particular intervention from a sim, use sim.get_intervention().
    
    Args:
        label       (str): a label for the intervention (used for plotting, and for ease of identification)
        show_label (bool): whether or not to include the label in the legend
        do_plot    (bool): whether or not to plot the intervention
        line_args  (dict): arguments passed to pl.axvline() when plotting
    '''
    def __init__(self, label=None, show_label=False, do_plot=None, line_args=None):
        self._store_args() # Store the input arguments so the intervention can be recreated
        if label is None: label = self.__class__.__name__ # Use the class name if no label is supplied
        self.label = label # e.g. "Close schools"
        self.show_label = show_label # Do not show the label by default
        self.do_plot = do_plot if do_plot is not None else True # Plot the intervention, including if None
        self.line_args = sc.mergedicts(dict(linestyle='--', c='#aaa', lw=1.0), line_args) # Do not set alpha by default due to the issue of overlapping interventions
        self.days = [] # The start and end days of the intervention
        self.initialized = False # Whether or not it has been initialized
        self.finalized = False # Whether or not it has been initialized
        return


    def __repr__(self, jsonify=False):
        ''' Return a JSON-friendly output if possible, else revert to short repr '''

        if self.__class__.__name__ in __all__ or jsonify:
            try:
                json = self.to_json()
                which = json['which']
                pars = json['pars']
                parstr = ', '.join([f'{k}={v}' for k,v in pars.items()])
                output = f"cv.{which}({parstr})"
            except Exception as E:
                output = type(self) + f' (error: {str(E)})' # If that fails, print why
            return output
        else:
            return f'{self.__module__}.{self.__class__.__name__}()'


    def disp(self):
        ''' Print a detailed representation of the intervention '''
        return sc.pr(self)


    def _store_args(self):
        ''' Store the user-supplied arguments for later use in to_json '''
        f0 = inspect.currentframe() # This "frame", i.e. Intervention.__init__()
        f1 = inspect.getouterframes(f0) # The list of outer frames
        parent = f1[2].frame # The parent frame, e.g. change_beta.__init__()
        _,_,_,values = inspect.getargvalues(parent) # Get the values of the arguments
        if values:
            self.input_args = {}
            for key,value in values.items():
                if key == 'kwargs': # Store additional kwargs directly
                    for k2,v2 in value.items():
                        self.input_args[k2] = v2 # These are already a dict
                elif key not in ['self', '__class__']: # Everything else, but skip these
                    self.input_args[key] = value
        return


    def initialize(self, sim=None):
        '''
        Initialize intervention -- this is used to make modifications to the intervention
        that can't be done until after the sim is created.
        '''
        self.initialized = True
        self.finalized = False
        return


    def finalize(self, sim=None):
        '''
        Finalize intervention
        
        This method is run once as part of `sim.finalize()` enabling the intervention to perform any
        final operations after the simulation is complete (e.g. rescaling)
        '''
        if self.finalized:
            raise RuntimeError('Intervention already finalized')  # Raise an error because finalizing multiple times has a high probability of producing incorrect results e.g. applying rescale factors twice
        self.finalized = True
        return


    def apply(self, sim):
        '''
        Apply the intervention. This is the core method which each derived intervention
        class must implement. This method gets called at each timestep and can make
        arbitrary changes to the Sim object, as well as storing or modifying the
        state of the intervention.
        
        Args:
            sim: the Sim instance
            
        Returns:
            None
        '''
        raise NotImplementedError


    def plot_intervention(self, sim, ax=None, **kwargs):
        '''
        Plot the intervention
        
        This can be used to do things like add vertical lines on days when
        interventions take place. Can be disabled by setting self.do_plot=False.
        
        Note 1: you can modify the plotting style via the ``line_args`` argument when
        creating the intervention.
        
        Note 2: By default, the intervention is plotted at the days stored in self.days.
        However, if there is a self.plot_days attribute, this will be used instead.
        
        Args:
            sim: the Sim instance
            ax: the axis instance
            kwargs: passed to ax.axvline()
            
        Returns:
            None
        '''
        line_args = sc.mergedicts(self.line_args, kwargs)
        if self.do_plot or self.do_plot is None:
            if ax is None:
                ax = pl.gca()
            if hasattr(self, 'plot_days'):
                days = self.plot_days
            else:
                days = self.days
            if sc.isiterable(days):
                label_shown = False # Don't show the label more than once
                for day in days:
                    if sc.isnumber(day):
                        if self.show_label and not label_shown: # Choose whether to include the label in the legend
                            label = self.label
                            label_shown = True
                        else:
                            label = None
                        ax.axvline(day, label=label, **line_args)
        return


    def to_json(self):
        '''
        Return JSON-compatible representation
        
        Custom classes can't be directly represented in JSON. This method is a
        one-way export to produce a JSON-compatible representation of the
        intervention. In the first instance, the object dict will be returned.
        However, if an intervention itself contains non-standard variables as
        attributes, then its `to_json` method will need to handle those.
        
        Note that simply printing an intervention will usually return a representation
        that can be used to recreate it.
        
        Returns:
            JSON-serializable representation (typically a dict, but could be anything else)
        '''
        which = self.__class__.__name__
        pars = sc.jsonify(self.input_args)
        output = dict(which=which, pars=pars)
        return output



class Analyzer(sc.prettyobj):
    '''
    Base class for analyzers. Based on the Intervention class. Analyzers are used
    to provide more detailed information about a simulation than is available by
    default -- for example, pulling states out of sim.people on a particular timestep
    before it gets updated in the next timestep.
    
    To retrieve a particular analyzer from a sim, use sim.get_analyzer().
    
    Args:
        label (str): a label for the Analyzer (used for ease of identification)
    '''

    def __init__(self, label=None):
        if label is None:
            label = self.__class__.__name__ # Use the class name if no label is supplied
        self.label = label # e.g. "Record ages"
        self.initialized = False
        self.finalized = False
        return


    def initialize(self, sim=None):
        '''
        Initialize the analyzer, e.g. convert date strings to integers.
        '''
        self.initialized = True
        self.finalized = False
        return


    def finalize(self, sim=None):
        '''
        Finalize analyzer
        
        This method is run once as part of `sim.finalize()` enabling the analyzer to perform any
        final operations after the simulation is complete (e.g. rescaling)
        '''
        if self.finalized:
            raise RuntimeError('Analyzer already finalized')  # Raise an error because finalizing multiple times has a high probability of producing incorrect results e.g. applying rescale factors twice
        self.finalized = True
        return


    def apply(self, sim):
        '''
        Apply analyzer at each time point. The analyzer has full access to the
        sim object, and typically stores data/results in itself. This is the core
        method which each analyzer object needs to implement.
        
        Args:
            sim: the Sim instance
        '''
        raise NotImplementedError


    def to_json(self):
        '''
        Return JSON-compatible representation
        
        Custom classes can't be directly represented in JSON. This method is a
        one-way export to produce a JSON-compatible representation of the
        intervention. This method will attempt to JSONify each attribute of the
        intervention, skipping any that fail.
        
        Returns:
            JSON-serializable representation
        '''
        # Set the name
        json = {}
        json['analyzer_name'] = self.label if hasattr(self, 'label') else None
        json['analyzer_class'] = self.__class__.__name__

        # Loop over the attributes and try to process
        attrs = self.__dict__.keys()
        for attr in attrs:
            try:
                data = getattr(self, attr)
                try:
                    attjson = sc.jsonify(data)
                    json[attr] = attjson
                except Exception as E:
                    json[attr] = f'Could not jsonify "{attr}" ({type(data)}): "{str(E)}"'
            except Exception as E2:
                json[attr] = f'Could not jsonify "{attr}": "{str(E2)}"'
        return json


class snapshot(Analyzer):
    '''
    Analyzer that takes a "snapshot" of the sim.people array at specified points
    in time, and saves them to itself.
    
    Args:
        timesteps (list): list of timesteps on which to take the snapshot
        args   (list): additional timestep(s)
        die    (bool): whether or not to raise an exception if a date is not found (default true)
        kwargs (dict): passed to Analyzer()
        
        
    **Example**::
    
        sim = cv.Sim(analyzers=fps.snapshot('2020-04-04', '2020-04-14'))
        sim.run()
        snapshot = sim.pars['analyzers'][0]
        people = snapshot.snapshots[0]
    '''

    def __init__(self, timesteps, *args, die=True, **kwargs):
        super().__init__(**kwargs) # Initialize the Analyzer object
        timesteps = sc.promotetolist(timesteps) # Combine multiple days
        timesteps.extend(args) # Include additional arguments, if present
        self.die       = die  # Whether or not to raise an exception
        self.timesteps = timesteps # String representations
        self.snapshots = sc.odict() # Store the actual snapshots
        return


    def apply(self, sim):
        """
        Apply snapshot at each timestep listed in timesteps and
        save result at snapshot[str(timestep)]
        """
        if sim.i in self.timesteps:
            self.snapshots[str(sim.i)] = sc.dcp(sim.people) # Take snapshot!


class timeseries_recorder(Analyzer):
    '''
    Record every attribute in people as a timeseries.

    Attributes:

        self.i: The list of timesteps (ie, 0 to 261 steps).
        self.t: The time elapsed in years given how many timesteps have passed (ie, 25.75 years).
        self.y: The calendar year of timestep (ie, 1975.75).
        self.keys: A list of people states excluding 'dobs'.
        self.data: A dictionary where self.data[state][timestep] is the mean of the state at that timestep.
    '''

    def __init__(self):
        """
        Initializes self.i/t/y as empty lists and self.data as empty dictionary
        """
        super().__init__()
        self.i = []
        self.t = []
        self.y = []
        self.data = sc.objdict(defaultdict=list)
        return

    def initialize(self, sim):
        """
        Initializes self.keys from sim.people
        """
        super().initialize()
        self.keys = sim.people.keys()
        self.keys.remove('dobs')
        return


    def apply(self, sim):
        """
        Applies recorder at each timestep
        """
        self.i.append(sim.i)
        self.t.append(sim.t)
        self.y.append(sim.y)
        for k in self.keys:
            val = np.mean(sim.people[k])
            self.data[k].append(val)


    def plot(self, x='y', fig_args=None, pl_args=None):
        """
        Plots time series of each state as a line graph
        """

        xmap = dict(i=self.i, t=self.t, y=self.y)
        x = xmap[x]

        fig_args  = sc.mergedicts(fig_args)
        pl_args = sc.mergedicts(pl_args)
        nkeys = len(self.keys)
        rows,cols = sc.get_rows_cols(nkeys)

        fig = pl.figure(**fig_args)

        for k,key in enumerate(self.keys):
            pl.subplot(rows,cols,k+1)
            try:
                data = np.array(self.data[key], dtype=float)
                mean = data.mean()
                label = f'mean: {mean}'
                pl.plot(x, data, label=label, **pl_args)
                pl.title(key)
                pl.legend()
            except:
                pl.title(f'Could not plot {key}')

        return fig


class age_pyramids(Analyzer):
    '''
    Records age pyramids for each timestep.

    Attributes:

        self.bins: A list of ages, default is a sequence from 0 to max_age + 1.
        self.data: A matrix of shape (number of timesteps, number of bins - 1) containing age pyramid data.
    '''

    def __init__(self, bins=None):
        """
        Initializes bins and data variables
        """
        super().__init__()
        self.bins = bins
        self.data = None
        return

    def initialize(self, sim):
        """
        Initializes bins and data with proper shapes
        """
        super().initialize()
        if self.bins is None:
            self.bins = np.arange(0, sim.pars['max_age']+2)
            nbins = len(self.bins)-1
        self.data = np.full((sim.npts, nbins), np.nan)
        self._raw = sc.dcp(self.data)
        return

    def apply(self, sim):
        """
        Records histogram of ages of all alive individuals at a timestep such that
        self.data[timestep] = list of proportions where index signifies age
        """
        ages = sim.people.age[sc.findinds(sim.people.alive)]
        self._raw[sim.i, :] = np.histogram(ages, self.bins)[0]
        self.data[sim.i, :] = self._raw[sim.i, :]/self._raw[sim.i, :].sum()

    def plot(self):
        """
        Plots self.data as 2D pyramid plot
        """
        fig = pl.figure()
        pl.pcolormesh(self.data.T)
        pl.xlabel('Timestep')
        pl.ylabel('Age (years)')
        return fig

    def plot3d(self):
        """
        Plots self.data as 3D pyramid plot
        """
        print('Warning, very slow...')
        fig = pl.figure()
        sc.bar3d(self.data.T)
        pl.xlabel('Timestep')
        pl.ylabel('Age (years)')
        return fig



def key2ind(sim, key):
    """
    Take a method key and convert to an int, e.g. 'Condoms' → 7
    """
    ind = key
    if ind in [None, 'all']:
        ind = slice(None) # This is equivalent to ":" in matrix[:,:]
    elif isinstance(ind, str):
        ind = sim.pars['methods']['map'][key]
    return ind


def getval(v):
    ''' Handle different ways of supplying a value -- number, distribution, function '''
    if sc.isnumber(v):
        return v
    elif isinstance(v, dict):
        return fpu.sample(**v)[0]
    elif callable(v):
        return v()


class update_methods(Intervention):
    """
    Intervention to modify method efficacy and/or switching matrix.

    Attributes:
        self.year::float: The year we want to change the method.
        self.scen::dict: Has the following keys:

            probs::str
                An optional key with the value of a list of dictionaries where each dictionary has 
                the following keys:

                source::str
                    The source method to be changed.
                dest::str
                    The destination method to be changed.
                factor::float
                    The factor by which to multiply existing probability.
                value::float
                    The value to replace the switching probability value.
                keys::list
                    A list of strings representing age groups to affect.

            eff::str
                An optional key for changing efficacy; its value is a dictionary with the following schema:

                    {method: efficacy}
                        Where method is the method to be changed, and efficacy is the new efficacy (can include multiple keys). 

        self.matrix::str: One of ['probs_matrix', 'probs_matrix_1', 'probs_matrix_1-6'] where:

            probs_matrix:
                Changes the specified uptake at the corresponding year regardless of state.
            probs_matrix_1
                Changes the specified uptake for all individuals in their first month postpartum.
            probs_matrix_1-6
                Changes the specified uptake for all individuals that are in the first 6 months postpartum.
    """

    def __init__(self, year, scen, matrix='probs_matrix'):
        """
        Initializes self.year/scen/matrix from parameters
        """
        super().__init__()
        self.year   = year
        self.scen   = scen
        self.matrix = matrix
        valid_matrices = ['probs_matrix', 'probs_matrix_1', 'probs_matrix_1-6'] # TODO: be less subtle about the difference between normal and postpartum matrices
        if matrix not in valid_matrices:
            raise sc.KeyNotFoundError(f'Matrix must be one of {valid_matrices}, not "{matrix}"')
        return


    def apply(self, sim, verbose=True):
        """
        Applies the efficacy or contraceptive uptake changes if it is the specified year
        based on scenario specifications.
        """

        if sim.y >= self.year and not(hasattr(sim, 'modified')):
            sim.modified = True

            # Implement efficacy
            if 'eff' in self.scen:
                for k,rawval in self.scen.eff.items():
                    v = getval(rawval)
                    ind = key2ind(sim, k)
                    orig = sim.pars['method_efficacy'][ind]
                    sim.pars['method_efficacy'][ind] = v
                    if verbose:
                        print(f'At time {sim.y:0.1f}, efficacy for method {k} was changed from {orig:0.3f} to {v:0.3f}')

            # Implement method mix shift
            if 'probs' in self.scen:
                for entry in self.scen.probs:
                    source = key2ind(sim, entry['source'])
                    dest   = key2ind(sim, entry['dest'])
                    factor = entry.pop('factor', None)
                    value  = entry.pop('value', None)
                    keys   = entry.pop('keys', None)
                    if keys is None:
                        keys = sim.pars['methods']['probs_matrix'].keys()

                    for k in keys:
                        if self.matrix == 'probs_matrix':
                            matrices = sim.pars['methods']
                        else:
                            matrices = sim.pars['methods_postpartum']
                        matrix = matrices[self.matrix][k]
                        orig = matrix[source, dest]
                        if factor is not None:
                            matrix[source, dest] *= getval(factor)
                        elif value is not None:
                            matrix[source, dest] = getval(value)
                        if verbose:
                            print(f'At time {sim.y:0.1f}, matrix for age group {k} was changed from:\n{orig}\nto\n{matrix[source, dest]}')


        return
