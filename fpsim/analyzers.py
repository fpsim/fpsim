'''
Specify the core analyzers available in FPsim. Other analyzers can be
defined by the user by inheriting from these classes.
'''

import os
import numpy as np
import pandas as pd
import sciris as sc
import pylab as pl
from . import defaults as fpd


#%% Generic intervention classes

__all__ = ['Analyzer', 'snapshot', 'timeseries_recorder', 'age_pyramids', 'verbose_sim']


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
        pass


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

        sim = fp.Sim(analyzers=fps.snapshot('2020-04-04', '2020-04-14'))
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
        for t in self.timesteps:
            if np.isclose(sim.i, t):
                self.snapshots[str(sim.i)] = sc.dcp(sim.people) # Take snapshot!
        return


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
        self.data = sc.objdict()
        return

    def initialize(self, sim):
        """
        Initializes self.keys from sim.people
        """
        super().initialize()
        self.keys = sim.people.keys()
        self.keys.remove('dobs')
        for key in self.keys:
            self.data[key] = []
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


class verbose_sim(Analyzer):
    def __init__(self, to_csv=False, custom_csv_tables=None, to_file=False):
        """
        Initializes a verbose_sim analyzer which extends the logging functionality of the sim with calculated channels,
        total state results of a sim run, the story() feature, and configurable file formatting for results
        """

        self.to_csv = to_csv
        self.custom_csv_tables = custom_csv_tables
        self.to_file = to_file
        self.initialized = False

        self.total_results = sc.ddict(lambda: {})

        self.dead_moms = set()
        self.is_sexactive = set()
        self.events = sc.ddict(dict)
        self.channels = ["Births", "Conceptions", "Miscarriages", "Deaths"]
        self.set_baseline = False
        self.states = list(fpd.person_defaults.keys()) + ['dobs'] # states saved by timestep

    def apply(self, sim):
        """
        Logs data for total_results and events at each timestep.

        Output:
            self.total_results::dict
                Dictionary of all individual results formatted as {timestep: attribute: [values]}
                keys correspond to fpsim.defaults debug_states
            self.events::dict
                Dictionary of events correponding to self.channels formatted as {timestep: channel: [indices]}.
        """
        if not self.set_baseline:
            initial_pop = sim.pars['n_agents']
            self.last_year_births = [0] * initial_pop
            self.last_year_gestations = [0] * initial_pop
            self.last_year_alive = [0] * initial_pop
            self.last_year_pregnant = [0] * initial_pop
            self.set_baseline = True

        for state in self.states:
            self.total_results[sim.y][state] = sc.dcp(getattr(sim.people, state))

        # Getting births gestation and sexual_activity
        self.this_year_births = sc.dcp(self.total_results[sim.y]["parity"])
        self.this_year_gestations = sc.dcp(self.total_results[sim.y]["gestation"])
        self.this_year_alive = sc.dcp(self.total_results[sim.y]["alive"])
        self.this_year_pregnant = sc.dcp(self.total_results[sim.y]["pregnant"])

        for channel in self.channels:
            self.events[sim.y][channel] = []

        # Comparing parity of previous year to this year, adding births
        for index, last_parity in enumerate(self.last_year_births):
            if last_parity < self.this_year_births[index]:
                for i in range(self.this_year_births[index] - last_parity):
                    self.events[sim.y]['Births'].append(index)

        # Comparing pregnancy of previous year to get conceptions
        for index, last_pregnant in enumerate(self.last_year_pregnant):
            if last_pregnant == 0 and self.this_year_pregnant[index]:
                self.events[sim.y]['Conceptions'].append(index)

        # Comparing gestaton of previous year to get miscarriages
        for index, last_gestation in enumerate(self.last_year_gestations):
            # This is when miscarriages are checked in Sim
            if last_gestation == (sim.pars['end_first_tri'] - 1) and self.this_year_gestations[index] == 0:
                self.events[sim.y]['Miscarriages'].append(index)

        for index, alive in enumerate(self.last_year_alive):
            if alive > self.this_year_alive[index]:
                self.events[sim.y]['Deaths'].append(index)

        # Aggregate channels taken from people.results
        self.last_year_births = sc.dcp(self.this_year_births)
        self.last_year_gestations = sc.dcp(self.this_year_gestations)
        self.last_year_alive = sc.dcp(self.this_year_alive)
        self.last_year_pregnant = sc.dcp(self.this_year_pregnant)

    def save(self, to_csv=True, to_json=False, custom_csv_tables=None):
        """
        At the end of sim run, stores total_results as either a json or feather file.

        Inputs
            self.to_csv::bool
                If True, writes results to csv files in /sim_output where each state's history is a separate file
            self.to_json::bool
                If True, writes results to json file
            custom_csv_tables::list
                List of states that the user wants to write to csv, default is all
        Outputs:
            Either a json file at "sim_output/total_results.json"
            or a csv file for each state at "sim_output/{state}_state.csv"
        """
        os.makedirs("sim_output", exist_ok=True)
        if to_json:
            sc.savejson(filename="sim_output/total_results.json", obj=self.total_results)

        if to_csv:
            states = self.states if self.custom_csv_tables is None else custom_csv_tables
            for state in states:
                state_frame = pd.DataFrame()
                max_length = len(self.total_results[max(self.total_results.keys())][state])
                for timestep, _ in self.total_results.items():
                    colname = str(timestep) + "_" + state
                    adjustment = max_length - len(self.total_results[timestep][state])
                    state_frame[colname] = list(self.total_results[timestep][state]) + [None] * adjustment # ONLY WORKS IF LAST YEAR HAS MOST PEOPLE

                state_frame.to_csv(f"sim_output/{state}_state.csv")

    def story(self, index, output=False, debug=False):
        """
        Prints a story of all major events in an individual's life based on calculated verbose_sim channels,
        base Sim channels, and statistics calculated within the function such as year of birth of individual.

        Args:
            index (int): index of the individual, must be less than population
            output (bool): return as output string rather than print
            debug (bool): print additional information

        Outputs:
            printed display of each major event in the individual's life
        """
        string = ''

        if debug:
            print(self.events.keys())

        def to_date(t):
            year = int(t)
            if debug:
                print(t)
            mo = round(((t) - year) * 12)
            month = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][mo]
            return f'{year}-{month}'

        if len(self.events) == 0:
            errormsg = 'Story function can only be used after sim is run. Try Experiment.run_model() first'
            raise RuntimeError(errormsg)

        last_year = max(self.total_results.keys())
        ages = self.total_results[last_year]['age'] # Progresses even if dead
        year_born = last_year - ages[index]
        if debug:
            print(last_year)
            print(year_born)
            print(ages[index])
        string += f'This is the story of Person {index} who was born {to_date(year_born)}:\n'

        event_response_dict = {
            "Births": "gives birth",
            "Conceptions": "conceives",
            "Miscarriages": "has a miscarriage",
            "Deaths": "dies"
        }
        method_list = list(fpd.method_map.keys())
        last_method = method_list[self.total_results[min(self.total_results.keys())]['method'][index]]
        for y in self.events:
            if y >= year_born:
                for new_channel in event_response_dict:
                    if index in self.events[y][new_channel]:
                        if new_channel == "Births":
                            string += f"{to_date(y)}: Person {index} gives birth to child number {self.total_results[y]['parity'][index]}\n"
                        else:
                            string += f"{to_date(y)}: Person {index} {event_response_dict[new_channel]}\n"
                    if self.total_results[y]['sexual_debut_age'][index] == 0:
                        string += f"{to_date(y)}: Person {index} had their sexual debut\n"
            new_method = method_list[self.total_results[y]['method'][index]]
            if new_method != last_method:
                string += f"{to_date(y)}: Person {index} switched from {last_method} to {new_method}\n"
            last_method = new_method

        if not output:
            print(string)
        else:
            return string
