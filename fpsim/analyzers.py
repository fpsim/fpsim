'''
Specify the core analyzers available in FPsim. Other analyzers can be
defined by the user by inheriting from these classes.
'''

import numpy as np
import sciris as sc
import pylab as pl
from . import defaults as fpd
from . import utils as fpu
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker


#%% Generic intervention classes
__all__ = ['Analyzer', 'snapshot', 'cpr_by_age', 'method_mix_by_age', 'age_pyramids', 'empowerment_recorder', 'education_recorder', 'lifeof_recorder']

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
            if np.isclose(sim.ti, t):
                self.snapshots[str(sim.ti)] = sc.dcp(sim.people) # Take snapshot!
        return


class cpr_by_age(Analyzer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)   # Initialize the Analyzer object
        self.age_bins = [v[1] for v in fpd.method_age_map.values()]
        self.results = None
        return

    def initialize(self, sim):
        super().initialize()
        self.results = {k: np.zeros(sim.npts) for k in fpd.method_age_map.keys()}
        self.total = np.zeros(sim.npts)

    def apply(self, sim):
        ppl = sim.people
        for key, (age_low, age_high) in fpd.method_age_map.items():
            match_low_high = fpu.match_ages(ppl.age, age_low, age_high)
            denom_conds = match_low_high * (ppl.sex == 0) * ppl.alive
            num_conds = denom_conds * (ppl.method != 0)
            self.results[key][sim.ti] = sc.safedivide(np.count_nonzero(num_conds), np.count_nonzero(denom_conds))

        total_denom_conds = (ppl.sex == 0) * ppl.alive
        total_num_conds = total_denom_conds * (ppl.method != 0)
        self.total[sim.ti] = sc.safedivide(np.count_nonzero(total_num_conds), np.count_nonzero(total_denom_conds))
        return


class method_mix_by_age(Analyzer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)   # Initialize the Analyzer object
        self.age_bins = [v[1] for v in fpd.method_age_map.values()]
        self.results = None
        self.n_methods = None
        return

    def initialize(self, sim):
        super().initialize()

    def finalize(self, sim):
        n_methods = len(sim.contraception_module.methods)
        self.results = {k: np.zeros(n_methods) for k in fpd.method_age_map.keys()}
        ppl = sim.people
        for key, (age_low, age_high) in fpd.method_age_map.items():
            match_low_high = fpu.match_ages(ppl.age, age_low, age_high)
            denom_conds = match_low_high * (ppl.sex == 0) * ppl.alive
            for mn in range(n_methods):
                num_conds = denom_conds * (ppl.method == mn)
                self.results[key][mn] = sc.safedivide(np.count_nonzero(num_conds), np.count_nonzero(denom_conds))
        return

class education_recorder(Analyzer):
        '''
        Analyzer records all education attributes of females + pregnancy + living status
        for all timesteps. Made for debugging purposes.

        Args:
            args   (list): additional timestep(s)
            kwargs (dict): passed to Analyzer()
        '''

        def __init__(self, **kwargs):
            super().__init__(**kwargs)   # Initialize the Analyzer object
            self.snapshots = sc.odict()  # Store the actual snapshots
            self.keys = ['edu_objective', 'edu_attainment', 'edu_completed',
                         'edu_dropout', 'edu_interrupted',
                         'pregnant', 'alive', 'age']
            self.max_agents = 0     # maximum number of agents this analyzer tracks
            self.time = []
            self.trajectories = {}  # Store education trajectories
            return

        def apply(self, sim):
            """
            Apply snapshot at each timestep listed in timesteps and
            save result at snapshot[str(timestep)]
            """
            females = sim.people.filter(sim.people.is_female)
            self.snapshots[str(sim.ti)] = {}
            for key in self.keys:
                self.snapshots[str(sim.ti)][key] = sc.dcp(females[key])  # Take snapshot!
                self.max_agents = max(self.max_agents, len(females))
            return

        def finalize(self, sim=None):
            """
             Process data in snapshots so we can plot it easily
            """
            if self.finalized:
                raise RuntimeError('Analyzer already finalized')
            self.finalized = True
            # Process data so we can plot it easily
            self.time = np.array([key for key in self.snapshots.keys()], dtype=int)
            for state in self.keys:
                self.trajectories[state] = np.full((len(self.time), self.max_agents), np.nan)
                for ti, t in enumerate(self.time):
                    stop_idx = len(self.snapshots[t][state])
                    self.trajectories[state][ti, 0:stop_idx] = self.snapshots[t][state]
            return

        def plot(self, index=0, fig_args=None, pl_args=None):
            """
            Plots time series of each state as a line graph
            Args:
               index: index of the female individual, must be less the analyzer's max_pop_size
            """
            fig_args = sc.mergedicts(fig_args, {'figsize': (5, 7)})
            pl_args = sc.mergedicts(pl_args)
            rows, cols = sc.get_rows_cols(2)

            fig = pl.figure(**fig_args)
            keys2 = ['edu_completed', 'edu_interrupted', 'edu_dropout']
            keys3 = ['pregnant', 'alive']

            k = 0
            pl.subplot(rows, cols, k + 1)
            age_data = self.trajectories["age"]
            state = "edu_attainment"
            data = self.trajectories[state]
            pl.step(self.time, data[:, index], color="black", label=f"{state}", where='mid', **pl_args)
            state = "edu_objective"
            data = self.trajectories[state]
            pl.step(self.time, data[:, index], color="red", ls="--", label=f"{state}", where='mid', **pl_args)
            pl.ylim([0, 24])
            pl.title('Education')
            pl.ylabel('Education (years)')
            pl.xlabel('Timesteps')
            pl.legend()

            k += 1
            for state in sc.mergelists(keys2, keys3):
                pl.subplot(rows, cols, k + 1)
                data = self.trajectories[state]
                if state in keys2:
                    if state  == 'edu_interrupted':
                        pl.step(self.time, 3*data[:, index], color=[0.7, 0.7, 0.7], label=f"{state}", ls=":", where='mid', **pl_args)
                    elif state == "edu_dropout":
                        pl.step(self.time, 3*data[:, index], color="black", label=f"{state}", ls=":", where='mid', **pl_args)
                    else:
                        pl.step(self.time, 3*data[:, index], color="#2ca25f", label=f"{state}", where='mid', **pl_args)
                elif state  == 'pregnant':
                    pl.step(self.time, data[:, index], color="#dd1c77", label=f"{state}", where='mid', **pl_args)
                elif state == 'alive':
                    plt.step(self.time, 4*data[:, index],  color="black", ls="--", label=f"{state}", where='mid', **pl_args)
                pl.title(f"Education trajectories - Start age: {int(age_data[0, index])}; final age {int(age_data[-1, index])}.")
                pl.ylabel('State')
                pl.xlabel('Timesteps')
                pl.legend()
            return fig

        def plot_waterfall(self, max_timepoints=30, min_age=18, max_age=40, fig_args=None, pl_args=None):
            """
            Plot a waterfall plot showing the evolution of education objective and attainment over time
            for a specified age group.

            Args:
                max_timepoints (int, optional): The maximum number of timepoints to plot, defaults to 30.
                min_age (int, optional): The minimum age for the age group, defaults to 18.
                max_age (int, optional): The maximum age for the age group, defaults to 20.

            Returns:
                figure handle

            The function generates uses kernel density estimation to visualize the data. If there's not data for the
            min max age specified, for a specific time step (ie, there are no agents in that age group), it adds a
            textbox. This is an edge case that can happen for a simulation with very few agents, and a very narrow
            age group.
            """

            from scipy.stats import gaussian_kde

            data_att = self.trajectories["edu_attainment"]
            data_obj = self.trajectories["edu_objective"]
            data_age = self.trajectories["age"]

            mask = (data_age < min_age) | (data_age > max_age) | np.isnan(data_age)

            data_att = np.ma.array(data_att, mask=mask)
            data_obj = np.ma.array(data_obj, mask=mask)

            n_tpts = data_att.shape[0]
            if n_tpts <= max_timepoints:
                tpts_to_plot = np.arange(n_tpts)
            else:
                tpts_to_plot = np.linspace(0, n_tpts - 1, max_timepoints, dtype=int)

            fig_args = sc.mergedicts(fig_args, {'figsize': (3, 10)})
            pl_args = sc.mergedicts(pl_args, {'y_scaling': 0.9})

            fig = plt.figure(**fig_args)
            ax = fig.add_subplot(111)

            edu_min, edu_max = 0, 25
            edu_mid = (edu_max-edu_min)/2 + edu_min
            edu_years = np.linspace(edu_min, edu_max, 50)
            y_scaling = pl_args['y_scaling']

            # Set the y-axis (time) labels
            ax.set_yticks(y_scaling*np.arange(len(tpts_to_plot)))
            ax.set_yticklabels(tpts_to_plot)

            # Initialize legend labels
            edu_att_label = None
            edu_obj_label = None

            # Loop through the selected time points and create kernel density estimates
            for idx, ti in enumerate(tpts_to_plot):
                data_att_ti = np.sort(data_att[ti, :][~data_att[ti, :].mask].data)
                data_obj_ti = np.sort(data_obj[ti, :][~data_obj[ti, :].mask].data)

                try:
                    kde_att = gaussian_kde(data_att_ti)
                    kde_obj = gaussian_kde(data_obj_ti)

                    y_att = kde_att(edu_years)
                    y_obj = kde_obj(edu_years)

                    if idx == len(tpts_to_plot) - 1:
                        edu_obj_label = 'Distribution of education objectives'
                        edu_att_label = 'Current distribution of education attainment'

                    ax.fill_between(edu_years, y_scaling*idx, y_obj / y_obj.max() + y_scaling*idx,
                                    color='#2f72de', alpha=0.3, label=edu_obj_label)
                    ax.plot(edu_years, y_att / y_att.max() + y_scaling*idx,
                            color='black', alpha=0.7, label=edu_att_label)
                except:
                    # No data available for this age group or age range,
                    ax.plot(edu_years,  (y_scaling * idx) * np.ones_like(edu_years),
                            color='black', alpha=0.2, label=edu_att_label)
                    ax.annotate('No data available ', xy=(edu_mid, y_scaling*idx), xycoords='data', fontsize=8,
                                ha='center', va='center', bbox=dict(boxstyle='round,pad=0.4', fc='none', ec="none"))

            # Labels and annotations
            ax.set_xlim([edu_min, edu_max])
            ax.set_xlabel('Education years')
            ax.set_ylabel('Timesteps')
            ax.legend()
            ax.set_title(f"Evolution of education \n objective and attainment for age group:\n{min_age}-{max_age}.")

            # Show the plot
            plt.show()
            return fig


class empowerment_recorder(Analyzer):
    '''
    Records timeseries of empowerment attributes for different age groups.
     - For boolean attributes it computes the percentage returns percentage.
     - For float attributes it computes the median of the attribute from the population of interes

    Attributes:
        self.bins: A list of ages, default is a sequence from 0 to max_age + 1.
        self.keys: A list of people's empowerment attributes.
        self.data: A dictionary where self.data[attribute] is a a matrix of shape (number of timesteps, number of bins - 1) containing age pyramid data.

    '''

    def __init__(self, bins=None):
        """
        Initializes self.ti/t/y as empty lists and self.data as empty dictionary
        """
        super().__init__()
        self.bins = bins
        self.data = sc.objdict()
        self.keys = ['age', 'paid_employment', 'decision_wages', 'decision_health', 'decision_purchase',
                     'buy_decision_major', 'buy_decision_daily', 'buy_decision_clothes',
                     'decide_spending_partner', 'has_savings', 'has_fin_knowl', 'has_fin_goals',
                     'sexual_autonomy', 'financial_autonomy', 'decision_making']
        self.nbins = None
        return

    def initialize(self, sim):
        """
        Initializes self.keys from sim.people
        """
        super().initialize()
        if self.bins is None:
            self.bins = np.arange(0, sim.pars['max_age']+2)
        self.nbins = len(self.bins)-1

        for key in self.keys:
            self.data[key] = np.full((self.nbins, sim.npts), np.nan)
        return

    def apply(self, sim):
        """
        Records histogram of empowerment attribute of all **alive female** individuals
        """
        # Alive and female
        living_females = sc.findinds(sim.people.alive, sim.people.is_female)
        ages = sim.people.age[living_females]
        age_group = np.digitize(ages, self.bins) - 1

        for key in self.keys:
            data = sim.people[key][living_females]
            if key == 'age':
                # Count how many living females we have in this age group
                temp = np.histogram(ages, self.bins)[0]
                vals = temp / temp.sum()  # Transform to density
            elif key in ['financial_autonomy', 'decision_making']:
                vals = [np.mean(data[age_group == group_idx]) for group_idx in range(1, len(self.bins))]
            else: # boolean states, expressed as proportion
                vals = [np.mean(data[age_group == group_idx]) for group_idx in range(1, len(self.bins))]
            self.data[key][:, sim.ti] = vals

    def plot(self, to_plot=None, fig_args=None, pl_args=None):
        """
        Plot all keys in self.keys or in to_plot as a heatmaps
        """
        fig_args = sc.mergedicts(fig_args)
        pl_args = sc.mergedicts(pl_args)
        fig = pl.figure(**fig_args)

        if to_plot is None:
            to_plot = self.keys

        nkeys = len(to_plot)
        rows, cols = sc.get_rows_cols(nkeys)

        axs = []
        for k, key in enumerate(to_plot):
            axs.append(fig.add_subplot(rows, cols, k+1))
            try:
                data = np.array(self.data[key], dtype=float)
                label = f'metric: {key}'
                if key in ['age']:
                    clabel = "proportion of agents"
                    cmap = 'Blues'
                    vmin, vmax = 0, np.nanmax(data[:])
                # Composite measures
                elif key in ['financial_autonomy', 'decision_making']:
                    clabel = f"Median of composite:\n{key}"
                    cmap = 'RdPu'
                    vmin, vmax = 0, 4.0
                else:
                    clabel = f"proportion of {key}"
                    cmap = 'coolwarm'
                    vmin, vmax = 0, 1

                pcm = axs[k].pcolormesh(data, label=label, cmap=cmap, vmin=vmin, vmax=vmax, **pl_args)

                # Add colorbar to the right of the subplot
                divider = make_axes_locatable(axs[k])
                cax = divider.append_axes("right", size="2.5%", pad=0.05)

                # Add colorbar to the right of the subplot
                plt.colorbar(pcm, cax=cax, label=clabel)

                # Generate age group labels and tick positions
                ytick_labels = [f"{self.bins[i]:.0f}-{self.bins[i+1]-1:.0f}" for i in range(self.nbins)]
                ytick_positions = np.arange(0.5, self.nbins + 0.5)  # Center positions for ticks

                # Reduce the number of labels if we have too many bins
                max_labels = 10
                if len(ytick_labels) > max_labels:
                    step_size = len(ytick_labels) // max_labels
                    ytick_labels = ytick_labels[::step_size]
                    ytick_positions = ytick_positions[::step_size]

                # Label plots
                axs[k].set_yticks(ytick_positions)
                axs[k].set_yticklabels(ytick_labels)
                axs[k].set_title(key)
                axs[k].set_xlabel('Timestep')
                axs[k].set_ylabel('Age (years)')
            except:
                pl.title(f'Could not plot {key}')

        return fig


class lifeof_recorder(Analyzer):
    '''
    Analyzer records sexual and reproductive history, and contraceptions
    females, plus age and living status for all timesteps.
    Made for debugging purposes.

    Args:
        args   (list): additional timestep(s)
        kwargs (dict): passed to Analyzer()
    '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # Initialize the Analyzer object
        self.snapshots = sc.odict()  # Store the actual snapshots
        self.keys = ['method', 'pregnant', 'lam', 'postpartum', 'sexually_active',
                     'abortion', 'stillbirth', 'parity', 'method',
                     'miscarriage', 'age', 'on_contra', 'alive']
        self.max_agents = 0  # maximum number of agents this analyzer tracks
        self.time = []
        self.trajectories = {}  # Store education trajectories
        from .methods import Methods
        self.method_map = {idx: method.label for idx, method in enumerate(Methods.values())}
        self.m2y = 1.0/fpd.mpy  # Transform timesteps in months to years

        return

    def apply(self, sim):
        """
        Apply snapshot at each timestep listed in timesteps and
        save result at snapshot[str(timestep)]
        """
        females = sim.people.filter(sim.people.is_female)
        self.snapshots[str(sim.ti)] = {}
        for key in self.keys:
            self.snapshots[str(sim.ti)][key] = sc.dcp(
                females[key])  # Take snapshot!
            self.max_agents = max(self.max_agents, len(females))
        return

    def finalize(self, sim=None):
        """
         Process data in snapshots so we can plot it easily
        """
        if self.finalized:
            raise RuntimeError('Analyzer already finalized')
        self.finalized = True
        # Process data so we can plot it easily
        self.time = np.array([key for key in self.snapshots.keys()],
                             dtype=int)
        for state in self.keys:
            self.trajectories[state] = np.full(
                (len(self.time), self.max_agents), np.nan)
            for ti, t in enumerate(self.time):
                stop_idx = len(self.snapshots[t][state])
                self.trajectories[state][ti, 0:stop_idx] = \
                self.snapshots[t][state]
        return

    def plot(self, index=0, fig_args=None, pl_args=None):
        """
        Plots time series of each state as a line graph
        Args:
           index: index of the female individual, must be less the analyzer's max_pop_size
        """
        fig_args = sc.mergedicts(fig_args, {'figsize': (15, 6)})
        pl_args = sc.mergedicts(pl_args, {'markersize': 12,
                                                'markeredgecolor': 'black',
                                                'markeredgewidth': 1.5,
                                                'ls': 'none'})
        ymin, ymax = 0, 4
        rows, cols = sc.get_rows_cols(1)

        fig = pl.figure(**fig_args)
        pl.subplot(rows, cols,1)

        preg_outcome_state = ["parity", "stillbirth", "miscarriage", "abortion"]
        preg_outcome_symbl = {"stillbirth": u"\u29BB",
                              "parity": u"\u29C2",       # use to determine live births
                              "miscarriage" : u"\u29B0",
                              "abortion": u"\u2A09"}


        preg_outcome_lbl = {"stillbirth": "Stillbirth",
                            "parity": "Live birth",      # use to determine live births
                            "miscarriage": "Miscarriage",
                            "abortion": "Abortion"}

        # Alive
        state = "alive"
        temp = self._state_intervals(state, index)
        temp_age = self._transform_to_age(temp, index)
        al_bar = pl.broken_barh(temp_age,
                                (3.5, 0.25),
                                facecolors="lemonchiffon", label=f"{state}",
                                hatch="..")

        # Add vertical lines indicating age at start of simulation
        yoffset = 0.25
        yp = [ymin-yoffset, ymin+yoffset, (ymax-ymin)/2, ymax-yoffset, ymax+yoffset],
        xp = temp_age[0, 0]*np.ones(len(yp))
        pl.plot(xp, yp, color='k', ls=':', marker=">")


        # Mark the age at the end of the simulation (if they died before the yellowish "alive" bar will stop befor this line)
        sim_len = len(self.time) * self.m2y
        xp = (temp_age[0, 0] + sim_len) *len(yp)
        pl.plot(xp, yp, color='k', ls=':', marker="<")

        # Sexually active
        state = "sexually_active"
        temp = self._state_intervals(state, index)
        sa_bar = pl.broken_barh(self._transform_to_age(temp, index),
                       (1, 1), facecolors="#9bf1fe", label=f"{state}")

        # Is she on contraception?
        with plt.rc_context({"hatch.linewidth": 3}):
        # NOTE: the context manager does not work (known matplotlib bug)
        # but if we set the a thicker hatch linewidth as common value for all bars
        # then some hatch patters look ugly ¯\_(ツ)_/¯ ...
            state = "on_contra"
            temp = self._state_intervals(state, index)
            temp_age = self._transform_to_age(temp, index)
            on_contra_yo, on_contra_height = 1, 1
            oc_bar = plt.broken_barh(self._transform_to_age(temp, index),
                                     (on_contra_yo, on_contra_height),
                                     facecolors="none", label=f"{state}", hatch="//")

        # What contraceptive method is she on?
        state = "method"
        contramethod = self.trajectories[state][:, index]
        bbox = dict(boxstyle="round", fc="w", ec="none", alpha=0.8)
        for start, age in zip(temp[:, 0], temp_age[:, 0]):
            pl.annotate(
                f"{self.method_map[contramethod[start]]}",
                (age, on_contra_yo + on_contra_height/2), xycoords='data',
                xytext=(age, on_contra_yo + on_contra_height/2), textcoords='data',
                size=10, va="center", ha="left",
                bbox=bbox)

        # Pregnant
        state = "pregnant"
        temp = self._state_intervals(state, index)
        pr_bar = pl.broken_barh(self._transform_to_age(temp, index),
                       (2, 1), facecolors="deeppink", label=f"{state}")

        # What happened with the pregnancy?
        po_plots = []
        for state in preg_outcome_state:
            if state not in preg_outcome_state:
                break
            else:
                temp = self._preg_outcome_instants(state, index)

            lbl = f"{preg_outcome_lbl[state]}"
            marker = f"${preg_outcome_symbl[state]}$"
            age = self.trajectories["age"][:, index]
            po = pl.plot(age, 2.5*temp, marker=marker, label=f"{lbl}",  **pl_args)
            po_plots.append(po[0])


        # Postpartum
        state = "postpartum"
        temp = self._state_intervals(state, index)
        pp_bar = pl.broken_barh(self._transform_to_age(temp, index),
                       (0.7, 2.5), facecolors="oldlace", label=f"{state}")

        # Lam period
        state = "lam"
        temp = self._state_intervals(state, index)
        lam_bar = pl.broken_barh(self._transform_to_age(temp, index),
                       (0.7, 2.5), facecolors="none", edgecolors="#a5a5a5",
                       label=f"{state}",
                       hatch="\\\\\\"
                       )


        # Labels and annotations
        pl.xlim(-0.5, 95)
        pl.ylim(ymin, ymax)
        pl.xlabel('Age (years)')
        pl.ylabel('')
        pl.title(f"Life course of a woman")

        # Hierarchical legend
        # TODO: figure out a better/systematic way to do the bbox anchoring of multiple legends
        lgn_loc = 'center right'
        lvl_1_fnt_sze = 12
        lvl_2_fnt_sze = 9

        nonpreg_lgnd = plt.legend([al_bar, sa_bar, oc_bar],
                                  ["Alive", "Sexually active", "On contraception"],
                                 fontsize=lvl_1_fnt_sze, loc=lgn_loc,
                                 bbox_to_anchor=(0.965, 0.5),
                                 frameon=False)

        preg_lgnd = plt.legend([pr_bar], ["Pregnant"],
                                 fontsize=lvl_1_fnt_sze, loc=lgn_loc,
                                 bbox_to_anchor=(0.91, 0.35),
                                 frameon=False)

        lbls = [po._label for po in po_plots]

        po_lgnd = plt.legend(po_plots,
                             lbls,
                             fontsize=lvl_2_fnt_sze,
                             loc=lgn_loc, bbox_to_anchor=(0.93, 0.25),
                             frameon=False)

        pp_lgnd = plt.legend([lam_bar, pp_bar], ["LAM", "Pospartum"],
                                 fontsize=lvl_1_fnt_sze, loc=lgn_loc,
                                 bbox_to_anchor=(0.92, 0.1),
                                 frameon=False)

        plt.gca().add_artist(nonpreg_lgnd)
        plt.gca().add_artist(preg_lgnd)
        plt.gca().add_artist(po_lgnd)
        plt.gca().add_artist(pp_lgnd)

        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))

        # Set the minor ticks at every year
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))


        return fig

    def _state_intervals(self, state, index):
        """
        Extract information about start and length of an interval where the
        state is true. Works only for boolean States. Needed for broken_barh()
        plots.
        """
        # Find where we go from nonzero to zero in our data (mostly boolean states)
        data_padded = np.pad(self.trajectories[state][:, index],
                             (1, 1), mode='constant')
        crossings = np.where(np.diff(data_padded != 0))[0]

        starts = crossings[:-1:2]
        ends   = crossings[1::2]
        lengths = ends - starts

        intervals = np.column_stack((starts, lengths))
        return intervals

    def _transform_to_age(self, intervals, index):
        age = self.trajectories["age"][:, index]
        intervals_age = np.full(shape=intervals.shape, fill_value=0.0, dtype=np.float64)
        intervals_age[:, 1] = self.m2y*intervals[:, 1].astype(np.float64)
        intervals_age[:, 0] = age[intervals[:, 0]]
        return intervals_age


    def _preg_outcome_instants(self, state, index):
        """
        Use the States that count how many instances of each type of pregnancy
        outcome a woman has over the course of her life, to extract the
        event instants.
        """
        data = self.trajectories[state][:, index]
        temp = np.full(shape=data.shape,
                       fill_value=np.nan)
        instants = sc.findinds(np.diff(data)) + 1
        temp[instants] = 1.0
        return temp


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
        self._raw[sim.ti, :] = np.histogram(ages, self.bins)[0]
        self.data[sim.ti, :] = self._raw[sim.ti, :]/self._raw[sim.ti, :].sum()

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


class track_switching(Analyzer):

    def __init__(self):
        super().__init__()
        self.results = sc.objdict()

    def initialize(self, sim):
        n_methods = len(sim.methods)
        keys = [
            'switching_events_annual',
            'switching_events_postpartum',
            'switching_events_<18',
            'switching_events_18-20',
            'switching_events_21-25',
            'switching_events_26-35',
            'switching_events_>35',
            'switching_events_pp_<18',
            'switching_events_pp_18-20',
            'switching_events_pp_21-25',
            'switching_events_pp_26-35',
            'switching_events_pp_>35',
        ]
        for key in keys:
            self.results[key] = {}  # CK: TODO: refactor
            for p in range(sim.npts):
                self.results[key][p] = np.zeros((n_methods, n_methods), dtype=int)

    def apply(self, sim):
        # Store results of number of switching events in each age group
        ti = sim.ti
        r = sim.results
        scale = sim['scaled_pop'] / sim['n_agents']
        switch_events = r.pop('switching')
        self.results['switching_events_<18'][ti] = scale * r.switching_annual['<18']
        self.results['switching_events_18-20'][ti] = scale * r.switching_annual['18-20']
        self.results['switching_events_21-25'][ti] = scale * r.switching_annual['21-25']
        self.results['switching_events_26-35'][ti] = scale * r.switching_annual['26-35']
        self.results['switching_events_>35'][ti] = scale * r.switching_annual['>35']
        self.results['switching_events_pp_<18'][ti] = scale * r.switching_postpartum['<18']
        self.results['switching_events_pp_18-20'][ti] = scale * r.switching_postpartum['18-20']
        self.results['switching_events_pp_21-25'][ti] = scale * r.switching_postpartum['21-25']
        self.results['switching_events_pp_26-35'][ti] = scale * r.switching_postpartum['26-35']
        self.results['switching_events_pp_>35'][ti] = scale * r.switching_postpartum['>35']
        self.results['switching_events_annual'][ti] = scale * switch_events['annual']
        self.results['switching_events_postpartum'][ti] = scale * switch_events['postpartum']

        return