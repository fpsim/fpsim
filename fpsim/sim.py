"""
Defines the Sim class, the core class of the FP model (FPsim).
"""

# %% Imports
import numpy as np  # Needed for a few things not provided by pl
import pylab as pl
import seaborn as sns
import sciris as sc
import pandas as pd
import starsim as ss
import fpsim as fp
from .settings import options as fpo
from . import utils as fpu
from . import defaults as fpd
from . import parameters as fpp
from . import people as fpppl
from . import methods as fpm
from . import education as fped

# Specify all externally visible things this file defines
__all__ = ['Sim']


#%% Plotting helper functions

def fixaxis(useSI=True, set_lim=True, legend=True):
    """ Format the axis using SI units and limits """
    if legend:
        pl.legend()  # Add legend
    if set_lim:
        sc.setylim()
    if useSI:
        sc.SIticks()
    return


def tidy_up(fig, do_show=None, do_save=None, filename=None):
    """ Helper function to handle the slightly complex logic of showing, saving, returing -- not for users """

    # Handle inputs
    if do_show is None: do_show = fpo.show
    if do_save is None: do_save = fpo.save
    backend = pl.get_backend()

    # Handle show
    if backend == 'agg':  # Cannot show plots for a non-interactive backend
        do_show = False
    if do_show:  # Now check whether to show, and atually do it
        pl.show()

    # Handle saving
    if do_save:
        if isinstance(do_save, str):  # No figpath provided - see whether do_save is a figpath
            filename = sc.makefilepath(filename)  # Ensure it's valid, including creating the folder
        sc.savefig(fig=fig, filename=filename)  # Save the figure

    # Handle close
    if fpo.close and not do_show:
        pl.close(fig)

    # Return the figure or figures unless we're in Jupyter
    if not fpo.returnfig:
        return
    else:
        return fig


# %% Sim class

class Sim(ss.Sim):
    """
    The Sim class handles the running of the simulation. It extends the Starim Sim class, so all Starsim Sim methods
    are available to FPsims.

    When a Sim is initialized, it triggers the creation of the population. Methods related
    to creating, initializing, and updating people can be found in the People class.

    Args:
        pars     (dict):   parameters to modify from their default values
        location (str):    name of the location (country) to look for data file to load
        label    (str):    the name of the simulation (useful to distinguish in batch runs)
        track_children (bool): whether to track links between mothers and their children (slow, so disabled by default)
        kwargs   (dict):   additional parameters; passed to ``fp.make_pars()``

    **Examples**::

        sim = fp.Sim()
        sim = fp.Sim(n_agents=10e3, location='senegal', label='My small Senegal sim')
    """

    def __init__(self, pars=None, sim_pars=None, fp_pars=None, contra_pars=None, edu_pars=None,
                 fp_module=None, contraception_module=None, empowerment_module=None, education_module=None,
                 label=None, people=None, demographics=None, diseases=None, networks=None,
                 interventions=None, analyzers=None, connectors=None, copy_inputs=True, **kwargs):

        # Inputs and defaults
        self.contra_pars = None    # Parameters for the contraception module - processed later
        self.edu_pars = None     # Parameters for the education module - processed later
        self.fp_pars = None       # Parameters for the family planning module - processed later
        self.pars = None        # Parameters for the simulation - processed later

        # Call the constructor of the parent class WITHOUT pars or module args, the make defaults
        super().__init__(pars=None, label=label)
        self.pars = fpp.make_sim_pars()  # Make default parameters using values from parameters.py

        # Separate the parameters, storing sim pars and fp_pars now and saving module pars to process in init
        # Four sources of par values in decreasing order of priority:
        # 1-2. kwargs == args (if multiple definitions, raise exception)
        # 3. pars
        # 4. default pars
        # combine copies of them in this order if copy_inputs
        # remap any as necessary
        # separate into sim and fp-specific pars
        sim_kwargs = dict(label=label, people=people, demographics=demographics, diseases=diseases, networks=networks,
                    interventions=interventions, analyzers=analyzers, connectors=connectors)
        sim_kwargs = {key: val for key, val in sim_kwargs.items() if val is not None}
        all_sim_pars = self.separate_pars(pars, sim_pars, fp_pars, contra_pars, edu_pars, sim_kwargs, **kwargs)
        self.pars.update(all_sim_pars)

        # Process modules by adding them as Starsim connectors
        default_contra = fpm.StandardChoice(location=self.pars.location, pars=self.contra_pars)
        default_edu = fped.Education(location=self.pars.location, pars=self.edu_pars)
        default_fp = fp.FPmod(location=self.pars.location, pars=self.fp_pars)
        contraception_module = contraception_module or sc.dcp(default_contra)
        education_module = education_module or sc.dcp(default_edu)
        fp_module = fp_module or sc.dcp(default_fp)
        connectors = sc.tolist(connectors) + [contraception_module, education_module, fp_module]
        if empowerment_module is not None:
            connectors += sc.tolist(empowerment_module)
        self.pars['connectors'] = connectors  # TODO, check this

        # Metadata and settings
        # self.test_mode = False
        fpu.set_metadata(self)  # Set version, date, and git info
        self.summary = None

        # # Add a new parameter to pars that determines the size of the circular buffer = TODO, remove?
        # unit = self.pars.unit if self.pars.unit != "" else 'year'
        # self.fp_pars['tiperyear'] = ss.time_ratio('year', 1, unit, self.pars.dt)

        return

    # Basic properties
    @property
    def ty(self):
        return self.t.tvec[self.ti]  # years elapsed since beginning of sim (ie, 25.75... )

    @property
    def y(self):
        return self.t.yearvec[self.ti]

    @staticmethod
    def remap_pars(pars):
        """
        Remap the parameters to the new names. This is useful for backwards compatibility.
        """
        if 'start_year' in pars:
            pars['start'] = pars.pop('start_year')
        if 'end_year' in pars:
            pars['stop'] = pars.pop('end_year')
        if 'seed' in pars:
            pars['rand_seed'] = pars.pop('seed')
        if 'location' in pars and pars['location'] == 'test':
            pars['location'] = 'senegal'
            pars['test'] = True
        return pars

    # def separate_pars(self, pars):
    def separate_pars(self, pars=None, sim_pars=None, fp_pars=None, contra_pars=None, edu_pars=None, sim_kwargs=None, **kwargs):
        """
        Separate the parameters into simulation and fp-specific parameters.
        """
        # Marge in pars and kwargs
        all_pars = fpp.mergepars(pars, sim_pars, fp_pars, contra_pars, edu_pars, sim_kwargs, kwargs)
        all_pars = self.remap_pars(all_pars)  # Remap any v2 parameters to v3 names

        # Deal with sim pars
        user_sim_pars = {k: v for k, v in all_pars.items() if k in self.pars.keys()}
        for k in user_sim_pars: all_pars.pop(k)
        sim_pars = sc.mergedicts(user_sim_pars, sim_pars, _copy=True)

        # Deal with fp pars
        default_fp_pars = fpp.make_fp_pars()
        user_fp_pars = {k: v for k, v in all_pars.items() if k in default_fp_pars.keys()}
        for k in user_fp_pars: all_pars.pop(k)
        fp_pars = sc.mergedicts(user_fp_pars, fp_pars, _copy=True)

        # Deal with contraception module pars
        default_contra_pars = fpm.make_contra_pars()
        user_contra_pars = {k: v for k, v in all_pars.items() if k in default_contra_pars.keys()}
        for k in user_contra_pars: all_pars.pop(k)
        contra_pars = sc.mergedicts(user_contra_pars, contra_pars, _copy=True)

        # Deal with education pars
        default_edu_pars = fped.make_edu_pars()
        user_edu_pars = {k: v for k, v in all_pars.items() if k in default_edu_pars.keys()}
        for k in user_edu_pars: all_pars.pop(k)
        edu_pars = sc.mergedicts(user_edu_pars, edu_pars, _copy=True)

        # Raise an exception if there are any leftover pars
        if all_pars:
            raise ValueError(f'Unrecognized parameters: {all_pars.keys()}. Refer to parameters.py for parameters.')

        # Store the parameters for the modules - thse will be fed into the modules during init
        self.fp_pars = fp_pars    # Parameters for the family planning module
        self.contra_pars = contra_pars    # Parameters for contraceptive choice
        self.edu_pars = edu_pars      # Parameters for the education module
        return sim_pars

    def init(self, force=False):
        """ Fully initialize the Sim with modules, people and result storage"""
        if force or not self.initialized:
            fpu.set_seed(self.pars['rand_seed'])
            if self.pars.people is None:
                ap = self.pars['connectors'][-1].pars['age_pyramid']  # TODO TEMP
                self.pars.people = fpppl.People(n_agents=self.pars.n_agents, age_pyramid=ap)
            super().init(force=force)

        return self

    def init_results(self):
        """
        Initialize the results dictionary. This is called at the start of the simulation.
        """
        super().init_results()
        scaling_kw = dict(shape=self.t.npts, timevec=self.t.timevec, dtype=int, scale=True)
        for key in fpd.sim_results:
            self.results += ss.Result(key, label=key, **scaling_kw)
        return

    def update_mortality(self):
        """
        Update infant and maternal mortality for the sim's current year.
        Update general mortality trend as this uses a spline interpolation instead of an array.
        """

        mapping = {
            'age_mortality': 'gen_trend',
            'infant_mortality': 'infant',
            'maternal_mortality': 'maternal',
            'stillbirth_rate': 'stillbirth',
        }

        self.fp_pars['mortality_probs'] = {}
        for key1, key2 in mapping.items():
            ind = sc.findnearest(self.pars.fp[key1]['year'], self.y)
            val = self.pars.fp[key1]['probs'][ind]
            self.pars.fp['mortality_probs'][key2] = val

        return

    def start_step(self):
        super().start_step()
        self.update_mortality()
        # self.people.step()
        return

    # Function to scale all y-axes in fig based on input channel
    @staticmethod
    def conform_y_axes(figure, bottom=0, top=100):
        for axes in figure.axes:
            axes.set_ylim([bottom, top])
        return figure

    def plot(self, to_plot=None, xlims=None, ylims=None, do_save=None, do_show=True, filename='fpsim.png', style=None,
             fig_args=None,
             plot_args=None, axis_args=None, fill_args=None, label=None, new_fig=True, colors=None):
        """
        Plot the results -- can supply arguments for both the figure and the plots.

        Args:
            to_plot   (str/dict): What to plot (e.g. 'default' or 'cpr'), or a dictionary of result:label pairs
            xlims     (list/dict): passed to pl.xlim() (use ``[None, None]`` for default)
            ylims     (list/dict): passed to pl.ylim()
            do_save   (bool): Whether or not to save the figure. If a string, save to that filename.
            do_show   (bool): Whether to show the plots at the end
            filename  (str):  If a figure is saved, use this filename
            style     (bool): Custom style arguments
            fig_args  (dict): Passed to pl.figure() (plus ``nrows`` and ``ncols`` for overriding defaults)
            plot_args (dict): Passed to pl.plot()
            axis_args (dict): Passed to pl.subplots_adjust()
            fill_args (dict): Passed to pl.fill_between())
            label     (str):  Label to override default
            new_fig   (bool): Whether to create a new figure (true unless part of a multisim)
            colors    (list/dict): Colors for plots with multiple lines
        """
        if to_plot is None: to_plot = 'default'
        fig_args = sc.mergedicts(dict(figsize=(16, 10), nrows=None, ncols=None), fig_args)
        plot_args = sc.mergedicts(dict(lw=2, alpha=0.7), plot_args)
        axis_args = sc.mergedicts(dict(left=0.1, bottom=0.05, right=0.9, top=0.97, wspace=0.2, hspace=0.25), axis_args)
        fill_args = sc.mergedicts(dict(alpha=0.2), fill_args)

        with fpo.with_style(style):

            nrows, ncols = fig_args.pop('nrows'), fig_args.pop('ncols')
            fig = pl.figure(**fig_args) if new_fig else pl.gcf()
            pl.subplots_adjust(**axis_args)

            if to_plot is not None and 'as_' in to_plot:
                nrows, ncols = 2, 3

            res = self.results  # Shorten since heavily used
            agelim = ('-'.join([str(self.fp_pars['low_age_short_int']), str(
                self.fp_pars['high_age_short_int'])]))  ## age limit to be added to the title of short birth interval plot

            if isinstance(to_plot, dict):
                pass
            elif isinstance(to_plot, str):
                if to_plot == 'default':
                    to_plot = {
                        'mcpr_by_year':                'Modern contraceptive prevalence rate (%)',
                        'cum_live_births_by_year':     'Live births',
                        'cum_stillbirths_by_year':     'Stillbirths',
                        'cum_maternal_deaths_by_year': 'Maternal deaths',
                        'cum_infant_deaths_by_year':   'Infant deaths',
                        'imr':                         'Infant mortality rate',
                    }
                elif to_plot == 'cpr':
                    to_plot = {
                        'mcpr': 'MCPR (modern contraceptive prevalence rate)',
                        'cpr':  'CPR (contraceptive prevalence rate)',
                        'acpr': 'ACPR (alternative contraceptive prevalence rate)',
                    }
                elif to_plot == 'mortality':
                    to_plot = {
                        'mmr':                         'Maternal mortality ratio',
                        'cum_maternal_deaths_by_year': 'Maternal deaths',
                        'cum_infant_deaths_by_year':   'Infant deaths',
                        'imr':                         'Infant mortality rate',
                        }
                elif to_plot == 'apo': #adverse pregnancy outcomes
                    to_plot = {
                        'cum_pregnancies_by_year':     'Pregnancies',
                        'cum_stillbirths_by_year':     'Stillbirths',
                        'cum_miscarriages_by_year':    'Miscarriages',
                        'cum_abortions_by_year':       'Abortions',
                        }
                elif to_plot == 'method':
                    to_plot = {
                        'method_usage':                 'Method usage'
                    }
                elif to_plot == 'short-interval':
                    to_plot = {
                        'proportion_short_interval_by_year':     f"Proportion of short birth interval [{age_group})" for age_group in agelim.split()
                    }
                elif to_plot is not None:
                    errormsg = f"Your to_plot value: {to_plot} is not a valid option"
                    raise ValueError(errormsg)
            else:
                errmsg = f"to_plot can be a dictionary or a string. A {type(to_plot)} is not a valid option."
                raise TypeError(errmsg)

            rows, cols = sc.getrowscols(len(to_plot), nrows=nrows, ncols=ncols)
            if to_plot == 'cpr':
                rows, cols = 1, 3
            for p, key, reslabel in sc.odict(to_plot).enumitems():
                ax = pl.subplot(rows, cols, p + 1)

                this_res = res[key]
                is_dist = hasattr(this_res, 'best')
                if is_dist:
                    y, low, high = this_res.best, this_res.low, this_res.high
                else:
                    y, low, high = this_res, None, None

                # Figure out x axis
                years = res['tfr_years']
                timepoints = res.timevec  # Likewise
                x = None
                for x_opt in [years, timepoints]:
                    if len(y) == len(x_opt):
                        x = x_opt
                        break
                if x is None:
                    errormsg = f'Could not figure out how to plot {key}: result of length {len(y)} does not match a known x-axis'
                    raise RuntimeError(errormsg)

                percent_keys = ['mcpr_by_year', 'mcpr', 'cpr', 'acpr', 'method_usage',
                                'proportion_short_interval_by_year']
                if (
                        'cpr_' in key or 'acpr_' in key or 'mcpr_' in key or 'proportion_short_interval_' in key) and 'by_year' not in key:
                    percent_keys = percent_keys + list(to_plot.keys())
                if key in percent_keys and key != 'method_usage':
                    y = y * 100 # why doesn't *= syntax work here? Is it overloaded on Result objects?
                    if is_dist:
                        low *= 100
                        high *= 100

                # Handle label
                if label is not None:
                    plotlabel = label
                else:
                    if new_fig:  # It's a new figure, use the result label
                        plotlabel = reslabel
                    else:  # Replace with sim label to avoid duplicate labels
                        plotlabel = self.label

                # Actually plot
                if key == "method_usage":
                    data = self.format_method_df(timeseries=True)
                    method_names = data['Method'].unique()
                    flipped_data = {method: [percentage for percentage in data[data['Method'] == method]['Percentage']]
                                    for method in method_names}
                    colors = [colors[method] for method in method_names] if isinstance(colors, dict) else colors
                    ax.stackplot(data["Year"].unique(), list(flipped_data.values()), labels=method_names, colors=colors)
                else:
                    ax.plot(x, y, label=plotlabel, **plot_args)

                if is_dist:
                    if 'c' in plot_args:
                        fill_args['facecolor'] = plot_args['c']
                    ax.fill_between(x, low, high, **fill_args)

                # Plot interventions, if present
                if hasattr(self, 'interventions'):
                    for intv in sc.tolist(self['interventions']):
                        if hasattr(intv, 'plot_intervention'): # Don't plot e.g. functions
                            intv.plot_intervention(self, ax)

                # Handle annotations
                as_plot = (
                                  'cpr_' in key or 'acpr_' in key or 'mcpr_' in key or 'pregnancies_' in key or 'stillbirths' in key or 'tfr_' in key or 'imr_' in key or 'mmr_' in key or 'births_' in key or 'proportion_short_interval_' in key) and 'by_year' not in key
                fixaxis(useSI=fpd.useSI, set_lim=new_fig)  # If it's not a new fig, don't set the lim
                if key in percent_keys:
                    pl.ylabel('Percentage')
                elif 'mmr' in key:
                    pl.ylabel('Deaths per 100,000 live births')
                elif 'imr' in key:
                    pl.ylabel('Deaths per 1,000 live births')
                elif 'tfr_' in key:
                    pl.ylabel('Fertility rate per 1,000 women')
                elif 'mmr_' in key:
                    pl.ylabel('Maternal deaths per 10,000 births')
                elif 'stillbirths_' in key:
                    pl.ylabel('Number of stillbirths')
                elif 'intent' or 'employment' in key:
                    pl.ylabel('Percentage')
                else:
                    pl.ylabel('Count')
                pl.xlabel('Year')
                pl.title(reslabel, fontweight='bold')
                if xlims is not None:
                    pl.xlim(xlims)
                if ylims is not None:
                    pl.ylim(ylims)
                if (key == "method_usage") or as_plot:  # need to overwrite legend for some plots
                    ax.legend(loc='upper left', frameon=True)
                if 'cpr' in to_plot and '_' not in to_plot:
                    if is_dist:
                        top = int(np.ceil(max(self.results['acpr'].high) / 10.0)) * 10  # rounding up to nearest 10
                    else:
                        top = int(np.ceil(max(self.results['acpr']) * 10.0)) * 10
                    self.conform_y_axes(figure=fig, top=top)
        return tidy_up(fig=fig, do_show=do_show, do_save=do_save, filename=filename)

    def plot_age_first_birth(self, do_show=None, do_save=None, fig_args=None, filename="first_birth_age.png"):
        """
        Plot age at first birth

        Args:
            fig_args (dict): arguments to pass to ``pl.figure()``
            do_show (bool): whether the user wants to show the output plot (default: true)
            do_save (bool): whether the user wants to save the plot to filepath (default: false)
            filename (str): the name of the path to output the plot
        """
        birth_age = self.people.first_birth_age
        data = birth_age[birth_age > 0]
        fig = pl.figure(**sc.mergedicts(dict(figsize=(7, 5)), fig_args))
        pl.title("Age at first birth")
        sns.boxplot(x=data, orient='v', notch=True)
        pl.xlabel('Age (years')
        return tidy_up(fig=fig, do_show=do_show, do_save=do_save, filename=filename)

    def format_method_df(self, method_list=None, timeseries=False):
        """
        Outputs a dataframe for method mix plotting for either a single year or a timeseries

        Args:
            method_list (list):
                list of proportions where each index is equal to the integer value of the corresponding method
            timeseries (boolean):
                if true, provides a dataframe with data from every year, otherwise a method_list is required for the year

        Returns:
            pandas.DataFrame with columns ["Percentage", "Method", "Sim", "Seed"] and optionally "Year" if timeseries
        """
        inv_method_map = {index: name for name, index in self.fp_pars['methods']['map'].items()}

        def get_df_from_result(method_list):
            df_dict = {"Percentage": [], "Method": [], "Sim": [], "Seed": []}
            for method_index, prop in enumerate(method_list):
                if method_index != fpd.method_map['None']:
                    df_dict["Percentage"].append(100 * prop)
                    df_dict['Method'].append(inv_method_map[method_index])
                    df_dict['Sim'].append(self.label)
                    df_dict['Seed'].append(self.pars['rand_seed'])

            return pd.DataFrame(df_dict)

        if not timeseries:
            return get_df_from_result(method_list)

        else:
            initial_year = self.pars['start_year']
            total_df = pd.DataFrame()
            for year_offset, method_list in enumerate(self.results['method_usage']):
                year_df = self.format_method_df(method_list)
                year_df['Year'] = [initial_year + year_offset] * len(year_df)
                total_df = pd.concat([total_df, year_df], ignore_index=True)
            return total_df

    def list_available_results(self):
        """Pretty print availbale results keys, sorted alphabetically"""
        output = 'Result keys:\n'
        keylen = 35  # Maximum key length  -- "interactive"
        for k in sorted(self.results.keys()):
            keystr = sc.colorize(f'  {k:<{keylen}s} ', fg='blue', output=True)
            reprstr = sc.indent(n=0, text=keystr, width=None)
            output += f'{reprstr}'
        print(output)
