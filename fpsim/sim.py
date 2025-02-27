"""
Defines the Sim class, the core class of the FP model (FPsim).
"""

# %% Imports
import numpy as np  # Needed for a few things not provided by pl
import pylab as pl
import seaborn as sns
import sciris as sc
import pandas as pd
from .settings import options as fpo
from . import utils as fpu
from . import defaults as fpd
from . import base as fpb
from . import parameters as fpp
from . import people as fpppl
from . import methods as fpm
from . import education as fped

# Specify all externally visible things this file defines
__all__ = ['Sim', 'MultiSim', 'parallel']


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

class Sim(fpb.BaseSim):
    """
    The Sim class handles the running of the simulation. This class handles the mechanics
    of the actual simulation, while BaseSim takes care of housekeeping (saving,
    loading, exporting, etc.). Please see the BaseSim class for additional methods.
    
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

    def __init__(self, pars=None, location=None, label=None, track_children=False, regional=False,
                 contraception_module=None, empowerment_module=None, education_module=None, **kwargs):

        pars = sc.dcp(pars)
        # Handle location
        if location is None:
            if pars is not None and pars.get('location'):
                location = pars.pop('location')
        location = fpd.get_location(location, printmsg=True)  # Handle location
        self.location = location

        # Make parameters
        pars = fpp.pars(location=location, **sc.mergedicts(pars, kwargs))  # Update with location-specific parameters

        # Validate and initialize
        mismatches = [key for key in kwargs.keys() if key not in fpp.par_keys]
        if len(mismatches):
            errormsg = f'Key(s) {mismatches} not found; available keys are {fpp.par_keys}'
            raise sc.KeyNotFoundError(errormsg)

        super().__init__(pars, location=location, **kwargs)  # Initialize and set the parameters as attributes

        # Metadata and settings
        self.initialized = False
        self.already_run = False
        self.test_mode = False
        self.label = label
        self.track_children = track_children
        self.regional = regional
        self.ti = None  # The current timestep of the simulation
        self.scale = pars['scaled_pop'] / pars['n_agents'] if pars['scaled_pop'] is not None else 1
        fpu.set_metadata(self)  # Set version, date, and git info
        self.summary = None

        # Add a new parameter to pars that determines the size of the circular buffer
        self.pars['tiperyear'] = self.tiperyear

        # People and results - intialized later
        self.results = {}
        self.people = None  # Sims are generally constructed without people, since People construction is time-consuming

        # Add modules, also initialized later
        # self.contraception_module = contraception_module or fpm.RandomChoice()
        self.education_module = education_module
        self.contraception_module = contraception_module or fpm.SimpleChoice(location=location)
        # self.education_module = education_module or fped.Education(location=location)
        self.empowerment_module = empowerment_module

        return

    # Basic properties
    @property
    def ty(self):
        return self.ind2year(self.ti)  # years elapsed since beginning of sim (ie, 25.75... )

    @property
    def y(self):
        return self.ind2calendar(self.ti)  # y is calendar year of timestep (ie, 1975.75)

    def initialize(self, force=False):
        """ Fully initialize the Sim with people and result storage"""
        if force or not self.initialized:
            self.ti = 0  # The current time index
            fpu.set_seed(self['seed'])
            self.init_results()
            self.init_people()  # This step also initializes the empowerment and education modules if provided
            self.init_contraception()  # Initialize contraceptive methods
        return self

    def init_results(self):
        """
        Initialize result storage. Most default results are either arrays or lists; these are
        all stored in defaults.py. Any other results with different formats can also be added here.
        """
        for key in fpd.array_results:
            self.results[key] = np.zeros(int(self.npts))

        for key in fpd.list_results:
            self.results[key] = []

        # Store age-specific fertility rates
        self.results['asfr'] = {}
        self.results['method_usage'] = []
        for key in fpd.age_bin_map.keys():
            self.results['asfr'][key] = []
            self.results[f"tfr_{key}"] = []

        return

    def init_people(self):
        """
        Initialize people by calling the People constructor and initialization methods.
        See people.py for details of people construction.
        """
        self.people = fpppl.People(pars=self.pars, contraception_module=self.contraception_module,
                                    empowerment_module=self.empowerment_module, education_module=self.education_module)

    def init_contraception(self):
        if self.contraception_module is not None:
            self.people.decide_contraception(ti=self.ti, year=self.y, contraception_module=self.contraception_module)
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

        self['mortality_probs'] = {}
        for key1, key2 in mapping.items():
            ind = sc.findnearest(self[key1]['year'], self.y)
            val = self[key1]['probs'][ind]
            self['mortality_probs'][key2] = val

        return

    def update_mothers(self):
        """
        Add link between newly added individuals and their mothers
        TODO: move to People?
        """
        all_ppl = self.people.unfilter()
        for mother_index, postpartum in enumerate(all_ppl.postpartum):
            if postpartum and all_ppl.postpartum_dur[mother_index] < 2:
                for child in all_ppl.children[mother_index]:
                    all_ppl.mothers[child] = mother_index
        return

    def apply_interventions(self):
        """ Apply each intervention in the model """
        from . import interventions as fpi  # To avoid circular import
        for i, intervention in enumerate(sc.tolist(self['interventions'])):
            if isinstance(intervention, fpi.Intervention):
                if not intervention.initialized:  # pragma: no cover
                    intervention.initialize(self)
                intervention.apply(self)  # If it's an intervention, call the apply() method
            elif callable(intervention):
                intervention(self)  # If it's a function, call it directly
            else:  # pragma: no cover
                errormsg = f'Intervention {i} ({intervention}) is neither callable nor an Intervention object: it is {type(intervention)}'
                raise TypeError(errormsg)
        return

    def apply_analyzers(self):
        """ Apply each analyzer in the model """
        from . import analyzers as fpa  # To avoid circular import
        for i, analyzer in enumerate(sc.tolist(self['analyzers'])):
            if isinstance(analyzer, fpa.Analyzer):
                if not analyzer.initialized:  # pragma: no cover
                    analyzer.initialize(self)
                analyzer.apply(self)  # If it's an intervention, call the apply() method
            elif callable(analyzer):
                analyzer(self)  # If it's a function, call it directly
            else:  # pragma: no cover
                errormsg = f'Analyzer {i} ({analyzer}) is neither callable nor an Analyzer object: it is {type(analyzer)}'
                raise TypeError(errormsg)
        return

    def finalize_interventions(self):
        """ Make any final updates to interventions (e.g. to shrink) """
        from . import interventions as fpi  # To avoid circular import
        for intervention in sc.tolist(self['interventions']):
            if isinstance(intervention, fpi.Intervention):
                intervention.finalize(self)

    def finalize_analyzers(self):
        """ Make any final updates to analyzers (e.g. to shrink) """
        from . import analyzers as fpa  # To avoid circular import
        for analyzer in sc.tolist(self['analyzers']):
            if isinstance(analyzer, fpa.Analyzer):
                analyzer.finalize(self)

    def finalize_people(self):
        """Clean up and reset people's attributes at the end of a time step"""
        if not self.track_children:
            delattr(self.people, "mothers")

    def grow_population(self, n_new_people):
        """Expand population size"""
        # Births
        new_people = fpppl.People(
                    pars=self.pars, n=n_new_people, age=0,
                    education_module=self.education_module,
                    empowerment_module=self.empowerment_module
                    )
        new_people.decide_contraception(ti=self.ti, year=self.y, contraception_module=self.contraception_module)
        self.people += new_people

    def step(self):
        """ Update logic of a single time step """

        # Update mortality probabilities for year of sim
        self.update_mortality()

        # Update the people
        self.people.ti = self.ti
        self.people.ty = self.ty
        self.people.y = self.y

        # Step forward people's states and attributes
        self.people.step()

        # Apply interventions
        self.apply_interventions()

        # Populate the circular buffers with data from Peoples states that
        # are needed for methods/classes that use historical data (eg, need previous year's data)
        self.people.update_history_buffer()

        # Count results for this step
        step_results = self.people.get_step_results()

        # Store results
        res = sc.dictobj(**step_results)
        self.update_results(res, self.ti)

        # Add births
        n_new_people = res.births - res.infant_deaths  # Do not add agents who died before age 1 to population
        if n_new_people > 0: self.grow_population(n_new_people)

        # Update mothers
        if self.track_children:
            self.update_mothers()

        # Lastly, update analyzers. Needs to happen at the end of the sim as they report on events from this timestep
        self.apply_analyzers()

        self.people.step_age()

        return res

    def run(self, verbose=None):
        """ Run the simulation """

        # Initialize -- reset settings and results
        T = sc.timer()
        if verbose is None:
            verbose = self['verbose']
        self.initialize()
        if self.already_run:
            errormsg = 'Cannot re-run an already run sim; please recreate or copy prior to a run'
            raise RuntimeError(errormsg)

        # Main simulation loop
        for ti in range(self.npts):  # Range over number of timesteps in simulation (ie, 0 to 261 steps)

            self.ti = ti

            # Print progress
            elapsed = T.toc(output=True)
            if verbose:
                simlabel = f'"{self.label}": ' if self.label else ''
                string = f'  Running {simlabel}{self.y:0.0f} of {self["end_year"]} ({ti:2.0f}/{self.npts}) ({elapsed:0.2f} s) '
                if verbose >= 2:
                    sc.heading(string)
                elif verbose > 0:
                    if not (self.ty % int(1.0 / verbose)):
                        sc.progressbar(self.ti + 1, self.npts, label=string, length=20, newline=True)

            self.step()

        # Finalize people
        self.finalize_people()

        # Finalize results, interventions and analyzers
        self.finalize_results()
        self.finalize_interventions()
        self.finalize_analyzers()

        if verbose:
            print(f'Final population size: {self.n}.')
            elapsed = T.toc(output=True)
            print(f'Run finished for "{self.label}" after {elapsed:0.1f} s')

        self.summary = sc.objdict()
        self.summary.births = np.sum(self.results['births'])
        self.summary.deaths = np.sum(self.results['deaths'])
        self.summary.final = self.results['pop_size'][-1]

        self.already_run = True

        return self

    def update_results(self, res, ti):
        percent0to5 = (res.pp0to5 / res.total_women_fecund) * 100
        percent6to11 = (res.pp6to11 / res.total_women_fecund) * 100
        percent12to23 = (res.pp12to23 / res.total_women_fecund) * 100
        nonpostpartum = ((res.total_women_fecund - res.pp0to5 - res.pp6to11 - res.pp12to23) / res.total_women_fecund) * 100

        # Store results
        if self['scaled_pop']:
            scale = self['scaled_pop'] / self['n_agents']
        else:
            scale = 1
        self.results['t'][ti] = self.tvec[ti]
        self.results['pop_size_months'][ti] = self.n * scale
        self.results['births'][ti] = res.births * scale
        self.results['deaths'][ti] = res.deaths * scale
        self.results['stillbirths'][ti] = res.stillbirths * scale
        self.results['miscarriages'][ti] = res.miscarriages * scale
        self.results['abortions'][ti] = res.abortions * scale
        self.results['short_intervals'][ti] = res.short_intervals * scale
        self.results['secondary_births'][ti] = res.secondary_births * scale
        self.results['pregnancies'][ti] = res.pregnancies * scale
        self.results['total_births'][ti] = res.total_births * scale
        self.results['maternal_deaths'][ti] = res.maternal_deaths * scale
        self.results['infant_deaths'][ti] = res.infant_deaths * scale
        self.results['on_methods_mcpr'][ti] = res.on_methods_mcpr
        self.results['no_methods_mcpr'][ti] = res.no_methods_mcpr
        self.results['on_methods_cpr'][ti] = res.on_methods_cpr
        self.results['no_methods_cpr'][ti] = res.no_methods_cpr
        self.results['on_methods_acpr'][ti] = res.on_methods_acpr
        self.results['no_methods_acpr'][ti] = res.no_methods_acpr
        self.results['contra_access'][ti] = res.contra_access
        self.results['new_users'][ti] = res.new_users
        self.results['ever_used_contra'][ti] = res.ever_used_contra
        self.results['switchers'][ti] = res.switchers
        self.results['urban_women'][ti] = res.urban_women
        self.results['mcpr'][ti] = sc.safedivide(res.on_methods_mcpr, (res.no_methods_mcpr + res.on_methods_mcpr))
        self.results['cpr'][ti] = sc.safedivide(res.on_methods_cpr, (res.no_methods_cpr + res.on_methods_cpr))
        self.results['acpr'][ti] = sc.safedivide(res.on_methods_acpr, (res.no_methods_acpr + res.on_methods_acpr))
        self.results['pp0to5'][ti] = percent0to5
        self.results['pp6to11'][ti] = percent6to11
        self.results['pp12to23'][ti] = percent12to23
        self.results['parity0to1'][ti] = res.parity0to1
        self.results['parity2to3'][ti] = res.parity2to3
        self.results['parity4to5'][ti] = res.parity4to5
        self.results['parity6plus'][ti] = res.parity6plus
        self.results['wq1'][ti] = res.wq1
        self.results['wq2'][ti] = res.wq2
        self.results['wq3'][ti] = res.wq3
        self.results['wq4'][ti] = res.wq4
        self.results['wq5'][ti] = res.wq5
        self.results['nonpostpartum'][ti] = nonpostpartum
        self.results['total_women_fecund'][ti] = res.total_women_fecund * scale
        self.results['method_failures'][ti] = res.method_failures * scale

        # Education metrics
        self.results['edu_attainment'][ti] = res.edu_attainment
        self.results['edu_objective'][ti] = res.edu_objective

        # Intent
        # These will all be zero if empowerment module is not provided
        # Not used except for within kenya_empowerment repo
        if self.empowerment_module is not None:
            self.results['perc_contra_intent'][ti] = res.perc_contra_intent
            self.results['perc_fertil_intent'][ti] = res.perc_fertil_intent

            # Empowerment metrics
            # These will all be zero if empowerment module is not provided
            # Not used except for within kenya_empowerment repo
            self.results['paid_employment'][ti] = res.paid_employment
            self.results['decision_wages'][ti] = res.decision_wages
            self.results['decide_spending_partner'][ti] = res.decide_spending_partner
            self.results['buy_decision_major'][ti] = res.buy_decision_major
            self.results['buy_decision_daily'][ti] = res.buy_decision_daily
            self.results['buy_decision_clothes'][ti] = res.buy_decision_clothes
            self.results['decision_health'][ti] = res.decision_health
            self.results['has_savings'][ti] = res.has_savings
            self.results['has_fin_knowl'][ti] = res.has_fin_knowl
            self.results['has_fin_goals'][ti] = res.has_fin_goals
            #self.results['financial_autonomy'][ti] = res.financial_autonomy
            #self.results['decision_making'][ti] = res.decision_making

        for agekey in fpd.age_bin_map.keys():
            births_key = f'total_births_{agekey}'
            women_key = f'total_women_{agekey}'
            self.results[births_key][ti] = res.birth_bins[
                                              agekey] * scale  # Store results of total births per age bin for ASFR
            self.results[women_key][ti] = res.age_bin_totals[
                                             agekey] * scale  # Store results of total fecund women per age bin for ASFR

        scale = self.scale
        time_months = int(self.ti * self.pars["timestep"])  # time since the beginning of the sim, expresse in months
        # Calculate metrics over the last year in the model and save whole years and stats to an array
        if (time_months >= fpd.mpy) and (time_months % fpd.mpy) == 0:  # Start calculating annual metrics after we have at least 1 year of data
            self.results['tfr_years'].append(self.y-1.0)  # Substract one year as we're calculating the statistics between 1st jan 'year-1' and 1st jan 'year'. Results correspond to 'year-1'.
            stop_index = self.ti
            start_index = stop_index - self.tiperyear
            for res_name, new_res_name in fpd.to_annualize.items():
                res_over_year = self.annualize_results(res_name, start_index, stop_index)
                annual_res_name = f'{new_res_name}_over_year'
                self.results[annual_res_name].append(res_over_year)

            # self.results['method_usage'].append(self.compute_method_usage())  # only want this per year
            self.results['pop_size'].append(scale * self.n)  # CK: TODO: replace with arrays
            self.results['mcpr_by_year'].append(self.results['mcpr'][ti])
            self.results['cpr_by_year'].append(self.results['cpr'][ti])

            # Calculate annual ratios
            self.calculate_annual_ratios()

            tfr = 0
            for key in fpd.age_bin_map.keys():
                age_bin_births_year = np.sum(self.results['total_births_' + key][start_index:stop_index])
                age_bin_total_women_year = self.results['total_women_' + key][stop_index]
                age_bin_births_per_woman = sc.safedivide(age_bin_births_year, age_bin_total_women_year)
                self.results['asfr'][key].append(age_bin_births_per_woman * 1000)
                self.results[f'tfr_{key}'].append(age_bin_births_per_woman * 1000)
                tfr += age_bin_births_per_woman  # CK: TODO: check if this is right

            self.results['tfr_rates'].append(
                tfr * 5)  # CK: TODO: why *5? # SB: I think this corresponds to size of age bins?

    def finalize_results(self):
        # Convert all results to Numpy arrays
        for key, arr in self.results.items():
            if isinstance(arr, list):
                # These keys have list of lists with different lengths
                if key in ['imr_numerator', 'imr_denominator', 'mmr_numerator', 'mmr_denominator', 'imr_age_by_group',
                            'mmr_age_by_group', 'as_stillbirths', 'stillbirth_ages']:
                    self.results[key] = np.array(arr, dtype=object)
                else:
                    self.results[key] = np.array(arr)  # Convert any lists to arrays

        # Calculate cumulative totals
        self.results['cum_maternal_deaths_by_year'] = np.cumsum(self.results['maternal_deaths_over_year'])
        self.results['cum_infant_deaths_by_year'] = np.cumsum(self.results['infant_deaths_over_year'])
        self.results['cum_live_births_by_year'] = np.cumsum(self.results['live_births_over_year'])
        self.results['cum_stillbirths_by_year'] = np.cumsum(self.results['stillbirths_over_year'])
        self.results['cum_miscarriages_by_year'] = np.cumsum(self.results['miscarriages_over_year'])
        self.results['cum_abortions_by_year'] = np.cumsum(self.results['abortions_over_year'])
        self.results['cum_short_intervals_by_year'] = np.cumsum(self.results['short_intervals_over_year'])
        self.results['cum_secondary_births_by_year'] = np.cumsum(self.results['secondary_births_over_year'])
        self.results['cum_pregnancies_by_year'] = np.cumsum(self.results['pregnancies_over_year'])

        # Convert to an objdict for easier access
        self.results = sc.objdict(self.results)

    def annualize_results(self, key, start_index, stop_index):
        return self.scale * np.sum(self.results[key][start_index:stop_index])

    def calculate_annual_ratios(self):
        live_births_over_year = self.results['live_births_over_year'][-1]

        maternal_mortality_ratio = sc.safedivide(self.results['maternal_deaths_over_year'][-1], live_births_over_year) * 100000
        self.results['mmr'].append(maternal_mortality_ratio)

        infant_mortality_rate = sc.safedivide(self.results['infant_deaths_over_year'][-1], live_births_over_year) * 1000
        self.results['imr'].append(infant_mortality_rate)

        self.results['proportion_short_interval_by_year'].append(sc.safedivide(self.results['short_intervals_over_year'][-1], self.results['secondary_births_over_year'][-1]))
        return

    def store_postpartum(self):
        """
        Stores snapshot of who is currently pregnant, their parity, and various
        postpartum states in final step of model for use in calibration
        """

        min_age = 12.5
        max_age = self['age_limit_fecundity']

        ppl = self.people
        rows = []
        for i in range(len(ppl)):
            if ppl.alive[i] and ppl.sex[i] == 0 and min_age <= ppl.age[i] < max_age:
                row = dict(
                    Age=int(round(ppl.age[i])),
                    PP0to5=None,
                    PP6to11=None,
                    PP12to23=None,
                    NonPP=1 if not ppl.postpartum[i] else 0,
                    Pregnant=1 if ppl.pregnant[i] else 0,
                    Parity=ppl.parity[i],
                )
                if ppl.postpartum[i]:
                    pp_dur = ppl.postpartum_dur[i]
                    row['PP0to5'] = 1 if 0 <= pp_dur < 6 else 0
                    row['PP6to11'] = 1 if 6 <= pp_dur < 12 else 0
                    row['PP12to23'] = 1 if 12 <= pp_dur <= 24 else 0
                rows.append(row)

        pp = pd.DataFrame(rows, index=None,
                          columns=['Age', 'PP0to5', 'PP6to11', 'PP12to23', 'NonPP', 'Pregnant', 'Parity'])
        pp.fillna(0, inplace=True)
        return pp

    def to_df(self, include_range=False):
        """
        Export all sim results to a dataframe
        
        Args:
            include_range (bool): if True, and if the sim results have best, high, and low, then export all of them; else just best
        """
        raw_res = sc.odict(defaultdict=list)
        for reskey in self.results.keys():
            res = self.results[reskey]
            if isinstance(res, dict):
                for blh, blhres in res.items():  # Best, low, high
                    if len(blhres) == self.npts:
                        if not include_range and blh != 'best':
                            continue
                        if include_range:
                            blhkey = f'{reskey}_{blh}'
                        else:
                            blhkey = reskey
                        raw_res[blhkey] += blhres.tolist()
            elif sc.isarray(res) and len(res) == self.npts:
                raw_res[reskey] += res.tolist()
        df = pd.DataFrame(raw_res)
        self.df = df
        return df

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
            agelim = ('-'.join([str(self.pars['low_age_short_int']), str(
                self.pars['high_age_short_int'])]))  ## age limit to be added to the title of short birth interval plot

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
                elif to_plot == 'intent':
                    to_plot = {
                        'perc_contra_intent':     'Intent to use contraception (%)',
                        'perc_fertil_intent':     'Fertility intent (%)',
                        }
                elif to_plot == 'empowerment':
                    to_plot = {
                        'paid_employment':     'Paid employment (%)',
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
                timepoints = res['t']  # Likewise
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
                    y *= 100
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
                        top = int(np.ceil(max(self.results['acpr']) / 10.0)) * 10
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

    def compute_method_usage(self):
        """
        Computes method mix proportions from a sim object

        Returns:
            list of lists where list[years_after_start][method_index] == proportion of
            fecundity aged women using that method on that year
        """

        ppl = self.people
        min_age = fpd.min_age
        max_age = self['age_limit_fecundity']

        # filtering for women with appropriate characteristics
        bool_list = ppl.alive * (ppl.sex==0) * (ppl.age >= min_age) * (ppl.age <= max_age)
        filtered_methods = ppl.method[bool_list]

        unique, counts = np.unique(filtered_methods, return_counts=True)
        count_dict = dict(zip(unique, counts))

        result = [0] * (len(self.pars['methods']['eff']))
        for method in count_dict:
            result[method] = count_dict[method] / len(filtered_methods)

        return result

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
        inv_method_map = {index: name for name, index in self.pars['methods']['map'].items()}

        def get_df_from_result(method_list):
            df_dict = {"Percentage": [], "Method": [], "Sim": [], "Seed": []}
            for method_index, prop in enumerate(method_list):
                if method_index != fpd.method_map['None']:
                    df_dict["Percentage"].append(100 * prop)
                    df_dict['Method'].append(inv_method_map[method_index])
                    df_dict['Sim'].append(self.label)
                    df_dict['Seed'].append(self.pars['seed'])

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


# %% Multisim and running
class MultiSim(sc.prettyobj):
    """
    The MultiSim class handles the running of multiple simulations
    """

    def __init__(self, sims=None, base_sim=None, label=None, n=None, **kwargs):

        # Handle inputs
        if base_sim is None:
            if isinstance(sims, Sim):
                base_sim = sims
                sims = None
            elif isinstance(sims, list):
                base_sim = sims[0]
            else:
                errormsg = f'If base_sim is not supplied, sims must be either a single sim (treated as base_sim) or a list of sims, not {type(sims)}'
                raise TypeError(errormsg)

        # Set properties
        self.sims = sims
        self.base_sim = base_sim
        self.label = base_sim.label if (label is None and base_sim is not None) else label
        self.run_args = sc.mergedicts(kwargs)
        self.results = None
        self.which = None  # Whether the multisim is to be reduced, combined, etc.
        self.already_run = False
        fpu.set_metadata(self)  # Set version, date, and git info

        return

    def __len__(self):
        if isinstance(self.sims, list):
            return len(self.sims)
        elif isinstance(self.sims, Sim):
            return 1
        else:
            return 0

    def run(self, compute_stats=True, **kwargs):
        """ Run all simulations in the MultiSim """
        # Handle missing labels
        for s, sim in enumerate(sc.tolist(self.sims)):
            if sim.label is None:
                sim.label = f'Sim {s}'
        # Run
        if self.already_run:
            errormsg = 'Cannot re-run an already run MultiSim'
            raise RuntimeError(errormsg)
        self.sims = multi_run(self.sims, **kwargs)

        # Recompute stats
        if compute_stats:
            self.compute_stats()
        self.already_run = True
        return self

    def compute_stats(self, return_raw=False, quantiles=None, use_mean=False, bounds=None):
        """ Compute statistics across multiple sims """

        if use_mean:
            if bounds is None:
                bounds = 1
        else:
            if quantiles is None:
                quantiles = {'low': 0.1, 'high': 0.9}
            if not isinstance(quantiles, dict):
                try:
                    quantiles = {'low': float(quantiles[0]), 'high': float(quantiles[1])}
                except Exception as E:
                    errormsg = f'Could not figure out how to convert {quantiles} into a quantiles object: must be a dict with keys low, high or a 2-element array ({str(E)})'
                    raise ValueError(errormsg)

        base_sim = sc.dcp(self.sims[0])
        raw = sc.objdict()
        results = sc.objdict()
        axis = 1
        start_end = np.array([sim.tvec[[0, -1]] for sim in self.sims])
        if len(np.unique(start_end)) != 2:
            errormsg = f'Cannot compute stats for sims: start and end values do not match:\n{start_end}'
            raise ValueError(errormsg)

        reskeys = list(base_sim.results.keys())

        bad_keys = ['t', 'tfr_years', 'method_usage']
        for key in bad_keys:  # Don't compute high/low for these
            results[key] = base_sim.results[key]
            reskeys.remove(key)
        for reskey in reskeys:
            if isinstance(base_sim.results[reskey], dict):
                if return_raw:
                    for s, sim in enumerate(self.sims):
                        raw[reskey][s] = base_sim.results[reskey]
            else:
                results[reskey] = sc.objdict()
                npts = len(base_sim.results[reskey])
                raw[reskey] = np.zeros((npts, len(self.sims)))
                for s, sim in enumerate(self.sims):
                    raw[reskey][:, s] = sim.results[reskey]  # Stack into an array for processing

                if use_mean:
                    r_mean = np.mean(raw[reskey], axis=axis)
                    r_std = np.std(raw[reskey], axis=axis)
                    results[reskey].best = r_mean
                    results[reskey].low = r_mean - bounds * r_std
                    results[reskey].high = r_mean + bounds * r_std
                else:
                    results[reskey].best = np.quantile(raw[reskey], q=0.5, axis=axis)
                    results[reskey].low = np.quantile(raw[reskey], q=quantiles['low'], axis=axis)
                    results[reskey].high = np.quantile(raw[reskey], q=quantiles['high'], axis=axis)

        self.results = results
        self.base_sim.results = results  # Store here too, to enable plotting

        if return_raw:
            return raw
        else:
            return

    @staticmethod
    def merge(*args, base=False):
        """
        Convenience method for merging two MultiSim objects.

        Args:
            args (MultiSim): the MultiSims to merge (either a list, or separate)
            base (bool): if True, make a new list of sims from the multisim's two base sims; otherwise, merge the multisim's lists of sims

        Returns:
            msim (MultiSim): a new MultiSim object

        **Examples**::

            mm1 = fp.MultiSim.merge(msim1, msim2, base=True)
            mm2 = fp.MultiSim.merge([m1, m2, m3, m4], base=False)
        """

        # Handle arguments
        if len(args) == 1 and isinstance(args[0], list):
            args = args[0]  # A single list of MultiSims has been provided

        # Create the multisim from the base sim of the first argument
        msim = MultiSim(base_sim=sc.dcp(args[0].base_sim), sims=[], label=args[0].label)
        msim.sims = []
        msim.chunks = []  # This is used to enable automatic splitting later

        # Handle different options for combining
        if base:  # Only keep the base sims
            for i, ms in enumerate(args):
                sim = sc.dcp(ms.base_sim)
                sim.label = ms.label
                msim.sims.append(sim)
                msim.chunks.append([[i]])
        else:  # Keep all the sims
            for ms in args:
                len_before = len(msim.sims)
                msim.sims += list(sc.dcp(ms.sims))
                len_after = len(msim.sims)
                msim.chunks.append(list(range(len_before, len_after)))

        return msim

    def split(self, inds=None, chunks=None):
        """
        Convenience method for splitting one MultiSim into several. You can specify
        either individual indices of simulations to extract, via inds, or consecutive
        chunks of indices, via chunks. If this function is called on a merged MultiSim,
        the chunks can be retrieved automatically and no arguments are necessary.

        Args:
            inds (list): a list of lists of indices, with each list turned into a MultiSim
            chunks (int or list): if an int, split the MultiSim into that many chunks; if a list return chunks of that many sims

        Returns:
            A list of MultiSim objects

        **Examples**::

            m1 = fp.MultiSim(fp.Sim(label='sim1'))
            m2 = fp.MultiSim(fp.Sim(label='sim2'))
            m3 = fp.MultiSim.merge(m1, m2)
            m3.run()
            m1b, m2b = m3.split()

            msim = fp.MultiSim(fp.Sim(), n_runs=6)
            msim.run()
            m1, m2 = msim.split(inds=[[0,2,4], [1,3,5]])
            mlist1 = msim.split(chunks=[2,4]) # Equivalent to inds=[[0,1], [2,3,4,5]]
            mlist2 = msim.split(chunks=2) # Equivalent to inds=[[0,1,2], [3,4,5]]
        """

        # Process indices and chunks
        if inds is None:  # Indices not supplied
            if chunks is None:  # Chunks not supplied
                if hasattr(self, 'chunks'):  # Created from a merged MultiSim
                    inds = self.chunks
                else:  # No indices or chunks and not created from a merge
                    errormsg = 'If a MultiSim has not been created via merge(), you must supply either inds or chunks to split it'
                    raise ValueError(errormsg)
            else:  # Chunks supplied, but not inds
                inds = []  # Initialize
                sim_inds = np.arange(len(self))  # Indices for the simulations
                if sc.isiterable(chunks):  # e.g. chunks = [2,4]
                    chunk_inds = np.cumsum(chunks)[:-1]
                    inds = np.split(sim_inds, chunk_inds)
                else:  # e.g. chunks = 3
                    inds = np.split(sim_inds, chunks)  # This will fail if the length is wrong

        # Do the conversion
        mlist = []
        for indlist in inds:
            sims = sc.dcp([self.sims[i] for i in indlist])
            msim = MultiSim(sims=sims)
            mlist.append(msim)

        return mlist

    def remerge(self, base=True, recompute=True, **kwargs):
        """
        Split a sim, compute stats, and re-merge.

        Args:
            base (bool): whether to use the base sim (otherwise, has no effect)
            kwargs (dict): passed to msim.split()
            recompute (bool): whether to run compute_statson each sim

        Note: returns a new MultiSim object (if that concerns you).
        """
        ms = self.split(**kwargs)
        if recompute:
            for m in ms:
                m.compute_stats()  # Recompute the statistics on each separate MultiSim
        out = MultiSim.merge(*ms, base=base)  # Now re-merge, this time using the base_sim
        return out

    def to_df(self, yearly=False, mean=False):
        """
        Export all individual sim results to a dataframe
        """
        if mean:
            df = self.base_sim.to_df()
        else:
            raw_res = sc.odict(defaultdict=list)
            for s, sim in enumerate(self.sims):
                for reskey in sim.results.keys():
                    res = sim.results[reskey]
                    if sc.isarray(res):
                        if len(res) == sim.npts and not yearly:
                            raw_res[reskey] += res.tolist()
                        elif len(res) == len(sim.results['tfr_years']) and yearly:
                            raw_res[reskey] += res.tolist()

                scale = len(sim.results['tfr_years']) if yearly else sim.npts
                raw_res['sim'] += [s] * scale
                raw_res['sim_label'] += [sim.label] * scale

            df = pd.DataFrame(raw_res)
            self.df = df
        return df

    def plot(self, to_plot=None, plot_sims=True, do_save=None, filename='fp_multisim.png',
             fig_args=None, axis_args=None, plot_args=None, style=None, colors=None, **kwargs):
        """
        Plot the MultiSim

        Args:
            plot_sims (bool): whether to plot individual sims (else, plot with uncertainty bands)

        See ``sim.plot()`` for additional args.
        """
        fig_args = sc.mergedicts(dict(figsize=(16, 10)), fig_args)

        fig = pl.figure(**fig_args)
        do_show = kwargs.pop('do_show', True)
        labels = sc.autolist()
        labellist = sc.autolist()  # TODO: shouldn't need this
        for sim in self.sims:  # Loop over and find unique labels
            if sim.label not in labels:
                labels += sim.label
                labellist += sim.label
                label = sim.label
            else:
                labellist += ''
        n_unique = len(np.unique(labels))  # How many unique sims there are

        def get_scale_ceil(channel):
            is_dist = hasattr(self.sims[0].results['acpr'], 'best')  # picking a random channel
            if is_dist:
                maximum_value = max([max(sim.results[channel].high) for sim in self.sims])
            else:
                maximum_value = max([max(sim.results[channel]) for sim in self.sims])
            top = int(np.ceil(maximum_value / 10.0)) * 10  # rounding up to nearest 10
            return top

        if to_plot == 'method':
            axis_args_method = sc.mergedicts(dict(left=0.1, bottom=0.05, right=0.9, top=0.97, wspace=0.2, hspace=0.30),
                                             axis_args)
            with fpo.with_style(style):
                pl.subplots_adjust(**axis_args_method)
                for axis_index, label in enumerate(np.unique(labels)):
                    total_df = pd.DataFrame()
                    return_default = lambda name: fig_args[name] if name in fig_args else None
                    rows, cols = sc.getrowscols(n_unique, nrows=return_default('nrows'), ncols=return_default('ncols'))
                    ax = pl.subplot(rows, cols, axis_index + 1)
                    for sim in self.sims:
                        if sim.label == label:
                            total_df = pd.concat([total_df, sim.format_method_df(timeseries=True)], ignore_index=True)
                    method_names = total_df['Method'].unique()

                    # Getting the mean of each seed as a list of lists, could add conditional here if different method plots are added
                    percentage_by_method = []
                    for method in method_names:
                        method_df = total_df[(total_df['Method'] == method) & (total_df['Sim'] == label)]
                        seed_split = [method_df[method_df['Seed'] == seed]['Percentage'].values for seed in
                                      method_df['Seed'].unique()]
                        percentage_by_method.append(
                            [np.mean([seed[i] for seed in seed_split]) for i in range(len(seed_split[0]))])

                    legend = axis_index + 1 == cols  # True for last plot in first row
                    colors = [colors[method] for method in method_names] if isinstance(colors, dict) else colors
                    ax.stackplot(total_df["Year"].unique(), percentage_by_method, labels=method_names, colors=colors)
                    ax.set_title(label.capitalize())
                    ax.legend().set_visible(legend)
                    ax.set_xlabel('Year')
                    ax.set_ylabel('Percentage')
                    if legend:
                        ax.legend(loc='lower left', bbox_to_anchor=(1, -0.05), frameon=True) if len(
                            labels) > 1 else ax.legend(loc='upper left', frameon=True)
                    pl.ylim(0, max(
                        max([sum(proportion[1:] * 100) for proportion in results['method_usage']]) for results in
                        [sim.results for sim in self.sims]) + 1)
                return tidy_up(fig=fig, do_show=do_show, do_save=do_save, filename=filename)
        elif plot_sims:
            colors = sc.gridcolors(n_unique)
            colors = {k: c for k, c in zip(labels, colors)}
            for s, sim in enumerate(self.sims):  # Note: produces duplicate legend entries
                label = labellist[s]
                color = colors[sim.label]
                alpha = max(0.2, 1 / np.sqrt(n_unique))
                sim_plot_args = sc.mergedicts(dict(alpha=alpha, c=color), plot_args)
                kw = dict(new_fig=False, do_show=False, label=label, plot_args=sim_plot_args)
                sim.plot(to_plot=to_plot, **kw, **kwargs)

            if to_plot is not None:
                # Scale axes
                if to_plot == 'cpr':
                    fig = self.base_sim.conform_y_axes(figure=fig, top=get_scale_ceil('acpr'))
                if 'as_' in to_plot:
                    channel_type = to_plot.split("_")[1]
                    is_tfr = "tfr" in to_plot
                    age_bins = list(fpd.age_specific_channel_bins)[:-1]
                    if is_tfr:
                        age_bins = fpd.age_bin_map
                    if hasattr(sim.results[f'cpr_{list(fpd.age_specific_channel_bins.keys())[0]}'],
                               'best'):  # if compute_stats has been applied
                        top = max([max([max(group_result) for group_result in
                                        [sim.results[f'{channel_type}_{age_group}'].high for age_group in age_bins]])
                                   for sim in self.sims])
                    else:
                        top = max([max([max(group_result) for group_result in
                                        [sim.results[f'{channel_type}_{age_group}'] for age_group in age_bins]]) for sim
                                   in self.sims])
                    tidy_top = int(np.ceil(top / 10.0)) * 10  # rounds top of y axis up to the nearest ten
                    tidy_top = tidy_top + 20 if is_tfr or 'imr' in to_plot else tidy_top  # some custom axis adjustments for neatness
                    tidy_top = tidy_top + 50 if 'mmr' in to_plot else tidy_top
                    self.base_sim.conform_y_axes(figure=fig, top=tidy_top)
            return tidy_up(fig=fig, do_show=do_show, do_save=do_save, filename=filename)
        else:
            return self.base_sim.plot(to_plot=to_plot, do_show=do_show, fig_args=fig_args, plot_args=plot_args,
                                      **kwargs)

    def plot_age_first_birth(self, do_show=False, do_save=True, output_file='age_first_birth_multi.png'):
        length = sum([len([num for num in sim.people.first_birth_age if num is not None]) for sim in self.sims])
        data_dict = {"age": [0] * length, "sim": [0] * length}
        i = 0
        for sim in self.sims:
            for value in [num for num in sim.people.first_birth_age if num is not None]:
                data_dict['age'][i] = value
                data_dict['sim'][i] = sim.label
                i = i + 1

        data = pd.DataFrame(data_dict)
        pl.title("Age at first birth")
        sns.boxplot(data=data, y='age', x='sim', orient='v', notch=True)
        if do_show:
            pl.show()
        if do_save:
            print(f"Saved age at first birth plot at {output_file}")
            pl.savefig(output_file)


def single_run(sim):
    """ Helper function for multi_run(); rarely used on its own """
    sim.run()
    return sim


def multi_run(sims, **kwargs):
    """ Run multiple sims in parallel; usually used via the MultiSim class, not directly """
    sims = sc.parallelize(single_run, iterarg=sims, **kwargs)
    return sims


def parallel(*args, **kwargs):
    """
    A shortcut to ``fp.MultiSim()``, allowing the quick running of multiple simulations
    at once.

    Args:
        args (list): The simulations to run
        kwargs (dict): passed to multi_run()

    Returns:
        A run MultiSim object.

    **Examples**::

        s1 = fp.Sim(exposure_factor=0.5, label='Low')
        s2 = fp.Sim(exposure_factor=2.0, label='High')
        fp.parallel(s1, s2).plot()
        msim = fp.parallel(s1, s2)
    """
    sims = sc.mergelists(*args)
    return MultiSim(sims=sims).run(**kwargs)
