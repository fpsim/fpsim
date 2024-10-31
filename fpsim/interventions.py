'''
Specify the core interventions available in FPsim. Other interventions can be
defined by the user by inheriting from these classes.
'''
import numpy as np
import pylab as pl
import sciris as sc
import inspect
from . import utils as fpu
from . import methods as fpm
from . import defaults as fpd

#%% Generic intervention classes

__all__ = ['Intervention', 'change_par', 'update_methods', 'change_people_state', 'change_initiation_prob', 'change_initiation']


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
        self.years = [] # The start and end years of the intervention
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

            if hasattr(self, 'plot_years'):
                years = self.plot_years
            elif not self.years and hasattr(self, 'year'):
                years = sc.toarray(self.year)
            else:
                years = self.years

            if sc.isiterable(years):
                label_shown = False # Don't show the label more than once
                for y in years:
                    if sc.isnumber(y):
                        if self.show_label and not label_shown: # Choose whether to include the label in the legend
                            label = self.label
                            label_shown = True
                        else:
                            label = None
                        ax.axvline(y, label=label, **line_args)
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


class change_par(Intervention):
    '''
    Change a parameter at a specified point in time.

    Args:
        par   (str): the parameter to change
        years (float/arr): the year(s) at which to apply the change
        vals  (any): a value or list of values to change to (if a list, must have the same length as years); or a dict of year:value entries

    If any value is ``'reset'``, reset to the original value.

    **Example**::

        ec0 = fp.change_par(par='exposure_factor', years=[2000, 2010], vals=[0.0, 2.0]) # Reduce exposure factor
        ec0 = fp.change_par(par='exposure_factor', vals={2000:0.0, 2010:2.0}) # Equivalent way of writing
        sim = fp.Sim(interventions=ec0).run()
    '''
    def __init__(self, par, years=None, vals=None, verbose=False):
        super().__init__()
        self.par   = par
        self.verbose = verbose
        if isinstance(years, dict): # Swap if years is supplied as a dict, so can be supplied first
            vals = years
        if vals is None:
            errormsg = 'Values must be supplied'
            raise ValueError(errormsg)
        if isinstance(vals, dict):
            years = sc.dcp(list(vals.keys()))
            vals = sc.dcp(list(vals.values()))
        else:
            if years is None:
                errormsg = 'If vals is not supplied as a dict, then year(s) must be supplied'
                raise ValueError(errormsg)
            else:
                years = sc.toarray(sc.dcp(years))
                vals = sc.dcp(vals)
                if sc.isnumber(vals):
                    vals = sc.tolist(vals) # We want to be careful not to take something that might already be an array and interpret different values as years
                n_years = len(years)
                n_vals = len(vals)
                if n_years != n_vals:
                    errormsg = f'Number of years ({n_years}) does not match number of values ({n_vals})'
                    raise ValueError(errormsg)

        self.years = years
        self.vals = vals

        return


    def initialize(self, sim):
        super().initialize()

        # Validate parameter name
        if self.par not in sim.pars:
            errormsg = f'Parameter "{self.par}" is not a valid sim parameter'
            raise ValueError(errormsg)

        # Validate years and values
        years = self.years
        min_year = min(years)
        max_year = max(years)
        if min_year < sim['start_year']:
            errormsg = f'Intervention start {min_year} is before the start of the simulation'
            raise ValueError(errormsg)
        if max_year > sim['end_year']:
            errormsg = f'Intervention end {max_year} is after the end of the simulation'
            raise ValueError(errormsg)
        if years != sorted(years):
            errormsg = f'Years {years} should be monotonic increasing'
            raise ValueError(errormsg)

        # Convert intervention years to sim timesteps
        self.counter = 0
        self.inds = sc.autolist()
        for y in years:
            self.inds += sc.findnearest(sim.tvec, y)

        # Store original value
        self.orig_val = sc.dcp(sim[self.par])

        return


    def apply(self, sim):
        if len(self.inds) > self.counter:
            ind = self.inds[self.counter] # Find the current index
            if sim.ti == ind: # Check if the current timestep matches
                curr_val = sc.dcp(sim[self.par])
                val = self.vals[self.counter]
                if val == 'reset':
                    val = self.orig_val
                sim[self.par] = val # Update the parameter value -- that's it!
                if self.verbose:
                    label = f'Sim "{sim.label}": ' if sim.label else ''
                    print(f'{label}On {sim.y}, change {self.counter+1}/{len(self.inds)} applied: "{self.par}" from {curr_val} to {sim[self.par]}')
                self.counter += 1
        return


    def finalize(self, sim=None):
        # Check that all changes were applied
        n_counter = self.counter
        n_vals = len(self.vals)
        if n_counter != n_vals:
            errormsg = f'Not all values were applied ({n_vals} ≠ {n_counter})'
            raise RuntimeError(errormsg)
        return


class change_people_state(Intervention):
    """
    Intervention to modify values of a People's boolean state at one specific
    point in time.

    Args:
        state_name   (string): name of the People's state that will be modified
        year         (float): time expressed in years when the change is applied
        new_val      (float): the new state value eligible people will have
        eligibility  (inds/callable): indices OR callable that returns inds
    """

    def __init__(self, state_name, year, new_val, eligibility=None, prop=1.0):
        super().__init__()
        self.state_name = state_name
        self.year = year
        self.new_val = new_val
        self.eligibility = eligibility
        self.prop = prop
        self.applied = False
        return

    def initialize(self, sim=None):
        super().initialize()
        self._validate()
        return

    def _validate(self):
        # Validation
        if self.state_name is None:
            errormsg = 'A state name must be supplied.'
            raise ValueError(errormsg)
        if self.year is None:
            errormsg = 'A year must be supplied.'
            raise ValueError(errormsg)
        if self.new_val is None:
            errormsg = 'A new value must be supplied.'
            raise ValueError(errormsg)
        if self.eligibility is None:
            errormsg = 'Eligibility needs to be provided'
            raise ValueError(errormsg)
        return

    def check_eligibility(self, sim):
        """
        Return an array of indices of agents eligible
        """
        if callable(self.eligibility):
            is_eligible = self.eligibility(sim)
        elif sc.isarray(self.eligibility):
            eligible_inds = self.eligibility
            is_eligible = np.zeros(len(sim.people), dtype=bool)
            is_eligible[eligible_inds] = True
        else:
            errormsg = 'Eligibility must be a function or an array of indices'
            raise ValueError(errormsg)

        eligible_inds = sc.findinds(is_eligible)
        is_selected = fpu.n_binomial(self.prop, len(eligible_inds))
        is_eligible[eligible_inds] = is_selected

        return is_eligible

    def apply(self, sim):
        if not self.applied and sim.y >= self.year:
            self.applied = True  # Ensure we don't apply this more than once
            is_eligible = self.check_eligibility(sim)
            ppl = sim.people.filter(is_eligible)
            setattr(ppl, self.state_name, self.new_val)
        return


class update_methods(Intervention):
    """
    Intervention to modify method efficacy and/or switching matrix.

    Args:
        year (float): The year we want to change the method.

        eff (dict):
            An optional key for changing efficacy; its value is a dictionary with the following schema:
                {method: efficacy}
                    Where method is the name of the contraceptive method to be changed,
                    and efficacy is a number with the efficacy

        dur_use (dict):
            Optional key for changing the duration of use; its value is a dictionary with the following schema:
                {method: dur_use}
                    Where method is the method to be changed, and dur_use is a dict representing a distribution, e.g.
                    dur_use = {'Injectables: dict(dist='lognormal', par1=a, par2=b)}

        p_use (float): probability of using any form of contraception
        method_mix (list/arr): probabilities of selecting each form of contraception

    """

    def __init__(self, year, eff=None, dur_use=None, p_use=None, method_mix=None, verbose=False):
        super().__init__()
        self.year    = year
        self.eff     = eff
        self.dur_use = dur_use
        self.p_use = p_use
        self.method_mix = method_mix
        self.verbose = verbose
        self.applied = False
        return

    def initialize(self, sim=None):
        super().initialize()
        self._validate()
        par_name = None
        if self.p_use is not None and isinstance(sim.people.contraception_module, (fpm.SimpleChoice, fpm.EmpoweredChoice)):
            par_name = 'p_use'
        if self.method_mix is not None and isinstance(sim.people.contraception_module, (fpm.SimpleChoice, fpm.EmpoweredChoice)):
            par_name = 'method_mix'

        if par_name is not None:
            errormsg = (
                f"Contraceptive module  {type(sim.people.contraception_module)} does not have `{par_name}` parameter. "
                f"For this type of module, the probability of contraceptive use depends on people attributes and can't be reset using this intervention.")
            raise ValueError(errormsg)

        return

    def _validate(self):
        # Validation
        if self.year is None:
            errormsg = 'A year must be supplied'
            raise ValueError(errormsg)
        if self.eff is None and self.dur_use is None and self.p_use is None and self.method_mix is None:
            errormsg = 'Either efficacy, durations of use, probability of use, or method mix must be supplied'
            raise ValueError(errormsg)
        return

    def apply(self, sim):
        """
        Applies the efficacy or contraceptive uptake changes if it is the specified year
        based on scenario specifications.
        """

        if not self.applied and sim.y >= self.year:
            self.applied = True # Ensure we don't apply this more than once

            # Implement efficacy
            if self.eff is not None:
                for k, rawval in self.eff.items():
                    sim.contraception_module.update_efficacy(method_label=k, new_efficacy=rawval)

            # Implement changes in duration of use
            if self.dur_use is not None:
                for k, rawval in self.dur_use.items():
                    sim.contraception_module.update_duration(method_label=k, new_duration=rawval)

            # Change in probability of use
            if self.p_use is not None:
                sim.people.contraception_module.pars['p_use'] = self.p_use

            # Change in method mix
            if self.method_mix is not None:
                this_mix = self.method_mix / np.sum(self.method_mix) # Renormalise in case they are not adding up to 1
                sim.people.contraception_module.pars['method_mix'] = this_mix

        return


class change_initiation_prob(Intervention):
    """
    Intervention to change the probabilty of contraception use trend parameter in
    contraceptive choice modules that have a logistic regression model.

    Args:
        year (float): The year we want to start the intervention.
        prob_use_intercept (float): A number that changes the intercept in the logistic refgression model
        p_use = 1 / (1 + np.exp(-rhs + p_use_time_trend + p_use_intercept))
    """

    def __init__(self, year, prob_use_intercept=0.0, verbose=False):
        super().__init__()
        self.year    = year
        self.prob_use_intercept = prob_use_intercept
        self.verbose = verbose
        self.applied = False
        self.par_name = None
        return

    def initialize(self, sim=None):
        super().initialize()
        self._validate()
        if isinstance(sim.people.contraception_module, (fpm.SimpleChoice)):
            self.par_name = 'prob_use_intercept'

        if self.par_name is None:
            errormsg = (
                f"Contraceptive module  {type(sim.people.contraception_module)} does not have `{self.par_name}` parameter.")
            raise ValueError(errormsg)

        return

    def _validate(self):
        # Basic Validation
        if self.year is None:
            errormsg = 'A year must be supplied'
            raise ValueError(errormsg)
        return

    def apply(self, sim):
        """
        Applies the changes to efficacy or contraceptive uptake changes if it is the specified year
        based on scenario specifications.
        """

        if not self.applied and sim.y >= self.year:
            self.applied = True # Ensure we don't apply this more than once
            sim.people.contraception_module.pars[self.par_name] = self.prob_use_intercept

        return


class change_initiation(Intervention):
    """
    Intervention that modifies the outcomes of whether women are on contraception or not
    Select a proportion of women and sets them on a contraception method.

    Args:
        year (float): time expressed in years when the intervention starts is applied
        annual_increase (float): a value between 0 and 1 indicating the x% extra of women
            who will be made to select a contraception method.
            The proportion or % is with respect to the number of
            women who were on contraception the previous year.
        eligibility (callable): callable that returns a filtered version of
            people eligible to receive the intervention
    """

    def __init__(self, year=None, annual_increase=0.0, eligibility=None):
        super().__init__()
        self.year = year
        self.eligibility = eligibility
        self.increase = annual_increase / fpd.mpy
        self.dt_increase = None
        self.applied = False
        return

    def initialize(self, sim=None):
        super().initialize()
        self._validate()
        self.dt_increase = 1.0 + self.increase  # if we do the updates at every time point
        return

    def _validate(self):
        # Validation
        if self.year is None:
            errormsg = 'A year must be supplied.'
            raise ValueError(errormsg)
        return

    def check_eligibility(self, sim):
        """
        Select eligible who is eligible
        """
        if self.eligibility is None:
            contra_choosers = self._default_contra_choosers(sim.people)
        return contra_choosers

    def _default_contra_choosers(self, ppl):
        # TODO: check this is ok, or make a filter about the largest group of women who are eligible to choose contraception
        # TODO: do we care whether women people have ti_contra > 0? For instance postpartum women could be made to choose earlier?
        eligible = ppl.filter((ppl.sex == 0) & (ppl.alive) &                 # living women
                              (ppl.age < ppl.pars['age_limit_fecundity']) &  # who are fecund
                              (ppl.sexual_debut) &                           # who already had their sexual debut
                              (~ppl.pregnant)  &                             # who are not currently pregnant
                              (~ppl.postpartum) &                            # who are not in postpartum
                              (ppl.sexually_active) &                        # who are sexually active on this time step or doesn't matter???
                              (~ppl.on_contra)                               # who are not already on contra
                              )
        return eligible

    def apply(self, sim):
        ti = sim.ti
        if sim.y >= self.year:
            contra_choosers = self.check_eligibility(sim)
            n_choosers = len(contra_choosers)
            # Number currently on contra
            current_oncontra = sum(sim.people.on_contra) # PSL: not used at the moment, not sure if we should be using this value instead of past_oncontra
            #PSL: Number on contra one year ago.
            past_oncontra = sum(sim.people['longitude']['on_contra'][:, sim.people.yei])
            oncontra = past_oncontra
            # avoid overflow and selecting more agents than we have available. feel there should be a better way of doing this
            increase_factor = np.minimum(self.dt_increase, 1e9)
            new_oncontra = sc.randround(oncontra * increase_factor - oncontra)
            new_oncontra = np.minimum(new_oncontra, n_choosers)
            if new_oncontra:
                p_select = new_oncontra * np.ones(n_choosers) / n_choosers
                contra_choosers.on_contra = fpu.binomial_arr(p_select)
                new_users = contra_choosers.filter(contra_choosers.on_contra)
                new_users.method = sim.people.contraception_module.init_method_dist(new_users)
                new_users.ever_used_contra = 1
                method_dur = sim.people.contraception_module.set_dur_method(new_users)
                new_users.ti_contra = ti + method_dur
            self.dt_increase += self.increase # Not sure this is exactly what was required
        return