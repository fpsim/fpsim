"""
Contraceptive methods

Idea:
    A method selector should be a generic class (an intervention?). We want the existing matrix-based
    method and the new duration-based method to both be instances of this.

"""

# %% Imports
import numpy as np
import sciris as sc
import pandas as pd
import scipy.stats as sps  # For sampling durations
import starsim as ss  # TODO add to dependencies
from . import utils as fpu
from . import defaults as fpd
from . import locations as fplocs

__all__ = ['Method', 'Methods', 'ContraceptiveChoice', 'RandomChoice', 'SimpleChoice', 'EmpoweredChoice']


# %% Base definition of contraceptive methods -- can be overwritten by locations
class Method:
    def __init__(self, name=None, label=None, idx=None, efficacy=None, modern=None, dur_use=None, csv_name=None):
        self.name = name
        self.label = label or name
        self.csv_name = csv_name or label or name
        self.idx = idx
        self.efficacy = efficacy
        self.modern = modern
        self.dur_use = dur_use


# Helper function for setting lognormals
def ln(a, b): return dict(dist='lognormal', par1=a, par2=b)


method_list = [
    Method(name='none',     efficacy=0,     modern=False, dur_use=ln(2, 3), label='None'),
    Method(name='pill',     efficacy=0.945, modern=True,  dur_use=ln(2, 3), label='Pill'),
    Method(name='iud',      efficacy=0.986, modern=True, dur_use=ln(5, 3), label='IUDs', csv_name='IUD'),
    Method(name='inj',      efficacy=0.983, modern=True, dur_use=ln(2, 3), label='Injectables', csv_name='Injectable'),
    Method(name='cond',     efficacy=0.946, modern=True,  dur_use=ln(1, 3), label='Condoms', csv_name='Condom'),
    Method(name='btl',      efficacy=0.995, modern=True, dur_use=ln(50, 3), label='BTL', csv_name='F.sterilization'),
    Method(name='wdraw',    efficacy=0.866, modern=False, dur_use=ln(1, 3), label='Withdrawal', csv_name='Withdrawal'), #     # 1/2 periodic abstinence, 1/2 other traditional approx.  Using rate from periodic abstinence
    Method(name='impl',     efficacy=0.994, modern=True, dur_use=ln(2, 3), label='Implants', csv_name='Implant'),
    Method(name='othtrad',  efficacy=0.861, modern=False, dur_use=ln(1, 3), label='Other traditional', csv_name='Other.trad'),
    Method(name='othmod',   efficacy=0.880, modern=True, dur_use=ln(1, 3), label='Other modern', csv_name='Other.mod'),
]

idx = 0
for method in method_list:
    method.idx = idx
    idx += 1

method_map = {method.label: method.idx for method in method_list}
Methods = ss.ndict(method_list, type=Method)
SimpleMethods = sc.dcp(Methods)
for m in SimpleMethods.values(): m.dur_use = 1


# %% Define classes to contain information about the way women choose contraception

class ContraceptiveChoice:
    def __init__(self, methods=None, pars=None, **kwargs):
        self.methods = methods or SimpleMethods
        self.__dict__.update(kwargs)
        self.n_options = len(self.methods)
        self.n_methods = len([m for m in self.methods if m != 'none'])
        self.init_dist = None
        default_pars = dict(
            p_use=0.5,
            force_choose=False,  # Whether to force non-users to choose a method
        )
        self.pars = sc.mergedicts(default_pars, pars)

    @property
    def average_dur_use(self):
        av = 0
        for m in self.methods.values():
            if sc.isnumber(m.dur_use): av += m.dur_use
            elif isinstance(m.dur_use, dict): av += m.dur_use['par1']
        return av / len(self.methods)

    def init_method_dist(self, ppl):
        pass

    def get_prob_use(self, ppl, year=None, event=None, ti=None, tiperyear=None):
        """ Assign probabilities that each woman will use contraception """
        prob_use = np.full(len(ppl), fill_value=self.pars['p_use'], dtype=float)
        return prob_use

    def get_method_by_label(self, method_label):
        """ Extract method according to its label / long name """
        return_val = None
        for method_name, method in self.methods.items():
            if method.label == method_label:
                return_val = method
        if return_val is None:
            errormsg = f'No method matching {method_label} found.'
            raise ValueError(errormsg)
        return return_val

    def update_efficacy(self, method_label=None, new_efficacy=None):
        method = self.get_method_by_label(method_label)
        method.efficacy = new_efficacy

    def update_duration(self, method_label=None, new_duration=None):
        method = self.get_method_by_label(method_label)
        method.dur_use = new_duration

    def add_method(self, method):
        self.methods[method.name] = method

    def remove_method(self, method_label):
        method = self.get_method_by_label(method_label)
        del self.methods[method.name]

    def get_contra_users(self, ppl, year=None, event=None, ti=None, tiperyear=None):
        """ Select contraception users, return boolean array """
        prob_use = self.get_prob_use(ppl, event=event, year=year, ti=ti, tiperyear=tiperyear)
        uses_contra_bool = fpu.binomial_arr(prob_use)
        return uses_contra_bool

    def choose_method(self, ppl, event=None):
        pass

    def set_dur_method(self, ppl, method_used=None):
        dt = ppl.pars['timestep'] / fpd.mpy
        timesteps_til_update = np.full(len(ppl), np.round(self.average_dur_use/dt), dtype=int)
        return timesteps_til_update


class RandomChoice(ContraceptiveChoice):
    """ Randomly choose a method of contraception """
    def __init__(self, pars=None, methods=None, **kwargs):
        super().__init__(methods=methods, **kwargs)
        default_pars = dict(
            p_use=0.5,
            method_mix=np.array([1/self.n_methods]*self.n_methods),
            force_choose=False,  # Whether to force non-users to choose a method
        )
        self.pars = sc.mergedicts(default_pars, pars)
        self.init_dist = self.pars['method_mix']
        return

    def init_method_dist(self, ppl):
        return self.choose_method(ppl)

    def choose_method(self, ppl, event=None):
        choice_arr = np.random.choice(np.arange(1, self.n_methods+1), size=len(ppl), p=self.pars['method_mix'])
        return choice_arr.astype(int)


class SimpleChoice(RandomChoice):
    def __init__(self, pars=None, location=None, **kwargs):
        """ Args: coefficients """
        super().__init__(**kwargs)
        default_pars = dict(
            prob_use_year=2000,
            prob_use_trend_par=0.1,
            force_choose=False,  # Whether to force non-users to choose a method
            method_weights=np.ones(self.n_methods),
        )
        updated_pars = sc.mergedicts(default_pars, pars)
        self.pars = sc.mergedicts(self.pars, updated_pars)

        # Handle location
        location = location.lower()
        if location == 'kenya':
            self.contra_use_pars = fplocs.kenya.process_contra_use_simple()  # Set probability of use
            method_choice_pars, init_dist = fplocs.kenya.process_markovian_method_choice(self.methods)  # Method choice
            self.method_choice_pars = method_choice_pars
            self.init_dist = init_dist
            self.methods = fplocs.kenya.process_dur_use(self.methods)  # Reset duration of use

            # Handle age bins -- find a more robust way to do this
            self.age_bins = np.sort([fpd.method_age_map[k][1] for k in self.method_choice_pars[0].keys() if k != 'method_idx'])

        else:
            errormsg = f'Location "{location}" is not currently supported for method-time analyses'
            raise NotImplementedError(errormsg)
        return

    def init_method_dist(self, ppl):
        if self.init_dist is not None:
            choice_array = np.zeros(len(ppl))

            # Loop over age groups and methods
            for key, (age_low, age_high) in fpd.method_age_map.items():
                this_age_bools = fpu.match_ages(ppl.age, age_low, age_high)
                ppl_this_age = this_age_bools.nonzero()[-1]
                if len(ppl_this_age) > 0:
                    these_probs = self.init_dist[key]
                    these_probs = np.array(these_probs) * self.pars['method_weights']  # Scale by weights
                    these_probs = these_probs/np.sum(these_probs)  # Renormalize
                    these_choices = fpu.n_multinomial(these_probs, len(ppl_this_age))  # Choose
                    # Adjust method indexing to correspond to datafile (removing None: Marita to confirm)
                    choice_array[this_age_bools] = np.array(list(self.init_dist.method_idx))[these_choices]
            return choice_array.astype(int)
        else:
            errormsg = f'Distribution of contraceptive choices has not been provided.'
            raise ValueError(errormsg)

    def get_prob_use(self, ppl, year=None, event=None, ti=None, tiperyear=None):
        """
        Return an array of probabilities that each woman will use contraception.
        """
        # Figure out which coefficients to use
        if event is None : p = self.contra_use_pars[0]
        if event == 'pp1': p = self.contra_use_pars[1]
        if event == 'pp6': p = self.contra_use_pars[2]

        # Calculate probability of use
        rhs = np.full_like(ppl.age, fill_value=p.intercept)
        age_bins = np.digitize(ppl.age, self.age_bins)
        for ai, ab in enumerate(self.age_bins):
            rhs[age_bins == ai] += p.age_factors[ai]
            if ai > 1:
                rhs[(age_bins == ai) & ppl.ever_used_contra] += p.age_ever_user_factors[ai-1]
        rhs[ppl.ever_used_contra] += p.fp_ever_user
        rhs += (year - self.pars['prob_use_year'])*self.pars['prob_use_trend_par']
        prob_use = 1 / (1+np.exp(-rhs))
        return prob_use

    @staticmethod
    def _lognormal_dpars(dur_use, ai):
        par1 = dur_use['par1'] + dur_use['age_factors'][ai]
        par2 = np.exp(dur_use['par2'])
        return par1, par2

    @staticmethod
    def _lognormal_make_dict(dur_use, par1, par2):
        """
        Map par1 and par2 from duration use, to par1 and par2 needed
        by fpu.sample
        """
        # NOTE:TODO: fix me, after lognorm distribution has been consolidated in utils.py
        return dict(dist='lognorm_sps', par1=par2, par2=np.exp(par1))

    @staticmethod
    def _llogis_dpars(dur_use, ai):
        par1 = np.exp(dur_use['par1'] + dur_use['age_factors'][ai])
        par2 = np.exp(dur_use['par2'])
        return par1, par2

    @staticmethod
    def _llogis_make_dict(dur_use, par1, par2):
        return dict(dist=dur_use['dist'], par1=par1, par2=par2)

    @staticmethod
    def _gamma_dpars(dur_use, ai):
        par1 = np.exp(dur_use['par1'] + dur_use['age_factors'][ai])
        par2 = np.exp(method.dur_use['par2'])
        return par1, par2

    @staticmethod
    def _gamma_make_dict(dur_use, par1, par2):
        return dict(dist=dur_use['dist'], par1=par1, par2=1/par2)

    def _get_dist_funs(self, dist_name):
        if dist_name == 'lognormal':
            return self._lognormal_dpars, self._lognormal_make_dict
        elif dist_name == 'gamma':
            return self._gamma_dpars, self._gamma_make_dict
        elif dist_name == 'llogis':
            return self._llogis_dpars, self._llogis_make_dict
        else :
            raise ValueError(
                'Unrecognized distribution type for duration of use')

    def set_dur_method(self, ppl, method_used=None):
        """ Time on method depends on age and method """

        dur_method = np.zeros(len(ppl), dtype=float)
        if method_used is None: method_used = ppl.method

        for mname, method in self.methods.items():
            dur_use = method.dur_use
            users = np.nonzero(method_used == method.idx)[-1]
            n_users = len(users)

            if isinstance(dur_use, dict):
                # NOTE: List of available/supported distros can be a property of the class?
                if not (dur_use['dist'] in ['lognormal', 'gamma', 'llogis']):
                    # bail early
                    raise ValueError(
                        'Unrecognized distribution type for duration of use')

                if 'age_factors' in dur_use.keys():
                    age_bins = np.digitize(ppl.age, self.age_bins)
                    par1 = np.zeros(n_users)
                    par2 = np.zeros(n_users)

                    # Get functions based on distro
                    dist_pars_fun, make_dist_dict = self._get_dist_funs(dur_use['dist'])

                    # First set parameters for every agent
                    for ai, ab in enumerate(self.age_bins):
                        par1[age_bins[users] == ai], par2 = dist_pars_fun(dur_use, ai)
                    # Transform to parameters needed by fpsim distributions
                    dist_dict = make_dist_dict(dur_use, par1, par2)
                else:
                    par1 = dur_use['par1']
                    par2 = dur_use['par2']
                    dist_dict = dict(dist=dur_use['dist'], par1=par1, par2=par2)
                # Draw samples
                dur_method[users] = fpu.sample(**dist_dict, size=n_users)
            elif sc.isnumber(dur_use):
                dur_method[users] = dur_use
            else:
                errormsg = 'Unrecognized type for duration of use: expecting a distribution dict or a number'
                raise ValueError(errormsg)

        dt = ppl.pars['timestep'] / fpd.mpy
        timesteps_til_update = np.clip(np.round(dur_method/dt), 1, None)

        return timesteps_til_update

    def choose_method(self, ppl, event=None, jitter=1e-4):
        if event == 'pp1': return self.choose_method_post_birth(ppl)

        else:
            if event is None:  mcp = self.method_choice_pars[0]
            if event == 'pp6': mcp = self.method_choice_pars[6]

            # Initialize arrays and get parameters
            jitter_dist = dict(dist='normal_pos', par1=jitter, par2=jitter)
            choice_array = np.zeros(len(ppl))

            # Loop over age groups and methods
            for key, (age_low, age_high) in fpd.method_age_map.items():
                match_low_high = fpu.match_ages(ppl.age, age_low, age_high)

                for mname, method in self.methods.items():
                    # Get people of this age who are using this method
                    using_this_method = match_low_high & (ppl.method == method.idx)
                    switch_iinds = using_this_method.nonzero()[-1]

                    if len(switch_iinds):

                        # Get probability of choosing each method
                        if mname == 'btl':
                            choice_array[switch_iinds] = method.idx  # Continue, can't actually stop this method
                        else:
                            these_probs = mcp[key][mname]  # Cannot stay on method
                            these_probs = [p if p > 0 else p+fpu.sample(**jitter_dist)[0] for p in these_probs]  # No 0s
                            these_probs = np.array(these_probs) * self.pars['method_weights']  # Scale by weights
                            these_probs = these_probs/sum(these_probs)  # Renormalize
                            these_choices = fpu.n_multinomial(these_probs, len(switch_iinds))  # Choose

                            # Adjust method indexing to correspond to datafile (removing None: Marita to confirm)
                            choice_array[switch_iinds] = np.array(list(mcp.method_idx))[these_choices]

        return choice_array.astype(int)

    def choose_method_post_birth(self, ppl, jitter=1e-4):
        mcp = self.method_choice_pars[1]
        jitter_dist = dict(dist='normal_pos', par1=jitter, par2=jitter)
        choice_array = np.zeros(len(ppl))

        # Loop over age groups and methods
        for key, (age_low, age_high) in fpd.method_age_map.items():
            match_low_high = fpu.match_ages(ppl.age, age_low, age_high)
            switch_iinds = match_low_high.nonzero()[-1]

            if len(switch_iinds):
                these_probs = mcp[key]
                these_probs = [p if p > 0 else p+fpu.sample(**jitter_dist)[0] for p in these_probs]  # No 0s
                these_probs = np.array(these_probs) * self.pars['method_weights']  # Scale by weights
                these_probs = these_probs/sum(these_probs)  # Renormalize
                these_choices = fpu.n_multinomial(these_probs, len(switch_iinds))  # Choose
                choice_array[switch_iinds] = np.array(list(mcp.method_idx))[these_choices]

        return choice_array


class EmpoweredChoice(ContraceptiveChoice):

    def __init__(self, methods=None, location=None, **kwargs):
        super().__init__(**kwargs)
        self.methods = methods or Methods

        # Handle location
        location = location.lower()
        if location == 'kenya':
            self.contra_use_pars = fplocs.kenya.process_contra_use_pars()
            self.method_choice_pars = fplocs.kenya.process_simple_method_pars(self.methods)
        else:
            errormsg = f'Location "{location}" is not currently supported for empowerment analyses'
            raise NotImplementedError(errormsg)

    def init_method_dist(self, ppl):
        # TODO look for initial values
        return self.choose_method(ppl)

    def get_prob_use(self, ppl, year=None, inds=None, event=None, ti=None, tiperyear=None):
        """
        Given an array of indices, return an array of probabilities that each woman will use contraception.
        Probabilities are a function of:
            - data: age, urban, education, wealth, parity, previous contraceptive use, empowerment metrics
            - pars: coefficients applied to each of the above
        """
        p = self.contra_use_pars
        if inds is None: inds = Ellipsis
        rhs = p.intercept
        for vname, vval in p.items():
            if vname not in ['intercept', 'contraception', 'wealthquintile']:
                rhs += vval * ppl[vname]
        rhs += p.contraception * ppl.on_contra[inds]
        prob_use = 1 / (1+np.exp(-rhs))
        return prob_use

    def choose_method(self, ppl, inds=None, event=None, jitter=1e-4):
        """ Choose which method to use """

        # Initialize arrays and get parameters
        mcp = self.method_choice_pars
        jitter_dist = dict(dist='normal_pos', par1=jitter, par2=jitter)
        if inds is None: inds = np.arange(len(ppl))
        n_ppl = len(inds)
        choice_array = np.empty(n_ppl)

        # Loop over age groups, parity, and methods
        for key, (age_low, age_high) in fpd.immutable_method_age_map.items():
            match_low_high = fpu.match_ages(ppl.age[inds], age_low, age_high)
            for parity in range(7):

                for mname, method in self.methods.items():
                    # Get people of this age & parity who are using this method
                    using_this_method = match_low_high & (ppl.parity[inds] == parity) & (ppl.method[inds] == method.idx)
                    switch_iinds = using_this_method.nonzero()[-1]

                    if len(switch_iinds):

                        # Get probability of choosing each method
                        these_probs = [v for k, v in mcp[key][parity].items() if k != mname]  # Cannot stay on method
                        these_probs = [p if p > 0 else p+fpu.sample(**jitter_dist)[0] for p in these_probs]  # No 0s
                        these_probs = np.array(these_probs)/sum(these_probs)  # Renormalize
                        these_choices = fpu.n_multinomial(these_probs, len(switch_iinds))  # Choose

                        # Adjust method indexing for the methods that were removed
                        method_inds = np.array([self.methods[m].idx for m in mcp[key][parity].keys()])
                        choice_array[switch_iinds] = method_inds[these_choices]  # Set values

        return choice_array.astype(int)

    def set_dur_method(self, ppl, method_used=None):
        """ Placeholder function whereby the mean time on method scales with age """

        dur_method = np.empty(len(ppl))
        if method_used is None: method_used = ppl.method

        for mname, method in self.methods.items():  # TODO: refactor this so it loops over the methods used by the ppl
            users = np.nonzero(method_used == method.idx)[-1]
            n_users = len(users)
            dist_dict = sc.dcp(method.dur_use)
            dist_dict['par1'] = dist_dict['par1'] + ppl.age[users]/100
            dist_dict['par2'] = np.array([dist_dict['par2']]*n_users)
            dur_method[users] = fpu.sample(**dist_dict, size=n_users)

        dt = ppl.pars['timestep'] / fpd.mpy
        timesteps_til_update = np.round(dur_method/dt)

        return timesteps_til_update

