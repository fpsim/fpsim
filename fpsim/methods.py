"""
Contraceptive methods

Idea:
    A method selector should be a generic class (an intervention?). We want the existing matrix-based
    method and the new duration-based method to both be instances of this.

"""

# %% Imports
import numpy as np
import sciris as sc
import starsim as ss  # TODO add to dependencies
from . import utils as fpu
from . import defaults as fpd
from . import locations as fplocs

__all__ = ['Method', 'make_methods', 'ContraceptiveChoice', 'RandomChoice', 'SimpleChoice', 'StandardChoice']


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


def make_methods():

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

    m = sc.prettyobj()
    m.method_list = sc.dcp(method_list)
    m.method_map = sc.dcp(method_map)
    m.Methods = sc.dcp(Methods)
    return m



# %% Define classes to contain information about the way women choose contraception

class ContraceptiveChoice:
    def __init__(self, methods=None, pars=None, **kwargs):
        self.methods = methods or make_methods().Methods
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
    def __init__(self, pars=None, location=None, method_choice_df=None, method_time_df=None, **kwargs):
        """ Args: coefficients """
        super().__init__(**kwargs)
        default_pars = dict(
            prob_use_year=2000,
            prob_use_intercept=0.0,
            prob_use_trend_par=0.0,
            force_choose=False,  # Whether to force non-users to choose a method
            method_weights=np.ones(self.n_methods),
            max_dur=100*fpd.mpy,  # Maximum duration of use in months
        )
        updated_pars = sc.mergedicts(default_pars, pars)
        self.pars = sc.mergedicts(self.pars, updated_pars)
        self.pars.update(kwargs)  # TODO: check

        # Handle location
        location = fpd.get_location(location)
        self.init_method_pars(location, method_choice_df=method_choice_df, method_time_df=method_time_df)

        return

    def init_method_pars(self, location, method_choice_df=None, method_time_df=None):
        # Get the correct module, from either registry or built-in
        if location in fpd.location_registry:
            location_module = fpd.location_registry[location]
        else:
            location_module = fplocs  # fallback to built-in only if not registered

        self.contra_use_pars = location_module.data_utils.process_contra_use('simple', location)  # Set probability of use
        method_choice_pars, init_dist = location_module.data_utils.process_markovian_method_choice(self.methods, location, df=method_choice_df)  # Method choice
        self.method_choice_pars = method_choice_pars
        self.init_dist = init_dist
        self.methods = location_module.data_utils.process_dur_use(self.methods, location, df=method_time_df)  # Reset duration of use

        # Handle age bins -- find a more robust way to do this
        self.age_bins = np.sort([fpd.method_age_map[k][1] for k in self.method_choice_pars[0].keys() if k != 'method_idx'])
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

        # Initialize probability of use
        rhs = np.full_like(ppl.age, fill_value=p.intercept)
        age_bins = np.digitize(ppl.age, self.age_bins)
        for ai, ab in enumerate(self.age_bins):
            rhs[age_bins == ai] += p.age_factors[ai]
            if ai > 1:
                rhs[(age_bins == ai) & ppl.ever_used_contra] += p.age_ever_user_factors[ai-1]
        rhs[ppl.ever_used_contra] += p.fp_ever_user

        # The yearly trend
        rhs += (year - self.pars['prob_use_year']) * self.pars['prob_use_trend_par']
        # This parameter can be positive or negative
        rhs += self.pars['prob_use_intercept']
        prob_use = 1 / (1+np.exp(-rhs))
        return prob_use

    @staticmethod
    def _lognormal_dpars(dur_use, ai):
        par1 = np.exp(dur_use['par1'] + dur_use['age_factors'][ai])  # par1 is the 'meanlog' from the csv file. exp(par1) is the 'scale' parameter
        par2 = np.exp(dur_use['par2'])
        return par1, par2

    @staticmethod
    def _llogis_dpars(dur_use, ai):
        par1 = np.exp(dur_use['par1'])
        par2 = np.exp(dur_use['par2'] + dur_use['age_factors'][ai])
        return par1, par2

    @staticmethod
    def _weibull_dpars(dur_use, ai):
        par1 = np.exp(dur_use['par1'])
        par2 = np.exp(dur_use['par2'] + dur_use['age_factors'][ai])
        return par1, par2

    @staticmethod
    def _exp_dpars(dur_use, ai):
        par1 = 1/np.exp(dur_use['par1'] + dur_use['age_factors'][ai])
        return par1, None

    @staticmethod
    def _gamma_dpars(dur_use, ai):
        par1 = np.exp(dur_use['par1'])
        par2 = 1/np.exp(dur_use['par2'] + dur_use['age_factors'][ai])
        return par1, par2

    @staticmethod
    def _make_dict(dur_use, par1, par2):
        return dict(dist=dur_use['dist'], par1=par1, par2=par2)

    def _get_dist_funs(self, dist_name):
        if dist_name == 'lognormal_sps':
            return self._lognormal_dpars, self._make_dict
        elif dist_name == 'gamma':
            return self._gamma_dpars, self._make_dict
        elif dist_name == 'llogis':
            return self._llogis_dpars, self._make_dict
        elif dist_name == 'weibull':
            return self._weibull_dpars, self._make_dict
        elif dist_name == 'exponential':
            return self._exp_dpars, self._make_dict
        else:
            raise ValueError(
                f'Unrecognized distribution type {dist_name} for duration of use')

    def set_dur_method(self, ppl, method_used=None):
        """ Time on method depends on age and method """

        dur_method = np.zeros(len(ppl), dtype=float)
        if method_used is None: method_used = ppl.method

        for mname, method in self.methods.items():
            dur_use = method.dur_use
            users = np.nonzero(method_used == method.idx)[-1]
            n_users = len(users)

            if n_users:
                if isinstance(dur_use, dict):
                    # NOTE: List of available/supported distros can be a property of the class?
                    if not (dur_use['dist'] in ['lognormal_sps', 'gamma', 'llogis', 'exponential', 'weibull', 'unif']):
                        # bail early
                        raise ValueError(
                            f'Unrecognized distribution type for duration of use: {dur_use["dist"]}')

                    if 'age_factors' in dur_use.keys():
                        # Get functions based on distro and set for every agent
                        dist_pars_fun, make_dist_dict = self._get_dist_funs(dur_use['dist'])
                        age_bins = np.digitize(ppl.age[users], self.age_bins)
                        par1, par2 = dist_pars_fun(dur_use, age_bins)

                        # Transform to parameters needed by fpsim distributions
                        dist_dict = make_dist_dict(dur_use, par1, par2)
                    else:
                        par1 = dur_use['par1']
                        par2 = dur_use['par2']
                        dist_dict = dict(dist=dur_use['dist'], par1=par1, par2=par2)

                    # Draw samples of how many months women use this method
                    dur_method[users] = fpu.sample(**dist_dict, size=n_users)

                elif sc.isnumber(dur_use):
                    dur_method[users] = dur_use
                else:
                    errormsg = 'Unrecognized type for duration of use: expecting a distribution dict or a number'
                    raise ValueError(errormsg)

        timesteps_til_update = np.clip(np.round(dur_method), 1, self.pars['max_dur'])  # Include a maximum. Durs seem way too high

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
                            try:
                                these_probs = mcp[key][mname]  # Cannot stay on method
                            except:
                                errormsg = f'Cannot find {key} in method switch for {mname}!'
                                raise ValueError(errormsg)
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


class StandardChoice(SimpleChoice):
    """
    Default contraceptive choice module.
    Contraceptive choice is based on age, education, wealth, parity, and prior use.
    """
    def __init__(self, pars=None, location=None, **kwargs):
        # Initialize base class - this adds parameters and default data
        super().__init__(pars=pars, location=location, **kwargs)

        # Get the correct module, from either registry or built-in
        if location in fpd.location_registry:
            location_module = fpd.location_registry[location]
        else:
            location_module = fplocs  # fallback to built-in only if not registered

        # Now overwrite the default prob_use parameters with the mid-choice coefficients
        location = fpd.get_location(location)
        self.contra_use_pars = location_module.data_utils.process_contra_use('mid', location)  # Process the coefficients

        # Store the age spline
        self.age_spline = location_module.data_utils.age_spline('25_40')

        return

    def get_prob_use(self, ppl, year=None, event=None, ti=None, tiperyear=None):
        """
        Return an array of probabilities that each woman will data_use contraception.
        """
        # Figure out which coefficients to data_use
        if event is None : p = self.contra_use_pars[0]
        if event == 'pp1': p = self.contra_use_pars[1]
        if event == 'pp6': p = self.contra_use_pars[2]

        # Initialize with intercept
        rhs = np.full_like(ppl.age, fill_value=p.intercept)

        # Add all terms that don't involve age/education level factors
        for term in ['ever_used_contra', 'urban', 'parity', 'wealthquintile']:
            rhs += p[term] * ppl[term]

        # Add age
        int_age = ppl.int_age
        int_age[int_age < fpd.min_age] = fpd.min_age
        int_age[int_age >= fpd.max_age_preg] = fpd.max_age_preg-1
        dfa = self.age_spline.loc[int_age]
        rhs += p.age_factors[0] * dfa['knot_1'].values + p.age_factors[1] * dfa['knot_2'].values + p.age_factors[2] * dfa['knot_3'].values
        rhs += (p.age_ever_user_factors[0] * dfa['knot_1'].values * ppl.ever_used_contra
                + p.age_ever_user_factors[1] * dfa['knot_2'].values * ppl.ever_used_contra
                + p.age_ever_user_factors[2] * dfa['knot_3'].values * ppl.ever_used_contra)

        # Add education levels
        primary = (ppl.edu_attainment > 1) & (ppl.edu_attainment <= 6)
        secondary = ppl.edu_attainment > 6
        rhs += p.edu_factors[0] * primary + p.edu_factors[1] * secondary

        # Add time trend
        rhs += (year - self.pars['prob_use_year'])*self.pars['prob_use_trend_par']

        # Finish
        prob_use = 1 / (1+np.exp(-rhs))

        return prob_use
