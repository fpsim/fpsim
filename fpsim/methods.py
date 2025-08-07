"""
Contraceptive methods

Idea:
    A method selector should be a generic class (an intervention?). We want the existing matrix-based
    method and the new duration-based method to both be instances of this.

"""

# %% Imports
import numpy as np
import sciris as sc
import starsim as ss
from scipy.special import expit
from . import utils as fpu
from . import defaults as fpd
from . import locations as fplocs

__all__ = ['Method', 'make_methods', 'make_method_list', 'ContraPars', 'make_contra_pars', 'ContraceptiveChoice', 'RandomChoice', 'SimpleChoice', 'StandardChoice']


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


def make_method_list():
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
    return sc.dcp(method_list)


def make_method_map(method_list):
    method_map = {method.label: method.idx for method in method_list}
    return method_map


def make_methods(method_list=None):
    if method_list is None: method_list = make_method_list()
    return ss.ndict(method_list, type=Method)


# %% Define parameters
class ContraPars(ss.Pars):
    def __init__(self, **kwargs):
        super().__init__()

        # Methods
        self.methods = make_method_list()  # Default methods

        # Probabilities and choices
        self.p_use = ss.bernoulli(p=0.5)
        self.force_choose = False  # Whether to force non-users to choose a method
        self.method_mix = 'uniform'  #np.array([1/self.n_methods]*self.n_methods)
        self.method_weights = None  #np.ones(n_methods)

        # mCPR trend
        self.prob_use_year = 2000
        self.prob_use_intercept = 0.0
        self.prob_use_trend_par = 0.0

        # Settings and other misc
        self.max_dur = ss.years(100)  # Maximum duration of use in years
        self.update(kwargs)
        return


def make_contra_pars():
    """ Shortcut for making a new instance of ContraPars """
    return ContraPars()


# %% Define classes to contain information about the way women choose contraception

class ContraceptiveChoice(ss.Connector):
    def __init__(self, pars=None, **kwargs):
        """
        Base contraceptive choice module
        """
        super().__init__(name='contraception')

        # Handle parameters
        default_pars = ContraPars()
        self.define_pars(**default_pars)
        self.update_pars(pars, **kwargs)

        # Copy methods as main attribute
        self.methods = make_methods(self.pars.methods)  # Store the methods as an ndict
        self.n_options = len(self.methods)
        self.n_methods = len([m for m in self.methods if m != 'none'])

        # Process pars
        if self.pars.method_mix == 'uniform':
            self.pars.method_mix = np.array([1/self.n_methods]*self.n_methods)
        if self.pars.method_weights is None:
            self.pars.method_weights = np.ones(self.n_methods)

        self.init_dist = None

        return

    def init_results(self):
        """
        Initialize results for this module
        """
        self.define_results(
            ss.Result('n_at_risk_non_users', scale=False, label="Number of non-users at risk of pregnancy (aCPR)"),
            ss.Result('n_at_risk_users', scale=False, label="Number of users at risk of pregnancy (aCPR)"),
            ss.Result('n_non_users', scale=False, label="Number of non-users (CPR)"),
            ss.Result('n_mod_users', scale=False, label="Number of modern contraceptive users (mCPR)"),
            ss.Result('n_users', scale=False, label="Number of contraceptive users (CPR)"),
            ss.Result('mcpr', scale=False, label="Modern contraceptive prevalence rate (mCPR)"),
            ss.Result('cpr', scale=False, label="Contraceptive prevalence rate (CPR)"),
            ss.Result('acpr', scale=False, label="Active contraceptive prevalence rate (aCPR)"),
        )
        return

    @property
    def average_dur_use(self):
        av = 0
        for m in self.methods.values():
            if sc.isnumber(m.dur_use): av += m.dur_use
            elif isinstance(m.dur_use, dict): av += m.dur_use['par1']
        return av / len(self.methods)

    def init_post(self):
        """
         Decide who will start using contraception, when, which contraception method and the
         duration on that method. This method is called by the simulation to initialise the
         people object at the beginning of the simulation and new people born during the simulation.
         """
        ppl = self.sim.people
        fecund = ppl.female & (ppl.age < self.sim.pars.fp['age_limit_fecundity'])
        fecund_uids = fecund.uids

        # Look for women who have reached the time to choose
        time_to_set_contra_uids = fecund_uids[(ppl.fp.ti_contra[fecund_uids] == 0)]
        self.init_contraception(time_to_set_contra_uids)
        return

    def init_contraception(self, uids):
        """
        Used for all agents at the start of a sim and for newly active agents throughout
        """
        contra_users, _ = self.get_contra_users(uids)
        self.start_contra(contra_users)
        self.init_methods(contra_users)
        return

    def start_contra(self, uids):
        """ Wrapper method to start contraception for a set of users """
        self.sim.people.fp.on_contra[uids] = True
        self.sim.people.fp.ever_used_contra[uids] = 1
        return

    def init_methods(self, uids):
        # Set initial distribution of methods
        self.sim.people.fp.method[uids] = self.init_method_dist(uids)
        method_dur = self.set_dur_method(uids)
        self.sim.people.fp.ti_contra[uids] = self.ti + method_dur
        return

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
        errormsg = ('remove_method is not currently functional. See example in test_parameters.py if you want to run a '
                    'simulation with a subset of the standard set of methods. The remove_method logic needs to be'
                    'replaced with something that can remove a method partway through a simulation.')
        raise NotImplementedError(errormsg)
        # method = self.get_method_by_label(method_label)
        # del self.methods[method.name]

    def get_prob_use(self, uids, event=None):
        pass

    def get_contra_users(self, uids, event=None):
        """ Select contraception users, return boolean array """
        self.get_prob_use(uids, event=event)  # Call this to reset p_use parameter
        users, non_users = self.pars.p_use.split(uids)
        return users, non_users

    def update_contra(self, uids):
        """ Update contraceptive choices for a set of users. """
        sim = self.sim
        ti = sim.ti
        ppl = sim.people

        # If people are 1 or 6m postpartum, we use different parameters for updating their contraceptive decisions
        is_pp1 = (ppl.postpartum_dur[uids] == 1)
        is_pp6 = (ppl.postpartum_dur[uids] == 6) & ~ppl.on_contra[uids]  # They may have decided to use contraception after 1m
        pp0 = uids[~(is_pp1 | is_pp6)]
        pp1 = uids[is_pp1]
        pp6 = uids[is_pp6]

        # Update choices for people who aren't postpartum
        if len(pp0):

            # If force_choose is True, then all non-users will be made to pick a method
            if self.pars['force_choose']:
                must_use = pp0[~ppl.on_contra[pp0]]
                choosers = pp0[ppl.on_contra[pp0]]

                if len(must_use):
                    self.start_contra(must_use)  # Start contraception for those who must use
                    ppl.method[must_use] = self.choose_method(must_use)

            else:
                choosers = pp0

            # Get previous users and see whether they will switch methods or stop using
            if len(choosers):

                users, non_users = self.get_contra_users(choosers)

                if len(non_users):
                    ppl.on_contra[non_users] = False  # Set non-users to not using contraception
                    ppl.method[non_users] = 0  # Set method to zero for non-users

                # For those who keep using, choose their next method
                if len(users):
                    self.start_contra(users)
                    ppl.method[users] = self.choose_method(users)

            # Validate
            n_methods = len(self.methods)
            invalid_vals = (ppl.method[pp0] >= n_methods) * (ppl.method[pp0] < 0) * (np.isnan(ppl.method[pp0]))
            if invalid_vals.any():
                errormsg = f'Invalid method set: ti={pp0.ti}, inds={invalid_vals.nonzero()[-1]}'
                raise ValueError(errormsg)

        # Now update choices for postpartum people. Logic here is simpler because none of these
        # people should be using contraception currently. We first check that's the case, then
        # have them choose their contraception options.
        ppdict = {'pp1': pp1, 'pp6': pp6}
        for event, pp in ppdict.items():
            if len(pp):
                if ppl.on_contra[pp].any():
                    errormsg = 'Postpartum women should not currently be using contraception.'
                    raise ValueError(errormsg)
                users, _ = self.get_contra_users(pp, event=event)
                self.start_contra(users)
                on_contra = pp[ppl.on_contra[pp]]
                off_contra = pp[~ppl.on_contra[pp]]

                # Set method for those who use contraception
                if len(on_contra):
                    method_used = self.choose_method(on_contra, event=event)
                    ppl.method[on_contra] = method_used

                if len(off_contra):
                    ppl.method[off_contra] = 0
                    if event == 'pp1':  # For women 1m postpartum, choose again when they are 6 months pp
                        ppl.ti_contra[off_contra] = ti + 5

        # Set duration of use for everyone, and reset the time they'll next update
        durs_fixed = (ppl.postpartum_dur[uids] == 1) & (ppl.method[uids] == 0)
        update_durs = uids[~durs_fixed]
        dur_methods = self.set_dur_method(update_durs)

        # Check validity
        if (dur_methods < 0).any():
            raise ValueError('Negative duration of method use')

        ppl.ti_contra[update_durs] = ti + dur_methods

        return

    def set_dur_method(self, uids, method_used=None):
        # todo make this aware of starsim time units. right now assumes average_dur_use is in years and timestep par is in years
        # dt = ppl.sim.t.dt_year * fpd.mpy
        dt = self.t.dt_year
        timesteps_til_update = np.full(len(uids), np.round(self.average_dur_use/dt), dtype=int)
        return timesteps_til_update

    def set_method(self, uids):
        """ Wrapper for choosing method and assigning duration of use """
        ppl = self.sim.people
        method_used = self.choose_method(uids)
        ppl.method[uids] = method_used

        # Set the duration of use
        dur_method = self.set_dur_method(uids)
        ppl.ti_contra[uids] = self.ti + dur_method

        return

    def step(self):
        # TODO, could move all update logic to here...
        pass

    def update_results(self):
        """
        Note that we are not including LAM users in mCPR as this model counts
        all women passively using LAM but DHS data records only women who self-report
        LAM which is much lower. Follows the DHS definition of mCPR.
        """
        ppl = self.sim.people
        method_age = self.sim.pars.fp['method_age'] <= ppl.age
        fecund_age = ppl.age < self.sim.pars.fp['age_limit_fecundity']
        denominator = method_age * fecund_age * ppl.female * ppl.alive

        # Track mCPR
        modern_methods_num = [idx for idx, m in enumerate(self.methods.values()) if m.modern]
        numerator = np.isin(ppl.fp.method, modern_methods_num)
        n_no_method = np.sum((ppl.fp.method == 0) * denominator)
        n_mod_users = np.sum(numerator * denominator)
        self.results['n_non_users'][self.ti] += n_no_method
        self.results['n_mod_users'][self.ti] += n_mod_users
        self.results['mcpr'][self.ti] += sc.safedivide(n_mod_users, sum(denominator))

        # Track CPR: includes newer ways to conceptualize contraceptive prevalence.
        # Includes women using any method of contraception, including LAM
        numerator = ppl.fp.method != 0
        cpr = np.sum(numerator * denominator)
        self.results['n_users'][self.ti] += cpr
        self.results['cpr'][self.ti] += sc.safedivide(cpr, sum(denominator))

        # Track aCPR
        # Denominator of possible users excludes pregnant women and those not sexually active in the last 4 weeks
        # Used to compare new metrics of contraceptive prevalence and eventually unmet need to traditional mCPR definitions
        denominator = method_age * fecund_age * ppl.female * ~ppl.fp.pregnant * ppl.fp.sexually_active
        numerator = ppl.fp.method != 0
        n_at_risk_non_users = np.sum((ppl.fp.method == 0) * denominator)
        n_at_risk_users = np.sum(numerator * denominator)
        self.results['n_at_risk_non_users'][self.ti] += n_at_risk_non_users
        self.results['n_at_risk_users'][self.ti] += n_at_risk_users
        self.results['acpr'][self.ti] = sc.safedivide(n_at_risk_users, sum(denominator))

        return


class RandomChoice(ContraceptiveChoice):
    """ Randomly choose a method of contraception """
    def __init__(self, pars=None, **kwargs):
        super().__init__(pars=pars, **kwargs)
        self.init_dist = self.pars['method_mix']
        self._method_mix = ss.choice(a=np.arange(1, self.n_methods+1))
        return

    def init_method_dist(self, uids):
        return self.choose_method(uids)

    def choose_method(self, uids, event=None):
        choice_arr = self._method_mix.rvs(uids)
        return choice_arr.astype(int)


class SimpleChoice(RandomChoice):
    def __init__(self, pars=None, location=None, method_choice_df=None, method_time_df=None, **kwargs):
        """ Args: coefficients """
        super().__init__(pars=pars, **kwargs)
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
        self._method_mix.set(p=init_dist)  # TODO check
        self.methods = location_module.data_utils.process_dur_use(self.methods, location, df=method_time_df)  # Reset duration of use

        # Handle age bins -- find a more robust way to do this
        self.age_bins = np.sort([fpd.method_age_map[k][1] for k in self.method_choice_pars[0].keys() if k != 'method_idx'])
        return

    def init_method_dist(self, uids):
        ppl = self.sim.people
        if self.init_dist is not None:
            choice_array = np.zeros(len(uids))

            # Loop over age groups and methods
            for key, (age_low, age_high) in fpd.method_age_map.items():
                this_age_bools = (ppl.age[uids] >= age_low) & (ppl.age[uids] < age_high)
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

    def get_prob_use(self, uids, event=None):
        """
        Return an array of probabilities that each woman will use contraception.
        """
        ppl = self.sim.people
        year = self.t.now()

        # Figure out which coefficients to use
        if event is None : p = self.contra_use_pars[0]
        if event == 'pp1': p = self.contra_use_pars[1]
        if event == 'pp6': p = self.contra_use_pars[2]

        # Initialize probability of use
        rhs = np.full_like(ppl.age[uids], fill_value=p.intercept)
        age_bins = np.digitize(ppl.age[uids], self.age_bins)
        for ai, ab in enumerate(self.age_bins):
            rhs[age_bins == ai] += p.age_factors[ai]
            if ai > 1:
                rhs[(age_bins == ai) & ppl.ever_used_contra[uids]] += p.age_ever_user_factors[ai-1]
        rhs[ppl.ever_used_contra[uids]] += p.fp_ever_user

        # The yearly trend
        rhs += (year - self.pars['prob_use_year']) * self.pars['prob_use_trend_par']
        # This parameter can be positive or negative
        rhs += self.pars['prob_use_intercept']
        prob_use = 1 / (1+np.exp(-rhs))

        # Set
        self.pars.p_use.set(p=prob_use)  # Set the probability of use parameter
        return

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

    def set_dur_method(self, uids, method_used=None):
        """ Time on method depends on age and method """
        ppl = self.sim.people

        dur_method = np.zeros(len(uids), dtype=float)
        if method_used is None: method_used = ppl.fp.method[uids]

        for mname, method in self.methods.items():
            dur_use = method.dur_use
            users = np.nonzero(method_used == method.idx)[-1]
            n_users = len(users)

            if n_users:
                if isinstance(dur_use, dict):
                    # NOTE: List of available/supported distros can be a property of the class?
                    if not (dur_use['dist'] in ['lognormal', 'lognormal_sps', 'gamma', 'llogis', 'exponential', 'weibull', 'unif']):
                        # bail early
                        raise ValueError(
                            f'Unrecognized distribution type for duration of use: {dur_use["dist"]}')

                    if 'age_factors' in dur_use.keys():
                        # Get functions based on distro and set for every agent
                        dist_pars_fun, make_dist_dict = self._get_dist_funs(dur_use['dist'])
                        age_bins = np.digitize(ppl.age[uids[users]], self.age_bins)
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

        dt = ppl.sim.t.dt_year * fpd.mpy
        timesteps_til_update = np.clip(np.round(dur_method/dt), 1, self.pars['max_dur'].v)  # Include a maximum. Durs seem way too high

        return timesteps_til_update

    def choose_method(self, uids, event=None, jitter=1e-4):
        ppl = self.sim.people
        if event == 'pp1': return self.choose_method_post_birth(uids)

        else:
            if event is None:  mcp = self.method_choice_pars[0]
            if event == 'pp6': mcp = self.method_choice_pars[6]

            # Initialize arrays and get parameters
            jitter_dist = dict(dist='normal_pos', par1=jitter, par2=jitter)
            choice_array = np.zeros(len(uids))

            # Loop over age groups and methods
            for key, (age_low, age_high) in fpd.method_age_map.items():
                match_low_high = (ppl.age[uids] >= age_low) & (ppl.age[uids] < age_high)

                for mname, method in self.methods.items():
                    # Get people of this age who are using this method
                    using_this_method = match_low_high & (ppl.method[uids] == method.idx)
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

    def choose_method_post_birth(self, uids, jitter=1e-4):
        ppl = self.sim.people
        mcp = self.method_choice_pars[1]
        jitter_dist = dict(dist='normal_pos', par1=jitter, par2=jitter)
        choice_array = np.zeros(len(uids))

        # Loop over age groups and methods
        for key, (age_low, age_high) in fpd.method_age_map.items():
            match_low_high = (ppl.age[uids] >= age_low) & (ppl.age[uids] < age_high)
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

    def get_prob_use(self, uids, event=None):
        """
        Return an array of probabilities that each woman will use contraception.
        """
        ppl = self.sim.people
        year = self.t.now()

        # Figure out which coefficients to use
        if event is None : p = self.contra_use_pars[0]
        if event == 'pp1': p = self.contra_use_pars[1]
        if event == 'pp6': p = self.contra_use_pars[2]

        # Initialize with intercept
        rhs = np.full_like(ppl.age[uids], fill_value=p.intercept)

        # Add all terms that don't involve age/education level factors
        for term in ['ever_used_contra', 'urban', 'parity', 'wealthquintile']:
            rhs += p[term] * ppl[term][uids]

        # Add age
        int_age = ppl.int_age(uids)
        int_age[int_age < fpd.min_age] = fpd.min_age
        int_age[int_age >= fpd.max_age_preg] = fpd.max_age_preg-1
        dfa = self.age_spline.loc[int_age]
        rhs += p.age_factors[0] * dfa['knot_1'].values + p.age_factors[1] * dfa['knot_2'].values + p.age_factors[2] * dfa['knot_3'].values
        rhs += (p.age_ever_user_factors[0] * dfa['knot_1'].values * ppl.ever_used_contra[uids]
                + p.age_ever_user_factors[1] * dfa['knot_2'].values * ppl.ever_used_contra[uids]
                + p.age_ever_user_factors[2] * dfa['knot_3'].values * ppl.ever_used_contra[uids])

        # Add education levels
        primary = (ppl.edu.attainment[uids] > 1) & (ppl.edu.attainment[uids] <= 6)
        secondary = ppl.edu.attainment[uids] > 6
        rhs += p.edu_factors[0] * primary + p.edu_factors[1] * secondary

        # Add time trend
        rhs += (year - self.pars['prob_use_year'])*self.pars['prob_use_trend_par']

        # Finish
        prob_use = expit(rhs)
        self.pars.p_use.set(p=prob_use)
        return
