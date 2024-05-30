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
        default_pars = dict(
            p_use=0.5,
        )
        self.pars = sc.mergedicts(default_pars, pars)

    @property
    def average_dur_use(self):
        av = 0
        for m in self.methods.values():
            if sc.isnumber(m.dur_use): av += m.dur_use
            elif isinstance(m.dur_use, dict): av += m.dur_use['par1']
        return av / len(self.methods)

    def get_prob_use(self, ppl, event=None):
        """ Calculate probabilities that each woman will use contraception """
        prob_use = np.random.random(len(ppl))
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

    def get_contra_users(self, ppl, event=None):
        """ Select contraction users, return boolean array """
        prob_use = self.get_prob_use(ppl, event=event)
        uses_contra_bool = self.pars['p_use'] > prob_use
        return uses_contra_bool

    def choose_method(self, ppl):
        pass

    def set_dur_method(self, ppl, method_used=None):
        dt = ppl.pars['timestep'] / fpd.mpy
        ti_contra_update = np.full(len(ppl), sc.randround(ppl.ti + self.average_dur_use/dt), dtype=int)
        return ti_contra_update


class RandomChoice(ContraceptiveChoice):
    """ Randomly choose a method of contraception """
    def __init__(self, pars=None, methods=None, **kwargs):
        super().__init__(methods=methods, **kwargs)
        default_pars = dict(
            p_use=0.5,
            method_mix=np.array([1/self.n_methods]*self.n_methods),
        )
        self.pars = sc.mergedicts(default_pars, pars)
        return

    def choose_method(self, ppl):
        choice_arr = np.random.choice(np.arange(self.n_methods), size=len(ppl), p=self.pars['method_mix'])
        return choice_arr.astype(int)


class SimpleChoice(RandomChoice):
    def __init__(self, coef=None, coef_pp1=None, coef_pp6=None, **kwargs):
        """ Args: coefficients """
        super().__init__(**kwargs)
        self.coef = coef
        self.coef_pp1 = coef_pp1 or coef
        self.coef_pp6 = coef_pp6 or coef
        self.n_age_bins = len(coef.age_bin_edges)
        return

    def get_prob_use(self, ppl, event=None):
        """
        Return an array of probabilities that each woman will use contraception.
        """
        # Figure out which coefficients to use
        if event is None : p = self.coef
        if event == 'pp1': p = self.coef_pp1
        if event == 'pp6': p = self.coef_pp6

        # Calculate probability of use
        rhs = np.full_like(ppl.age, fill_value=p.intercept)
        age_bins = np.digitize(ppl.age, self.coef.age_bin_edges)
        for ab in range(1, self.n_age_bins):
            rhs[age_bins==ab] += self.coef.age_bin_vals[ab-1]
        prob_use = 1 / (1+np.exp(-rhs))
        prob_use[(ppl.age<18) | (ppl.age>50)] = 0  # CHECK
        return prob_use

    def set_dur_method(self, ppl, method_used=None):
        """ Placeholder function whereby the mean time on method scales with age """

        dur_method = np.empty(len(ppl))
        if method_used is None: method_used = ppl.method
        age_bins = np.digitize(ppl.age, self.coef.age_bin_edges)

        for mname, method in self.methods.items():
            users = np.nonzero(method_used == method.idx)[-1]
            n_users = len(users)

            if mname in ['none', 'pill', 'inj', 'cond', 'othtrad', 'othmod']:
                par1 = np.zeros(n_users)
                for ab in range(1, self.n_age_bins):
                    par1[age_bins[users]==ab] = method.dur_use['par1'] + method.dur_use['age_bin_vals'][ab-1]
                par1[par1==0] = 1e-3
            else:
                par1 = method.dur_use['par1']

            dist_dict = dict(dist=method.dur_use['dist'], par1=par1, par2=method.dur_use['par2'])
            if (dist_dict['dist'] == 'gamma'):
                dist_dict['par1'][dist_dict['par1'] < 0] = -dist_dict['par1'][dist_dict['par1'] < 0]
                if dist_dict['par2'] < 0:
                    dist_dict['par2'] = -dist_dict['par2']
            dur_method[users] = fpu.sample(**dist_dict, size=n_users)

        dt = ppl.pars['timestep'] / fpd.mpy
        ti_contra_update = ppl.ti + sc.randround(dur_method/dt)

        return ti_contra_update



class EmpoweredChoice(ContraceptiveChoice):

    def __init__(self, methods=None, location=None, **kwargs):
        super().__init__(**kwargs)
        self.methods = methods or Methods

        # Handle location
        location = location.lower()
        if location == 'kenya':
            self.contra_use_pars = fplocs.kenya.process_contra_use_pars()
            self.method_choice_pars = fplocs.kenya.process_method_pars(self.methods)
        else:
            errormsg = f'Location "{location}" is not currently supported for empowerment analyses'
            raise NotImplementedError(errormsg)

    def get_prob_use(self, ppl, inds=None, event=None):
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

    def choose_method(self, ppl, inds=None, jitter=1e-4):
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
        ti_contra_update = ppl.ti + sc.randround(dur_method/dt)

        return ti_contra_update

