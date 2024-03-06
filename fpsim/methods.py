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
    def __init__(self, methods=None, **kwargs):
        self.methods = methods or SimpleMethods
        self.__dict__.update(kwargs)

    @property
    def average_dur_use(self):
        av = 0
        for m in self.methods.values():
            if sc.isnumber(m.dur_use): av += m.dur_use
            elif isinstance(m.dur_use, dict): av += m.dur_use['par1']
        return av / len(self.methods)

    def get_prob_use(self, ppl):
        """ Calculate probabilities that each woman will use contraception """
        return

    def get_contra_users(self, ppl):
        """ Select contraction users, return boolean array """
        prob_use = self.get_prob_use(ppl)
        uses_contra_bool = fpu.binomial_arr(prob_use)
        return uses_contra_bool

    def choose_method(self, ppl):
        pass

    def set_dur_method(self, ppl, method_used=None):
        dt = ppl.pars['timestep'] / fpd.mpy
        ti_contra_update = np.full(len(ppl), sc.randround(ppl.ti + self.average_dur_use/dt), dtype=int)
        return ti_contra_update


class RandomChoice(ContraceptiveChoice):
    """ Randomly choose a method of contraception """
    def __init__(self, **kwargs):
        super().__init__(kwargs)
        return

    def get_prob_use(self, ppl):
        prob_use = np.random.random(len(ppl))
        return prob_use

    def choose_method(self, ppl):
        n_methods = len(self.methods)
        choice_arr = np.random.choice(np.arange(n_methods), size=len(ppl))
        return choice_arr.astype(int)


class SimpleChoice(RandomChoice):
    def __init__(self, coefficients=None, **kwargs):
        """ Args: coefficients """
        super().__init__(**kwargs)
        self.coefficients = coefficients
        return

    def get_prob_use(self, ppl):
        """
        Return an array of probabilities that each woman will use contraception.
        """
        p = self.coefficients
        rhs = p.intercept
        for vname, vval in p.items():
            if vname not in ['intercept']:
                rhs += vval * ppl[vname]
        prob_use = 1 / (1+np.exp(-rhs))
        return prob_use


class EmpoweredChoice(ContraceptiveChoice):

    def __init__(self, methods=None, contra_use_file=None, method_choice_file=None, **kwargs):
        super().__init__(**kwargs)
        self.methods = methods or Methods
        self.contra_use_pars = self.process_contra_use_pars(contra_use_file)
        self.method_choice_pars = self.process_method_pars(method_choice_file)

    @staticmethod
    def process_contra_use_pars(contra_use_file):
        raw_pars = pd.read_csv(contra_use_file)
        pars = sc.objdict()
        for var_dict in raw_pars.to_dict('records'):
            var_name = var_dict['rhs'].replace('_0', '').replace('(', '').replace(')', '').lower()
            pars[var_name] = var_dict['Estimate']
        return pars

    def process_method_pars(self, method_choice_file):
        df = pd.read_csv(method_choice_file)
        # Awful code to speed pandas up
        dd = dict()
        for akey in df.age_grp.unique():
            dd[akey] = dict()
            for pkey in df.parity.unique():
                dd[akey][pkey] = dict()
                for mlabel in df.method.unique():
                    val = df.loc[(df.age_grp == akey) & (df.parity == pkey) & (df.method == mlabel)].percent.values[0]
                    if mlabel != 'Abstinence':
                        mname = [m.name for m in self.methods.values() if m.csv_name == mlabel][0]
                        dd[akey][pkey][mname] = val
                    else:
                        abstinence_val = sc.dcp(val)
                dd[akey][pkey]['othtrad'] += abstinence_val

        return dd

    def get_prob_use(self, ppl, inds=None):
        """
        Given an array of indices, return an array of probabilities that each woman will use contraception.
        Probabilities are a function of:
            - data: age, urban, education, wealth, parity, previous contraceptive use, empowerment metrics
            - pars: coefficients applied to each of the above - see example csv at
            https://github.com/fpsim/fpsim/blob/Kenya_empowerment_DHS/fpsim/locations/kenya/contra_coef.csv
        """
        p = self.contra_use_pars
        if inds is None: inds = Ellipsis
        rhs = (p.intercept
                + (p.age * ppl.age[inds])                           # Age
                + (p.parity * ppl.parity[inds])                     # Parity
                + (p.contraception * ppl.on_contra[inds])           # Whether previously using contraception
                + (p.urban * ppl.urban[inds])                       # Urban/rural [optional]
                # + (p.wealthquintile * ppl.wealthquintile[inds])     # Wealth [optional]
                + (p.edu_attainment * ppl.edu_attainment[inds])     # Educational attainment [optional]
                + (p.paid_employment * ppl.paid_employment[inds])   # Paid employment [optional]
                + (p.decision_wages * ppl.decision_wages[inds])     # Decision over wages [optional]
                + (p.decision_health * ppl.decision_health[inds])   # Decision over wages [optional]
                + (p.sexual_autonomy * ppl.sexual_autonomy[inds])   # Sexual autonomy [optional]
               )
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

