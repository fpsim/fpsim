'''
Define classes and functions for the Experiment class (running sims and comparing them to data)
'''


import yaml
import numpy as np
import pylab as pl
import pandas as pd
import sciris as sc
from .settings import options as fpo
from . import defaults as fpd
from . import parameters as fpp
from . import sim as fps


__all__ = ['Experiment', 'Fit', 'compute_gof', 'diff_summaries']

# ...more settings
min_age = 15
max_age = 50
bin_size = 5
first_birth_age = 25  # age to start assessing first birth age in model
mpy = 12  # Months per year

# Flags for what to run
default_flags = sc.objdict(
    popsize       = 1, # Population size and growth over time on whole years, adjusted for n number of agents; 'pop_size'
    ageparity   = 1, # Population distribution of agents in each age/parity bin (age-parity plot); 'ageparity'
    first_birth   = 1, # Age at first birth mean with standard deviation; 'age_first_birth'
    birth_space   = 1, # Birth spacing both in bins and mean with standard deviation; 'spacing'
    mcpr          = 1, # Modern contraceptive prevalence; 'mcpr'
    methods       = 1, # Overall percentage of method use and method use among users; 'methods'
    mmr           = 1, # Maternal mortality ratio at end of sim in model vs data; 'maternal_mortality_ratio'
    infant_m      = 1, # Infant mortality rate at end of sim in model vs data; 'infant_mortality_rate'
    cdr           = 1, # Crude death rate at end of sim in model vs data; 'crude_death_rate'
    cbr           = 1, # Crude birth rate (per 1000 inhabitants); 'crude_birth_rate'
    tfr           = 1, # Total fertility rate
    asfr          = 1, # Age-specific fertility rate
)


class Experiment(sc.prettyobj):
    '''
    Class for running calibration to data. Effectively, it runs a single sim and
    compares it to data.

    Args:
        pars (dict): dictionary of parameters
        flags (dict): which analyses to run; see ``fp.experiment.default_flags`` for options
        label (str): label of experiment
        kwargs (dict): passed into pars
    '''

    def __init__(self, pars=None, flags=None, label=None, **kwargs):
        self.flags = sc.mergedicts(default_flags, flags, _copy=True) # Set flags for what gets run
        self.pars = pars if pars else fpp.pars(**kwargs)
        self.model = sc.objdict()
        self.data = sc.objdict()
        self.method_keys = None
        self.initialized = False
        self.label = label
        return


    def load_data(self, key, **kwargs):
        ''' Load data from various formats '''
        files = self.pars['filenames']
        path = files['base'] / files[key]
        if path.suffix == '.obj':
            data = sc.load(path, **kwargs)
        elif path.suffix == '.json':
            data = sc.loadjson(path, **kwargs)
        elif path.suffix == '.csv':
            data = pd.read_csv(path, **kwargs)
        elif path.suffix == '.yaml':
            with open(path) as f:
                data = yaml.safe_load(f, **kwargs)
        else:
            errormsg = f'Unrecognized file format for: {path}'
            raise ValueError(errormsg)
        return data


    def extract_data(self):
        ''' Load data '''

        json = self.load_data('basic_dhs')

        self.data.update(json)

        #self.data['pregnancy_parity'] = self.load_data('pregnancy_parity')

        # Extract population size over time
        if self.pars:
            n = self.pars['n_agents']
        else:
            n = 1000 # Use default if not available
            print(f'Warning: parameters not defined, using default of n={n}')
        pop_size = self.load_data('popsize')
        self.data['pop_years'] = pop_size.year.to_numpy()
        self.data['pop_size']  = pop_size.population.to_numpy() / (pop_size.population[0] / n)  # Corrected for # of agents, needs manual adjustment for # agents

        # Extract population growth rate
        data_growth_rate = self.pop_growth_rate(self.data['pop_years'], self.data['pop_size'])
        self.data['pop_growth_rate'] = data_growth_rate

        # Extract mcpr over time
        mcpr = self.load_data('mcpr')
        self.data['mcpr_years'] = mcpr.iloc[:,0].to_numpy()
        #self.data['cpr'] = mcpr.iloc[:,1].to_numpy()
        self.data['mcpr'] = mcpr.iloc[:,2].to_numpy()

        self.initialized = True

        return


    def pop_growth_rate(self, years, population):
        growth_rate = np.zeros(len(years) - 1)

        for i in range(len(years)):
            if population[i] == population[-1]:
                break
            growth_rate[i] = ((population[i + 1] - population[i]) / population[i]) * 100

        return growth_rate


    def run_model(self, pars=None, **kwargs):
        ''' Create the sim and run the model '''

        if not self.initialized:
            self.extract_data()

        if pars is None:
            pars = self.pars

        self.sim = fps.Sim(pars=pars, **kwargs)
        self.sim.run()
        self.post_process_sim()

        return


    def post_process_sim(self):
        self.people = self.sim.people  # Extract people objects from sim
        self.model_results = self.sim.results  # Stores dictionary of results

        self.method_keys = list(self.sim['methods']['map'].keys())
        return


    def extract_model(self):
        if self.flags.popsize:  self.model_pop_size()
        if self.flags.mcpr:     self.model_mcpr()
        if self.flags.mmr:      self.model_mmr()
        if self.flags.infant_m: self.model_infant_mortality_rate()
        if self.flags.cdr:      self.model_crude_death_rate()
        if self.flags.cbr:      self.model_crude_birth_rate()
        if self.flags.tfr:      self.model_data_tfr()
        if self.flags.asfr:     self.model_data_asfr()
        return


    def model_pop_size(self):

        self.model['pop_size'] = self.model_results['pop_size']
        self.model['pop_years'] = self.model_results['tfr_years']

        model_growth_rate = self.pop_growth_rate(self.model['pop_years'], self.model['pop_size'])
        self.model['pop_growth_rate'] = model_growth_rate

        return


    def model_mcpr(self):

        model = {'years': self.model_results['t'], 'mcpr': self.model_results['mcpr']}
        model_frame = pd.DataFrame(model)

        # Filter to matching years
        data_years = self.data['mcpr_years'].tolist()
        filtered_model = model_frame.loc[model_frame.years.isin(data_years)]
        model_mcpr = filtered_model['mcpr'].to_numpy()
        mcpr_years = filtered_model['years'].to_numpy()

        self.model['mcpr'] = model_mcpr*100 # Since data is in 100
        self.model['mcpr_years'] = mcpr_years

        return


    def model_mmr(self):
        '''
        Calculate maternal mortality in model over most recent 3 years
        '''

        maternal_deaths = np.sum(self.model_results['maternal_deaths'][-mpy * 3:])
        births_last_3_years = np.sum(self.model_results['births'][-mpy * 3:])
        self.model['maternal_mortality_ratio'] = (maternal_deaths / births_last_3_years) * 100000

        return


    def model_infant_mortality_rate(self):

        infant_deaths = np.sum(self.model_results['infant_deaths'][-mpy:])
        births_last_year = np.sum(self.model_results['births'][-mpy:])
        self.model['infant_mortality_rate'] = (infant_deaths / births_last_year) * 1000

        return


    def model_crude_death_rate(self):
        total_deaths = np.sum(self.model_results['deaths'][-mpy:]) + \
                       np.sum(self.model_results['infant_deaths'][-mpy:]) + \
                       np.sum(self.model_results['maternal_deaths'][-mpy:])
        self.model['crude_death_rate'] = (total_deaths / self.model_results['pop_size'][-1]) * 1000
        return


    def model_crude_birth_rate(self):
        births_last_year = np.sum(self.model_results['births'][-mpy:])
        self.model['crude_birth_rate'] = (births_last_year / self.model_results['pop_size'][-1]) * 1000
        return


    def model_data_tfr(self):

        # Extract tfr over time in data - keep here to ignore dhs data if not using tfr for calibration
        tfr = self.load_data('tfr')  # From DHS
        self.data['tfr_years'] = tfr.iloc[:, 0].to_numpy()
        self.data['total_fertility_rate'] = tfr.iloc[:, 1].to_numpy()

        self.model['tfr_years'] = self.model_results['tfr_years']
        self.model['total_fertility_rate'] = self.model_results['tfr_rates']
        return


    def model_data_asfr(self, ind=-1):

        # Extract ASFR for different age bins
        asfr = self.load_data('asfr')  # From DHS
        age_bins = list(asfr.columns)
        age_bins.remove('year')
        self.data['asfr_bins'] = age_bins

        year_data = asfr[asfr['year'] == self.pars['end_year']]
        self.data['asfr'] = year_data.drop(['year'], axis=1).values.tolist()[0]

        # Model extraction
        age_bins = list(fpd.age_bin_map.keys())
        self.model['asfr_bins'] = age_bins
        self.model['asfr'] = []
        for ab in age_bins:
            val = self.model_results['asfr'][ab][ind] # Only use one index (default: last) CK: TODO: match year automatically
            self.model['asfr'].append(val)

        # Check
        assert self.data['asfr_bins'] == self.model['asfr_bins'], f'ASFR data age bins do not match sim: {sc.strjoin(age_bins)}'

        return


    def extract_ageparity(self):

        # Set up
        age_keys = list(fpd.age_bin_map.keys())[1:]
        age_bins = pl.arange(min_age, max_age, bin_size)
        parity_bins = pl.arange(0, 7)  # Plot up to parity 6
        n_age = len(age_bins)
        n_parity = len(parity_bins)

        # Load data TO NOTE: By default, the dataset that is used for comparison with the model is the last dataset (
        # typically the most recent) in the ageparity file
        sky_raw_data = self.load_data('ageparity')
        dataset = sky_raw_data.iloc[-1]['dataset']
        sky_raw_data = sky_raw_data[sky_raw_data.dataset == dataset]
        # sky_parity = sky_raw_data[2].to_numpy() # Not used currently
        sky_parity = sky_raw_data['parity'].to_numpy()
        sky_props = sky_raw_data['percentage'].to_numpy()
        sky_arr = sc.odict()

        sky_arr['Data'] = pl.zeros((len(age_keys), len(parity_bins)))

        for age, row in sky_raw_data.iterrows():
            if row.age in age_keys and row.parity < n_parity:
                age_ind = age_keys.index(row.age)
                sky_arr['Data'][age_ind, row.parity] = row.percentage

        # Extract from model
        sky_arr['Model'] = pl.zeros((len(age_bins), len(parity_bins)))
        ppl = self.people
        for i in range(len(ppl)):
            if ppl.alive[i] and not ppl.sex[i] and ppl.age[i] >= min_age and ppl.age[i] < max_age:
                age_bin = sc.findinds(age_bins <= ppl.age[i])[-1]
                parity_bin = sc.findinds(parity_bins <= ppl.parity[i])[-1]
                sky_arr['Model'][age_bin, parity_bin] += 1

        # Normalize
        for key in ['Data', 'Model']:
            sky_arr[key] /= sky_arr[key].sum() / 100

        self.data['ageparity'] = sky_arr['Data']
        self.model['ageparity'] = sky_arr['Model']
        self.age_bins = age_bins
        self.parity_bins = parity_bins

        return

    def extract_birth_spacing(self):

        # Set up
        data_afb = self.load_data('afb')
        data_afb = data_afb.sort_values(by='afb')
        data_spaces = self.load_data('spacing')
        data_spaces = data_spaces.sort_values(by='space_mo')
        spacing_bins = sc.odict({'0-12': 0, '12-24': 1, '24-48': 2, '>48': 4})  # Spacing bins in years
        model_age_first = []
        model_spacing = []
        model_spacing_counts = sc.odict().make(keys=spacing_bins.keys(), vals=0.0)
        data_spacing_counts = sc.odict().make(keys=spacing_bins.keys(), vals=0.0)
        ppl = self.people

        # Extract age at first birth and birth spaces from model
        for i in range(len(ppl)):
            if ppl.alive[i] and not ppl.sex[i] and min_age <= ppl.age[i] < max_age:
                if len(ppl.dobs[i]):
                    model_age_first.append(ppl.dobs[i][0])
                if len(ppl.dobs[i]) > 1:
                    for d in range(len(ppl.dobs[i]) - 1):
                        space = ppl.dobs[i][d + 1] - ppl.dobs[i][d]
                        ind = sc.findinds(space > spacing_bins[:])[-1]
                        model_spacing_counts[ind] += 1
                        model_spacing.append(space)

        # Normalize model birth space bin counts to percentages
        model_spacing_counts[:] /= model_spacing_counts[:].sum()
        model_spacing_counts[:] *= 100

        # Extract birth spaces and age at first birth from data
        for i, j in data_spaces.iterrows():
            space = j['space_mo'] / mpy
            ind = sc.findinds(space > spacing_bins[:])[-1]
            data_spacing_counts[ind] += j['Freq']

        # Normalize dat birth space bin counts to percentages
        data_spacing_counts[:] /= data_spacing_counts[:].sum()
        data_spacing_counts[:] *= 100

        # Extract afb and respective weights data
        afb_values = data_afb["afb"].values.tolist()
        afb_weights = data_afb["wt"].values.tolist()

        # Calculate the cumulative weights and total weight of the afb data
        afb_cum_weights = np.cumsum(afb_weights)
        afb_total_weight = afb_cum_weights[-1]

        # Extract birth spacing and respective frequency data
        birth_spacing_values = data_spaces["space_mo"].values.tolist()
        birth_spacing_weights = data_spaces["Freq"].values.tolist()

        # Calculate the cumulative weights and total weight of the birth spacing data
        birth_spacing_cum_weights = np.cumsum(birth_spacing_weights)
        birth_spacing_total_weight = birth_spacing_cum_weights[-1]

        data_spacing_stats = np.array([np.interp((.25 * afb_total_weight), afb_cum_weights, afb_values),
                                       np.interp((.50 * afb_total_weight), afb_cum_weights, afb_values),
                                       np.interp((.75 * afb_total_weight), afb_cum_weights, afb_values)])

        data_age_first_stats = np.array([np.interp((.25 * birth_spacing_total_weight), birth_spacing_cum_weights, birth_spacing_values),
                                       np.interp((.50 * birth_spacing_total_weight), birth_spacing_cum_weights, birth_spacing_values),
                                       np.interp((.75 * birth_spacing_total_weight), birth_spacing_cum_weights, birth_spacing_values)])

        # Save to dictionary
        self.data['spacing_bins'] = np.array(data_spacing_counts.values())
        self.data['spacing_stats'] = data_spacing_stats
        self.data['age_first_stats'] = data_age_first_stats

        try:
            model_spacing_stats = np.array([np.percentile(model_spacing, 25),
                                            np.percentile(model_spacing, 50),
                                            np.percentile(model_spacing, 75)])
            model_age_first_stats = np.array([np.percentile(model_age_first, 25),
                                              np.percentile(model_age_first, 50),
                                              np.percentile(model_age_first, 75)])
        except Exception as E:  # pragma: nocover
            print(f'Could not calculate birth spacing, returning zeros: {E}')
            model_spacing_counts = {k: 0 for k in spacing_bins.keys()}
            model_spacing_stats = np.zeros(data_spacing_stats.shape)
            model_age_first_stats = np.zeros(data_age_first_stats.shape)

        # Save arrays to dictionary
        self.model['spacing_bins'] = np.array(model_spacing_counts.values())
        self.model['spacing_stats'] = model_spacing_stats
        self.model['age_first_stats'] = model_age_first_stats

        return

    def extract_methods(self):
        data_method_counts = sc.odict().make(self.method_keys, vals=0.0)
        model_method_counts = sc.dcp(data_method_counts)

        # Extract from data
        data_methods = self.load_data('methods')
        for index, row in data_methods.iterrows():
            data_method_counts[row['method']] = row['perc']

        # Update data method mix using non-user percentage from 'use' file
        data_use = self.load_data('use')
        data_method_counts['None'] = data_use.loc[0, 'perc']
        use_freq = (data_use.loc[1, 'perc'])/100
        for key, value in data_method_counts.items():
            value /= 100
            if key != 'None':
                value *= use_freq
            data_method_counts.update({key: value})

        # Extract from model
        ppl = self.people
        for i in range(len(ppl)):
            if ppl.alive[i] and not ppl.sex[i] and ppl.age[i] >= min_age and ppl.age[i] < max_age:
                model_method_counts[ppl.method[i]] += 1

        model_method_counts[:] /= model_method_counts[:].sum()

        # Make labels
        data_labels = data_method_counts.keys()
        for d in range(len(data_labels)):
            if data_method_counts[d] > 0.01:
                data_labels[d] = f'{data_labels[d]}: {data_method_counts[d] * 100:0.1f}%'
            else:
                data_labels[d] = ''
        model_labels = model_method_counts.keys()
        for d in range(len(model_labels)):
            if model_method_counts[d] > 0.01:
                model_labels[d] = f'{model_labels[d]}: {model_method_counts[d] * 100:0.1f}%'
            else:
                model_labels[d] = ''

        self.data['method_counts'] = np.array(data_method_counts.values())
        self.model['method_counts'] = np.array(model_method_counts.values())

        return

    def compute_fit(self, *args, **kwargs):
        ''' Compute how good the fit is '''
        data = sc.dcp(self.data)
        try:
            sim = sc.dcp(self.model, die=False) # Sometimes fails with a dict_keys copy error (!)
        except:
            sim = {k:self.model[k] for k in data.keys()}
        for k in data.keys():
            data[k] = sc.promotetoarray(data[k])
            data[k] = data[k].flatten()
            sim[k] = sc.promotetoarray(sim[k])
            sim[k] = sim[k].flatten()
        self.fit = Fit(data, sim, *args, **kwargs)
        pass


    def post_process_results(self, keep_people=False, compute_fit=True, **kwargs):
        ''' Compare the model and the data '''
        self.extract_model()
        if self.flags.ageparity:   self.extract_ageparity()
        if self.flags.birth_space:   self.extract_birth_spacing()
        if self.flags.methods:       self.extract_methods()

        # Remove people, they're large!
        if not keep_people:
            del self.people


        # Compute comparison
        self.df = self.compare()

        # Compute fit
        if compute_fit:
            self.compute_fit(**kwargs)

        return


    def run(self, pars=None, keep_people=False, compute_fit=True, **kwargs):
        ''' Run the model and post-process the results '''
        self.run_model(pars=pars)
        self.post_process_results(keep_people=keep_people, compute_fit=compute_fit, **kwargs)
        return self


    def compare(self):
        ''' Create and print a comparison between model and data '''
        # Check that keys match
        data_keys = self.data.keys()
        model_keys = self.model.keys()
        assert set(data_keys) == set(model_keys), 'Data and model keys do not match'

        # Compare the two
        comparison = []
        for key in data_keys:
            dv = self.data[key] # dv = "Data value"
            mv = self.model[key] # mv = "Model value"
            cmp = sc.objdict(key=key,
                             d_type=type(dv),
                             m_type=type(mv),
                             d_shape=np.shape(dv),
                             m_shape=np.shape(mv),
                             d_val='array',
                             m_val='array')
            if sc.isnumber(dv):
                cmp.d_val = dv
            if sc.isnumber(mv):
                cmp.m_val = mv

            comparison.append(cmp)

        self.comparison_df = pd.DataFrame.from_dict(comparison)
        return self.comparison_df


    def summarize(self, as_df=False):
        '''
        Convert results to a one-number-per-key summary format. Returns summary,
        also saves to self.summary.

        Args:
            as_df (bool): if True, return a dataframe instead of a dict.
        '''
        summary = sc.objdict()
        summary.model = sc.objdict()
        summary.data = sc.objdict()

        data = self.data
        model = self.model
        keys = model.keys()

        # Compare the two
        for key in keys:
            if not (key.endswith('_years') or key.endswith('_bins')):
                dv = data[key] # dv = "Data value"
                mv = model[key] # mv = "Model value"
                if sc.isnumber(mv) and sc.isnumber(dv):
                    summary.data[key] = dv
                    summary.model[key] = mv
                else:
                    summary.data[key+'_mean'] = np.mean(dv)
                    summary.model[key+'_mean'] = np.mean(mv)

        self.summary = summary
        self.summary_df = pd.DataFrame(summary)

        if as_df:
            return self.summary.df
        else:
            return self.summary


    def to_json(self, filename=None, tostring=False, indent=2, verbose=False, **kwargs):
        '''
        Export results as JSON.

        Args:
            filename (str): if None, return string; else, write to file
            tostring (bool): if not writing to file, whether to write to string (alternative is sanitized dictionary)
            indent (int): if writing to file, how many indents to use per nested level
            verbose (bool): detail to print
            kwargs (dict): passed to savejson()

        Returns:
            A unicode string containing a JSON representation of the results,
            or writes the JSON file to disk

        **Examples**::

            json = exp.to_json()
            exp.to_json('results.json')
        '''
        d = self.summarize()
        if filename is None:
            output = sc.jsonify(d, tostring=tostring, indent=indent, verbose=verbose, **kwargs)
        else:
            output = sc.savejson(filename=filename, obj=d, indent=indent, **kwargs)

        return output


    def plot(self, do_show=None, do_save=None, filename='fp_experiment.png', axis_args=None, do_maximize=True):
        ''' Plot the model against the data '''
        data = self.data
        sim = self.model

        # Set up keys structure and remove non-plotted keys
        keys = ['rates'] + list(data.keys())
        rate_keys = ['maternal_mortality_ratio',
                     'infant_mortality_rate',
                     'crude_death_rate',
                     'crude_birth_rate']
        non_calibrated_keys = ['pop_years', 'mcpr_years', 'tfr_years', 'asfr_bins']
        for key in rate_keys + non_calibrated_keys:
            if key in keys:
                keys.remove(key)
        nkeys = len(keys)
        expected = 11
        if nkeys != expected:
            errormsg = f'Number of keys changed -- expected {expected}, actually {nkeys} -- did you use run_model() instead of run()?'
            raise ValueError(errormsg)

        with fpo.with_style():

            fig, axs = pl.subplots(nrows=4, ncols=3)
            pl.subplots_adjust(**sc.mergedicts(dict(bottom=0.05, top=0.97, left=0.05, right=0.97, wspace=0.3, hspace=0.3), axis_args))


            #%% Do the plotting!

            # Rates
            ax = axs[0,0]
            height = 0.4
            n_rates = len(rate_keys)
            y = np.arange(n_rates)
            data_rates = np.array([data[k] for k in rate_keys])
            sim_rates  = np.array([sim[k] for k in rate_keys])
            ax.barh(y=y+height/2, width=data_rates, height=height, align='center', label='Data')
            ax.barh(y=y-height/2, width=sim_rates,  height=height, align='center', label='Sim')
            ax.set_title('Rates')
            ax.set_xlabel('Rate')
            ax.set_yticks(range(n_rates))
            ax.set_yticklabels(rate_keys)
            ax.legend()

            # Population size
            ax = axs[1,0]
            ax.plot(data.pop_years, data.pop_size, 'o', label='Data')
            ax.plot(sim.pop_years,  sim.pop_size,  '-', label='Sim')
            ax.set_title('Population size')
            ax.set_xlabel('Year')
            ax.set_ylabel('Population size')
            ax.legend()

            # Population growth rate
            ax = axs[2,0]
            ax.plot(data.pop_years[:-1], data.pop_growth_rate, 'o', label='Data')
            ax.plot(sim.pop_years[:-1],  sim.pop_growth_rate,  '-', label='Sim')
            ax.set_title('Population growth rate')
            ax.set_xlabel('Year')
            ax.set_ylabel('Population growth rate')
            ax.legend()

            # MCPR
            ax = axs[3,0]
            ax.plot(data.mcpr_years, data.mcpr, 'o', label='Data')
            ax.plot(sim.mcpr_years,  sim.mcpr,  '-', label='Sim')
            ax.set_title('MCPR')
            ax.set_xlabel('Year')
            ax.set_ylabel('Modern contraceptive prevalence rate')
            ax.legend()

            # Data age-parity
            ax = axs[0,1]
            ax.pcolormesh(self.age_bins, self.parity_bins, data.ageparity.transpose(), shading='nearest', cmap='turbo')
            ax.set_aspect(1./ax.get_data_ratio()) # Make square
            ax.set_title('Age-parity plot: data')
            ax.set_xlabel('Age')
            ax.set_ylabel('Parity')

            # Sim age-parity
            ax = axs[1,1]
            ax.pcolormesh(self.age_bins, self.parity_bins, sim.ageparity.transpose(), shading='nearest', cmap='turbo')
            ax.set_aspect(1./ax.get_data_ratio())
            ax.set_title('Age-parity plot: sim')
            ax.set_xlabel('Age')
            ax.set_ylabel('Parity')

            # Spacing bins
            ax = axs[2, 1]
            height = 0.4

            spacing_bins = sc.odict({'0-12': 0, '12-24': 1, '24-48': 2, '>48': 4})  # Spacing bins in years
            n_bins = len(spacing_bins.keys())

            y = np.arange(len(data.spacing_bins))
            ax.barh(y=y+height/2, width=data.spacing_bins, height=height, align='center', label='Data')
            ax.barh(y=y-height/2, width=sim.spacing_bins,  height=height, align='center', label='Sim')
            ax.set_title('Birth spacing bins')
            ax.set_xlabel('Percent of births in each bin')
            ax.set_yticks(range(n_bins))
            ax.set_yticklabels(spacing_bins.keys())
            ax.set_ylabel('Birth space in months')
            ax.legend()

            # Age first stats
            quartile_keys = ['25th %',
                         'Median',
                         '75th %']
            n_quartiles = len(quartile_keys)

            ax = axs[3,1]
            height = 0.4
            y = np.arange(len(data.age_first_stats))
            ax.barh(y=y+height/2, width=data.age_first_stats, height=height, align='center', label='Data')
            ax.barh(y=y-height/2, width=sim.age_first_stats,  height=height, align='center', label='Sim')
            ax.set_title('Age at first birth')
            ax.set_xlabel('Age')
            ax.set_yticks(range(n_quartiles))
            ax.set_yticklabels(quartile_keys)
            ax.legend()


            # Method counts
            ax = axs[2,2]

            height = 0.4
            y = np.arange(len(data.method_counts))
            y1 = y + height/2
            y2 = y - height/2
            ax.barh(y=y1, width=data.method_counts, height=height, align='center', label='Data')
            ax.barh(y=y2, width=sim.method_counts,  height=height, align='center', label='Sim')
            ax.set_yticks(y, self.method_keys)
            ax.set_title('Method counts')
            ax.set_ylabel('Contraceptive method')
            ax.set_xlabel('Rate of use')
            ax.legend()

            # ASFR
            ax = axs[3,2]
            y = np.arange(len(data.asfr))
            y1 = y + height/2
            y2 = y - height/2
            ax.barh(y=y1, width=data.asfr, height=height, align='center', label='Data')
            ax.barh(y=y2, width=sim.asfr,  height=height, align='center', label='Sim')
            ax.set_yticks(y, sim.asfr_bins)
            ax.set_title('Age-specific fertility rate')
            ax.set_ylabel('Age bin')
            ax.set_xlabel('Fertility rate')
            ax.legend()

        # Tidy up
        if do_maximize:
            sc.maximize(fig=fig)

        return fps.tidy_up(fig=fig, do_show=do_show, do_save=do_save, filename=filename)



class Fit(sc.prettyobj):
    '''
    A class for calculating the fit between the model and the data. Note the
    following terminology is used here:

        - fit: nonspecific term for how well the model matches the data
        - difference: the absolute numerical differences between the model and the data (one time series per result)
        - goodness-of-fit: the result of passing the difference through a statistical function, such as mean squared error
        - loss: the goodness-of-fit for each result multiplied by user-specified weights (one time series per result)
        - mismatches: the sum of all the losses (a single scalar value per time series)
        - mismatch: the sum of the mismatches -- this is the value to be minimized during calibration

    Args:
        sim (Sim): the sim object
        weights (dict): the relative weight to place on each result (by default: 10 for deaths, 5 for diagnoses, 1 for everything else)
        keys (list): the keys to use in the calculation
        custom (dict): a custom dictionary of additional data to fit; format is e.g. {'my_output':{'data':[1,2,3], 'sim':[1,2,4], 'weights':2.0}}
        compute (bool): whether to compute the mismatch immediately
        verbose (bool): detail to print
        kwargs (dict): passed to cv.compute_gof() -- see this function for more detail on goodness-of-fit calculation options

    **Example**::

        sim = cv.Sim()
        sim.run()
        fit = sim.compute_fit()
        fit.plot()
    '''

    def __init__(self, data, sim, weights=None, keys=None, custom=None, compute=True, verbose=False, **kwargs):

        # Handle inputs
        self.custom     = sc.mergedicts(custom)
        self.verbose    = verbose
        self.weights    = sc.mergedicts(weights)
        self.gof_kwargs = kwargs

        # Copy data
        self.data = data
        self.sim_results = sim

        # Remove keys that aren't for fitting
        for key in self.data.keys():
            if key.endswith('_years') or key.endswith('_bins'):
                self.data.pop(key)
                self.sim_results.pop(key)
        self.keys = data.keys()

        # These are populated during initialization
        self.inds         = sc.objdict() # To store matching indices between the data and the simulation
        self.inds.sim     = sc.objdict() # For storing matching indices in the sim
        self.inds.data    = sc.objdict() # For storing matching indices in the data
        self.pair         = sc.objdict() # For storing perfectly paired points between the data and the sim
        self.diffs        = sc.objdict() # Differences between pairs
        self.gofs         = sc.objdict() # Goodness-of-fit for differences
        self.losses       = sc.objdict() # Weighted goodness-of-fit
        self.mismatches   = sc.objdict() # Final mismatch values
        self.mismatch     = None # The final value

        if compute:
            self.compute()

        return


    def compute(self):
        ''' Perform all required computations '''
        self.reconcile_inputs() # Find matching values
        self.compute_diffs() # Perform calculations
        self.compute_gofs()
        self.compute_losses()
        self.compute_mismatch()
        return self.mismatch


    def reconcile_inputs(self, verbose=False):
        ''' Find matching keys and indices between the model and the data '''

        data_cols = set(self.data.keys())

        if self.keys is None: # pragma: nocover
            sim_keys = self.sim_results.keys()
            intersection = list(set(sim_keys).intersection(data_cols)) # Find keys in both the sim and data
            self.keys = [key for key in sim_keys if key in intersection and key.startswith('cum_')] # Only keep cumulative keys
            if not len(self.keys):
                errormsg = f'No matches found between simulation result keys ({sim_keys}) and data columns ({data_cols})'
                raise sc.KeyNotFoundError(errormsg)
        mismatches = [key for key in self.keys if key not in data_cols]
        if len(mismatches): # pragma: nocover
            mismatchstr = ', '.join(mismatches)
            errormsg = f'The following requested key(s) were not found in the data: {mismatchstr}'
            raise sc.KeyNotFoundError(errormsg)

        for key in self.keys: # For keys present in both the results and in the data
            self.inds.sim[key]  = []
            self.inds.data[key] = []
            count = -1
            for d, datum in enumerate(self.data[key]):
                count += 1
                if np.isfinite(datum): # TODO: match dates for time series data
                    self.inds.sim[key].append(count)
                    self.inds.data[key].append(count)
            self.inds.sim[key]  = np.array(self.inds.sim[key])
            self.inds.data[key] = np.array(self.inds.data[key])

        # Convert into paired points
        for key in self.keys:
            self.pair[key] = sc.objdict()
            sim_inds = self.inds.sim[key]
            data_inds = self.inds.data[key]
            n_inds = len(sim_inds)
            self.pair[key].sim  = np.zeros(n_inds)
            self.pair[key].data = np.zeros(n_inds)
            for i in range(n_inds):
                try:
                    self.pair[key].sim[i]  = self.sim_results[key][sim_inds[i]]
                    self.pair[key].data[i] = self.data[key][data_inds[i]]
                except Exception:
                    if verbose:
                        print('WARNING: exception at', key, i, len(sim_inds), len(self.pair[key].sim),  len(self.sim_results[key]))

        # Process custom inputs
        self.custom_keys = list(self.custom.keys())
        for key in self.custom.keys(): # pragma: nocover

            # Initialize and do error checking
            custom = self.custom[key]
            c_keys = list(custom.keys())
            if 'sim' not in c_keys or 'data' not in c_keys:
                errormsg = f'Custom input must have "sim" and "data" keys, not {c_keys}'
                raise sc.KeyNotFoundError(errormsg)
            c_data = custom['data']
            c_sim  = custom['sim']
            try:
                assert len(c_data) == len(c_sim)
            except:
                errormsg = f'Custom data and sim must be arrays, and be of the same length: data = {c_data}, sim = {c_sim} could not be processed'
                raise ValueError(errormsg)
            if key in self.pair:
                errormsg = f'You cannot use a custom key "{key}" that matches one of the existing keys: {self.pair.keys()}'
                raise ValueError(errormsg)

            # If all tests pass, simply copy the data
            self.pair[key] = sc.objdict()
            self.pair[key].sim  = c_sim
            self.pair[key].data = c_data

            # Process weight, if available
            wt = custom.get('weight', 1.0) # Attempt to retrieve key 'weight', or use the default if not provided
            wt = custom.get('weights', wt) # ...but also try "weights"
            self.weights[key] = wt # Set the weight

        return


    def compute_diffs(self, absolute=False):
        ''' Find the differences between the sim and the data '''
        for key in self.pair.keys():
            self.diffs[key] = self.pair[key].sim - self.pair[key].data
            if absolute:
                self.diffs[key] = np.abs(self.diffs[key])
        return


    def compute_gofs(self, **kwargs):
        ''' Compute the goodness-of-fit '''
        kwargs = sc.mergedicts(self.gof_kwargs, kwargs)
        for key in self.pair.keys():
            actual    = sc.dcp(self.pair[key].data)
            predicted = sc.dcp(self.pair[key].sim)
            self.gofs[key] = compute_gof(actual, predicted, **kwargs)
        return


    def compute_losses(self):
        ''' Compute the weighted goodness-of-fit '''
        for key in self.gofs.keys():
            if key in self.weights:
                weight = self.weights[key]
                if sc.isiterable(weight): # It's an array
                    len_wt = len(weight)
                    len_sim = self.sim_npts
                    len_match = len(self.gofs[key])
                    if len_wt == len_match: # If the weight already is the right length, do nothing
                        pass
                    elif len_wt == len_sim: # Most typical case: it's the length of the simulation, must trim
                        weight = weight[self.inds.sim[key]] # Trim to matching indices
                    else:
                        errormsg = f'Could not map weight array of length {len_wt} onto simulation of length {len_sim} or data-model matches of length {len_match}'
                        raise ValueError(errormsg)
            else:
                weight = 1.0
            self.losses[key] = self.gofs[key]*weight
        return


    def compute_mismatch(self, use_median=False):
        ''' Compute the final mismatch '''
        for key in self.losses.keys():
            if use_median:
                self.mismatches[key] = np.median(self.losses[key])
            else:
                self.mismatches[key] = np.sum(self.losses[key])
        self.mismatch = self.mismatches[:].sum()
        return self.mismatch


    def plot(self, keys=None, width=0.8, font_size=18, fig_args=None, axis_args=None, plot_args=None, do_show=True):
        '''
        Plot the fit of the model to the data. For each result, plot the data
        and the model; the difference; and the loss (weighted difference). Also
        plots the loss as a function of time.

        Args:
            keys      (list):  which keys to plot (default, all)
            width     (float): bar width
            font_size (float): size of font
            fig_args  (dict):  passed to pl.figure()
            axis_args (dict):  passed to pl.subplots_adjust()
            plot_args (dict):  passed to pl.plot()
            do_show   (bool):  whether to show the plot
        '''

        fig_args  = sc.mergedicts(dict(figsize=(36,22)), fig_args)
        axis_args = sc.mergedicts(dict(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.3, hspace=0.3), axis_args)
        plot_args = sc.mergedicts(dict(lw=4, alpha=0.5, marker='o'), plot_args)
        pl.rcParams['font.size'] = font_size

        if keys is None:
            keys = self.keys + self.custom_keys
        n_keys = len(keys)

        loss_ax = None
        colors = sc.gridcolors(n_keys)
        n_rows = 3

        fig = pl.figure(**fig_args)
        pl.subplots_adjust(**axis_args)
        for k,key in enumerate(keys):
            if key in self.keys: # It's a time series, plot with days and dates
                days      = self.inds.sim[key] # The "days" axis (or not, for custom keys)
                daylabel  = 'Timestep'
            else: #It's custom, we don't know what it is
                days      = np.arange(len(self.losses[key])) # Just use indices
                daylabel  = 'Index'

            pl.subplot(n_rows, n_keys, k+0*n_keys+1)
            pl.plot(days, self.pair[key].data, c='k', label='Data', **plot_args)
            pl.plot(days, self.pair[key].sim, c=colors[k], label='Simulation', **plot_args)
            pl.title(key)
            if k == 0:
                pl.ylabel('Time series (counts)')
                pl.legend()

            pl.subplot(n_rows, n_keys, k+1*n_keys+1)
            pl.bar(days, self.diffs[key], width=width, color=colors[k], label='Difference')
            pl.axhline(0, c='k')
            if k == 0:
                pl.ylabel('Differences (counts)')
                pl.legend()

            loss_ax = pl.subplot(n_rows, n_keys, k+2*n_keys+1, sharey=loss_ax)
            pl.bar(days, self.losses[key], width=width, color=colors[k], label='Losses')
            pl.xlabel(daylabel)
            pl.title(f'Total loss: {self.losses[key].sum():0.3f}')
            if k == 0:
                pl.ylabel('Losses')
                pl.legend()

        if do_show:
            pl.show()

        return fig


def compute_gof(actual, predicted, normalize=True, use_frac=False, use_squared=False, as_scalar='none', eps=1e-9, skestimator=None, **kwargs):
    '''
    Calculate the goodness of fit. By default use normalized absolute error, but
    highly customizable. For example, mean squared error is equivalent to
    setting normalize=False, use_squared=True, as_scalar='mean'.

    Args:
        actual      (arr):   array of actual (data) points
        predicted   (arr):   corresponding array of predicted (model) points
        normalize   (bool):  whether to divide the values by the largest value in either series
        use_frac    (bool):  convert to fractional mismatches rather than absolute
        use_squared (bool):  square the mismatches
        as_scalar   (str):   return as a scalar instead of a time series: choices are sum, mean, median
        eps         (float): to avoid divide-by-zero
        skestimator (str):   if provided, use this scikit-learn estimator instead
        kwargs      (dict):  passed to the scikit-learn estimator

    Returns:
        gofs (arr): array of goodness-of-fit values, or a single value if as_scalar is True

    **Examples**::

        x1 = np.cumsum(np.random.random(100))
        x2 = np.cumsum(np.random.random(100))

        e1 = compute_gof(x1, x2) # Default, normalized absolute error
        e2 = compute_gof(x1, x2, normalize=False, use_frac=False) # Fractional error
        e3 = compute_gof(x1, x2, normalize=False, use_squared=True, as_scalar='mean') # Mean squared error
        e4 = compute_gof(x1, x2, skestimator='mean_squared_error') # Scikit-learn's MSE method
        e5 = compute_gof(x1, x2, as_scalar='median') # Normalized median absolute error -- highly robust
    '''

    # Handle inputs
    actual    = np.array(sc.dcp(actual), dtype=float)
    predicted = np.array(sc.dcp(predicted), dtype=float)

    # Custom estimator is supplied: use that
    if skestimator is not None: # pragma: nocover
        try:
            import sklearn.metrics as sm
            sklearn_gof = getattr(sm, skestimator) # Shortcut to e.g. sklearn.metrics.max_error
        except ImportError as E:
            raise ImportError(f'You must have scikit-learn >=0.22.2 installed: {str(E)}')
        except AttributeError:
            raise AttributeError(f'Estimator {skestimator} is not available; see https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter for options')
        gof = sklearn_gof(actual, predicted, **kwargs)
        return gof

    # Default case: calculate it manually
    else:
        # Key step -- calculate the mismatch!
        gofs = abs(np.array(actual) - np.array(predicted))

        if normalize and not use_frac:
            actual_max = abs(actual).max()
            if actual_max>0:
                gofs /= actual_max

        if use_frac:
            if (actual<0).any() or (predicted<0).any():
                print('Warning: Calculating fractional errors for non-positive quantities is ill-advised!')
            else:
                maxvals = np.maximum(actual, predicted) + eps
                gofs /= maxvals

        if use_squared:
            gofs = gofs**2

        if as_scalar == 'sum':
            gofs = np.sum(gofs)
        elif as_scalar == 'mean':
            gofs = np.mean(gofs)
        elif as_scalar == 'median':
            gofs = np.median(gofs)

        return gofs


def diff_summaries(sim1, sim2, skip_key_diffs=False, output=False, die=False):
    '''
    Compute the difference of the summaries of two FPsim calibration objects, and print any
    values which differ.

    Args:
        sim1 (sim/dict): the calib.summary dictionary, representing a single sim
        sim2 (sim/dict): ditto
        skip_key_diffs (bool): whether to skip keys that don't match between sims
        output (bool): whether to return the output as a string (otherwise print)
        die (bool): whether to raise an exception if the sims don't match
        require_run (bool): require that the simulations have been run

    **Example**::

        c1 = fp.Calibration()
        c2 = fp.Calibration()
        c1.run()
        c2.run()
        fp.diff_summaries(c1.summarize(), c2.summarize())
    '''

    for sim in [sim1, sim2]:
        if not isinstance(sim, dict): # pragma: no cover
            errormsg = f'Cannot compare object of type {type(sim)}, must be a FPsim calib.summary dict'
            raise TypeError(errormsg)

    # Ignore data for now
    sim1 = sim1['model']
    sim2 = sim2['model']

    # Compare keys
    keymatchmsg = ''
    sim1_keys = set(sim1.keys())
    sim2_keys = set(sim2.keys())
    #if sim1_keys !=    _keys and not skip_key_diffs: # pragma: no cover
        #keymatchmsg = "Keys don't match!\n"
       # missing = list(sim1_keys - sim2_keys)
        #extra   = list(sim2_keys - sim1_keys)
        #if missing:
            #keymatchmsg += f'  Missing sim1 keys: {missing}\n'
        #if extra:
            #keymatchmsg += f'  Extra sim2 keys: {extra}\n'

    # Compare values
    valmatchmsg = ''
    mismatches = {}
    for key in sim2.keys(): # To ensure order
        if key in sim1_keys: # If a key is missing, don't count it as a mismatch
            sim1_val = sim1[key] if key in sim1 else 'not present'
            sim2_val = sim2[key] if key in sim2 else 'not present'
            both_nan = sc.isnumber(sim1_val, isnan=True) and sc.isnumber(sim2_val, isnan=True)
            if sim1_val != sim2_val and not both_nan:
                mismatches[key] = {'sim1': sim1_val, 'sim2': sim2_val}

    if len(mismatches): # pragma: nocover
        valmatchmsg = '\nThe following values differ between the two simulations:\n'
        df = pd.DataFrame.from_dict(mismatches).transpose()
        diff   = []
        ratio  = []
        change = []
        small_change = 1e-3 # Define a small change, e.g. a rounding error
        for mdict in mismatches.values():
            old = mdict['sim1']
            new = mdict['sim2']
            numeric = sc.isnumber(sim1_val) and sc.isnumber(sim2_val)
            if numeric and old>0:
                this_diff  = new - old
                this_ratio = new/old
                abs_ratio  = max(this_ratio, sc.safedivide(1.0, this_ratio, np.inf))

                # Set the character to use
                if abs_ratio<small_change:
                    change_char = '≈'
                elif new > old:
                    change_char = '↑'
                elif new < old:
                    change_char = '↓'
                else:
                    errormsg = f'Could not determine relationship between sim1={old} and sim2={new}'
                    raise ValueError(errormsg)

                # Set how many repeats it should have
                repeats = 1
                if abs_ratio >= 1.1:
                    repeats = 2
                if abs_ratio >= 2:
                    repeats = 3
                if abs_ratio >= 10:
                    repeats = 4

                this_change = change_char*repeats
            else: # pragma: no cover
                this_diff   = np.nan
                this_ratio  = np.nan
                this_change = 'N/A'

            diff.append(this_diff)
            ratio.append(this_ratio)
            change.append(this_change)

        df['diff'] = diff
        df['ratio'] = ratio
        for col in ['sim1', 'sim2', 'diff', 'ratio']:
            df[col] = df[col].round(decimals=3)
        df['change'] = change
        valmatchmsg += str(df)

    # Raise an error if mismatches were found
    mismatchmsg = keymatchmsg + valmatchmsg
    if mismatchmsg: # pragma: no cover
        if die:
            raise ValueError(mismatchmsg)
        elif output:
            return mismatchmsg
        else:
            print(mismatchmsg)
    else:
        if not output:
            print('Sims match')
    return
