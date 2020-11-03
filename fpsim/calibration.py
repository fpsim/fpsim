'''
Define classes and functions for fitting (calibration)
'''


import os
import numpy as np
import pylab as pl
import pandas as pd
import sciris as sc
from .model import Sim



# ...more settings
min_age = 15
max_age = 50
bin_size = 5
year_str = '2017'
mpy = 12


# Files
def datapath(path):
    ''' Return the path of the parent folder '''
    return sc.thisdir(__file__, os.pardir, 'dropbox', path)

pregnancy_parity_file = datapath('SNIR80FL.DTA')  # DHS Senegal 2018 file
pop_pyr_year_file = datapath('Population_Pyramid_-_All.csv')
skyscrapers_file = datapath('Skyscrapers-All-DHS.csv')
methods_file = datapath('Method_v312.csv')
spacing_file = datapath('BirthSpacing.csv')
popsize_file = datapath('senegal-popsize.csv')
barriers_file = datapath('DHSIndividualBarriers.csv')
tfr_file = datapath('senegal-tfr.csv')



class Calibration(sc.prettyobj):
    '''
    Class for running calibration to data
    '''

    def __init__(self, flags):
        self.flags = flags # Set flags for what gets run
        self.model_results = sc.odict()
        self.model_people = sc.odict()
        self.model_to_calib = sc.odict()
        self.dhs_data = sc.odict()
        self.method_keys = None

        return

    def init_dhs_data(self):
        '''
        Assign data points of interest in DHS dictionary for Senegal data.  All data 2018 unless otherwise indicated
        Adjust data for a different year or country
        '''

        self.dhs_data['mcpr'] = 19.6  # From FP2020 data  --> Could make this an array to match model, need data
        self.dhs_data[
            'maternal_mortality_ratio'] = 315  # Per 100,000 live births, (2017) From World Bank https://data.worldbank.org/indicator/SH.STA.MMRT?locations=SN
        self.dhs_data['infant_mortality_rate'] = 33.6  # Per 1,000 live births, From World Bank
        self.dhs_data['crude_death_rate'] = 5.7  # Per 1,000 inhabitants, From World Bank
        self.dhs_data['crude_birth_rate'] = 34.5  # Per 1,000 inhabitants, From World Bank
        # self.dhs_data['total_fertility_rate'] = 4.62  # From World Bank -- TODO: see if can match the model

        return

    def extract_dhs_data(self):

        # Extract ages, currently pregnant, and parity in 2018 in dataframe
        dhs_pregnancy_parity = pd.read_stata(pregnancy_parity_file, convert_categoricals=False)
        dhs_pregnancy_parity = dhs_pregnancy_parity[['v012', 'v213', 'v218']]
        dhs_pregnancy_parity = dhs_pregnancy_parity.rename(columns={'v012': 'Age', 'v213': 'Pregnant',
                                                                    'v218': 'Parity'})  # Parity means # of living children in DHS
        self.dhs_data['pregnancy_parity'] = dhs_pregnancy_parity

        # Extract population size over time
        pop_size = pd.read_csv(popsize_file, header=None)  # From World Bank
        self.dhs_data['pop_years'] = pop_size.iloc[0,:].to_numpy()
        self.dhs_data['pop_size'] = pop_size.iloc[1,:].to_numpy()

        # Extract tfr over time
        tfr = pd.read_csv(tfr_file, header = None)  # From DHS
        self.dhs_data['tfr_years'] = tfr.iloc[0,:].to_numpy()
        self.dhs_data['total_fertility_rate'] = tfr.iloc0[1,:].to_numpy()

        return

    def run_model(self, pars):

        self.init_dhs_data()
        self.extract_dhs_data()

        sim = Sim(pars=pars)

        sim.run()
        self.people = list(sim.people.values())  # Extract people objects from sim

        self.model_results = sim.store_results()  # Stores dictionary of results

        # Store dataframe of agent's age, pregnancy status, and parity
        model_pregnancy_parity = sim.store_postpartum()
        model_pregnancy_parity = model_pregnancy_parity.drop(['PP0to5', 'PP6to11', 'PP12to23', 'NonPP'], axis=1)
        self.model_to_calib['pregnancy_parity'] = model_pregnancy_parity

        self.method_keys = sim.pars['methods']['names']

        return

    def extract_model(self):

        if self.flags.popsize:
            self.model_pop_size()
        if self.flags.mcpr:
            self.model_mcpr()
        if self.flags.mmr:
            self.model_mmr()
        if self.flags.infant_m:
            self.model_infant_mortality_rate()
        if self.flags.cdr:
            self.model_crude_death_rate()
        if self.flags.cbr:
            self.model_crude_birth_rate()
        if self.flags.tfr:
            self.model_tfr()

        return

    def model_pop_size(self):

        self.model_to_calib['pop_size'] = self.model_results['pop_size']
        self.model_to_calib['pop_years'] = self.model_results['tfr_years']

        return

    def model_mcpr(self):

        self.model_to_calib['mcpr'] = self.model_results['mcpr']

        return

    def model_mmr(self):
        '''
        Calculate maternal mortality in model over most recent 3 years
        '''

        maternal_deaths = pl.sum(self.model_results['maternal_deaths'][-mpy * 3:])
        births_last_3_years = pl.sum(self.model_results['births'][-mpy * 3:])
        self.model_to_calib['maternal_mortality_ratio'] = (maternal_deaths / births_last_3_years) * 100000

        return

    def model_infant_mortality_rate(self):

        infant_deaths = pl.sum(self.model_results['infant_deaths'][-mpy:])
        births_last_year = pl.sum(self.model_results['births'][-mpy:])
        self.model_to_calib['infant_mortality_rate'] = (infant_deaths / births_last_year) * 1000

        return

    def model_crude_death_rate(self):

        total_deaths = pl.sum(self.model_results['deaths'][-mpy:]) + \
                       pl.sum(self.model_results['infant_deaths'][-mpy:]) + \
                       pl.sum(self.model_results['maternal_deaths'][-mpy:])
        self.model_to_calib['crude_death_rate'] = (total_deaths / self.model_results['pop_size'][-1]) * 1000

        return

    def model_crude_birth_rate(self):

        births_last_year = pl.sum(self.model_results['births'][-mpy:])
        self.model_to_calib['crude_birth_rate'] = (births_last_year / self.model_results['pop_size'][-1]) * 1000

        return

    def model_tfr(self):

        self.model_to_calib['total_fertility_rate'] = self.model_results['tfr']

    def extract_skyscrapers(self):

        # Set up
        min_age = 15
        max_age = 50
        bin_size = 5
        age_bins = pl.arange(min_age, max_age, bin_size)
        parity_bins = pl.arange(0, 8)
        n_age = len(age_bins)
        n_parity = len(parity_bins)
        x_age = pl.arange(n_age)

        # Load data
        data_parity_bins = pl.arange(0, 18)
        sky_raw_data = pd.read_csv(skyscrapers_file, header=None)
        sky_raw_data = sky_raw_data[sky_raw_data[0] == year_str]
        # sky_parity = sky_raw_data[2].to_numpy() # Not used currently
        sky_props = sky_raw_data[3].to_numpy()
        sky_arr = sc.odict()

        sky_arr['Data'] = pl.zeros((len(age_bins), len(parity_bins)))
        count = -1
        for age_bin in x_age:
            for dpb in data_parity_bins:
                count += 1
                parity_bin = min(n_parity - 1, dpb)
                sky_arr['Data'][age_bin, parity_bin] += sky_props[count]
        assert count == len(sky_props) - 1  # Ensure they're the right length

        # Extract from model
        sky_arr['Model'] = pl.zeros((len(age_bins), len(parity_bins)))
        for person in self.people:
            if person.alive and not person.sex and person.age >= min_age and person.age < max_age:
                age_bin = sc.findinds(age_bins <= person.age)[-1]
                parity_bin = sc.findinds(parity_bins <= person.parity)[-1]
                sky_arr['Model'][age_bin, parity_bin] += 1

        # Normalize
        for key in ['Data', 'Model']:
            sky_arr[key] /= sky_arr[key].sum() / 100

        self.dhs_data['skyscrapers'] = sky_arr['Data']
        self.model_to_calib['skyscrapers'] = sky_arr['Model']

        return

    def extract_birth_order_spacing(self):

        spacing_bins = sc.odict({'0-12': 0, '12-24': 1, '24-36': 2, '>36': 3})  # Spacing bins in years

        # From data
        data = pd.read_csv(spacing_file)

        right_year = data['SurveyYear'] == '2017'   #TODO - Should be 2017?
        not_first = data['Birth Order'] != 0
        is_first = data['Birth Order'] == 0
        filtered = data[(right_year) & (not_first)]
        spacing = filtered['Birth Spacing'].to_numpy()
        sorted_spacing = sorted(spacing)

        first_filtered = data[(right_year) & (is_first)]
        first = first_filtered['Birth Spacing'].to_numpy()
        sorted_first = sorted(first)

        #Save to dictionary
        self.dhs_data['spacing'] = sorted_spacing
        self.dhs_data['age_first_birth'] = sorted_first

        # From model
        model_age_first = []
        model_spacing = []
        model_spacing_counts = sc.odict().make(keys=spacing_bins.keys(), vals=0.0)
        for person in self.people:
            if len(person.dobs):
                model_age_first.append(person.dobs[0])
            if len(person.dobs) > 1:
                for d in range(len(person.dobs) - 1):
                    space = person.dobs[d + 1] - person.dobs[d]
                    ind = sc.findinds(space > spacing_bins[:])[-1]
                    model_spacing_counts[ind] += 1

                    model_spacing.append(space)

        # Save arrays to dictionary
        self.model_to_calib['spacing'] = model_spacing
        self.model_to_calib['age_first_birth'] = model_age_first

        return

    def extract_methods(self):

        data_method_counts = sc.odict().make(self.method_keys, vals=0.0)
        model_method_counts = sc.dcp(data_method_counts)

        # Load data from DHS -- from dropbox/Method_v312.csv

        data = [
            ['Other', 'emergency contraception', 0.015216411570543636, 2017.698615635373],
            ['Condoms', 'female condom', 0.005239036180154552, 2017.698615635373],
            ['BTL', 'female sterilization', 0.24609377594176307, 2017.698615635373],
            ['Implants', 'implants/norplant', 5.881839602070953, 2017.698615635373],
            ['Injectables', 'injections', 7.101718239287355, 2017.698615635373],
            ['IUDs', 'iud', 1.4865067612487317, 2017.698615635373],
            ['Other', 'lactational amenorrhea (lam)', 0.04745447091361792, 2017.698615635373],
            ['Condoms', 'male condom', 1.0697377418682412, 2017.698615635373],
            ['None', 'not using', 80.10054235699272, 2017.698615635373],
            ['Other', 'other modern method', 0.007832257135437748, 2017.698615635373],
            ['Other', 'other traditional', 0.5127850142889963, 2017.698615635373],
            ['Rhythm', 'periodic abstinence', 0.393946698444533, 2017.698615635373],
            ['Pill', 'pill', 2.945874450486654, 2017.698615635373],
            ['Rhythm', 'standard days method (sdm)', 0.06132534128612159, 2017.698615635373],
            ['Withdrawal', 'withdrawal', 0.12388784228417069, 2017.698615635373],
        ]

        for entry in data:
            data_method_counts[entry[0]] += entry[2]
        data_method_counts[:] /= data_method_counts[:].sum()

        # From model
        for person in self.people:
            if person.alive and not person.sex and person.age >= min_age and person.age < max_age:
                model_method_counts[person.method] += 1
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

        return

    def run(self, pars):

        self.run_model(pars)
        self.extract_model()
        self.extract_dhs_data()
        if self.flags.skyscrapers:
            self.extract_skyscrapers()
        if self.flags.birth_space:
            self.extract_birth_order_spacing()
        if self.flags.methods:
            self.extract_methods()

        # Remove people, they're large!
        del self.people

        # Store model_to_calib and dhs_data dictionaries in preferred way

        return



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

    def __init__(self, sim, weights=None, keys=None, custom=None, compute=True, verbose=False, **kwargs):

        # Handle inputs
        self.weights    = weights
        self.custom     = sc.mergedicts(custom)
        self.verbose    = verbose
        self.weights    = sc.mergedicts({'cum_deaths':10, 'cum_diagnoses':5}, weights)
        self.keys       = keys
        self.gof_kwargs = kwargs

        # Copy data
        if sim.data is None:
            errormsg = 'Model fit cannot be calculated until data are loaded'
            raise RuntimeError(errormsg)
        self.data = sim.data

        # Copy sim results
        if not sim.results_ready:
            errormsg = 'Model fit cannot be calculated until results are run'
            raise RuntimeError(errormsg)
        self.sim_results = sc.objdict()
        for key in sim.result_keys() + ['t', 'date']:
            self.sim_results[key] = sim.results[key]
        self.sim_npts = sim.npts # Number of time points in the sim

        # Copy other things
        self.sim_dates = sim.datevec.tolist()

        # These are populated during initialization
        self.inds         = sc.objdict() # To store matching indices between the data and the simulation
        self.inds.sim     = sc.objdict() # For storing matching indices in the sim
        self.inds.data    = sc.objdict() # For storing matching indices in the data
        self.date_matches = sc.objdict() # For storing matching dates, largely for plotting
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


    def reconcile_inputs(self):
        ''' Find matching keys and indices between the model and the data '''

        data_cols = self.data.columns
        if self.keys is None:
            sim_keys = self.sim_results.keys()
            intersection = list(set(sim_keys).intersection(data_cols)) # Find keys in both the sim and data
            self.keys = [key for key in sim_keys if key in intersection and key.startswith('cum_')] # Only keep cumulative keys
            if not len(self.keys):
                errormsg = f'No matches found between simulation result keys ({sim_keys}) and data columns ({data_cols})'
                raise sc.KeyNotFoundError(errormsg)
        mismatches = [key for key in self.keys if key not in data_cols]
        if len(mismatches):
            mismatchstr = ', '.join(mismatches)
            errormsg = f'The following requested key(s) were not found in the data: {mismatchstr}'
            raise sc.KeyNotFoundError(errormsg)

        for key in self.keys: # For keys present in both the results and in the data
            self.inds.sim[key]  = []
            self.inds.data[key] = []
            self.date_matches[key] = []
            count = -1
            for d, datum in self.data[key].iteritems():
                count += 1
                if np.isfinite(datum):
                    if d in self.sim_dates:
                        self.date_matches[key].append(d)
                        self.inds.sim[key].append(self.sim_dates.index(d))
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
                self.pair[key].sim[i]  = self.sim_results[key].values[sim_inds[i]]
                self.pair[key].data[i] = self.data[key].values[data_inds[i]]

        # Process custom inputs
        self.custom_keys = list(self.custom.keys())
        for key in self.custom.keys():

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
        n_rows = 4

        figs = [pl.figure(**fig_args)]
        pl.subplots_adjust(**axis_args)
        main_ax1 = pl.subplot(n_rows, 2, 1)
        main_ax2 = pl.subplot(n_rows, 2, 2)
        bottom = sc.objdict() # Keep track of the bottoms for plotting cumulative
        bottom.daily = np.zeros(self.sim_npts)
        bottom.cumul = np.zeros(self.sim_npts)
        for k,key in enumerate(keys):
            if key in self.keys: # It's a time series, plot with days and dates
                days      = self.inds.sim[key] # The "days" axis (or not, for custom keys)
                daylabel  = 'Day'
            else: #It's custom, we don't know what it is
                days      = np.arange(len(self.losses[key])) # Just use indices
                daylabel  = 'Index'

            # Cumulative totals can't mix daily and non-daily inputs, so skip custom keys
            if key in self.keys:
                for i,ax in enumerate([main_ax1, main_ax2]):

                    if i == 0:
                        data = self.losses[key]
                        ylabel = 'Daily mismatch'
                        title = f'Daily total mismatch'
                    else:
                        data = np.cumsum(self.losses[key])
                        ylabel = 'Cumulative mismatch'
                        title = f'Cumulative mismatch: {self.mismatch:0.3f}'

                    dates = self.sim_results['date'][days] # Show these with dates, rather than days, as a reference point
                    ax.bar(dates, data, width=width, bottom=bottom[i][self.inds.sim[key]], color=colors[k], label=f'{key}')

                    if i == 0:
                        bottom.daily[self.inds.sim[key]] += self.losses[key]
                    else:
                        bottom.cumul = np.cumsum(bottom.daily)

                    if k == len(self.keys)-1:
                        ax.set_xlabel('Date')
                        ax.set_ylabel(ylabel)
                        ax.set_title(title)
                        ax.legend()

            pl.subplot(n_rows, n_keys, k+1*n_keys+1)
            pl.plot(days, self.pair[key].data, c='k', label='Data', **plot_args)
            pl.plot(days, self.pair[key].sim, c=colors[k], label='Simulation', **plot_args)
            pl.title(key)
            if k == 0:
                pl.ylabel('Time series (counts)')
                pl.legend()

            pl.subplot(n_rows, n_keys, k+2*n_keys+1)
            pl.bar(days, self.diffs[key], width=width, color=colors[k], label='Difference')
            pl.axhline(0, c='k')
            if k == 0:
                pl.ylabel('Differences (counts)')
                pl.legend()

            loss_ax = pl.subplot(n_rows, n_keys, k+3*n_keys+1, sharey=loss_ax)
            pl.bar(days, self.losses[key], width=width, color=colors[k], label='Losses')
            pl.xlabel(daylabel)
            pl.title(f'Total loss: {self.losses[key].sum():0.3f}')
            if k == 0:
                pl.ylabel('Losses')
                pl.legend()

        if do_show:
            pl.show()

        return figs


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
    if skestimator is not None:
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