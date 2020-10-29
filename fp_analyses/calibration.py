'''
File to extract outputs to calibration from model and compare to data
'''

import pylab as pl
import pandas as pd
import sciris as sc
import lemod_fp as lfp
import senegal_parameters as sp

sc.tic()

'''
CALIBRATION TARGETS:
'''
# ALWAYS ON - Information for scoring needs to be extracted from DataFrame with cols 'Age', 'Parity', 'Currently pregnant'
# Dict key: 'pregnancy_parity'
# Overall age distribution and/or percent of population in each age bin
# Age distribution of agents currently pregnant
# Age distribution of agents in each parity group (can be used to make box plot)
# Percent of reproductive age female population in each parity group

do_save = True # Whether to save the completed calibration

popsize = 1  # Population size and growth over time, adjusted for n number of agents; 'pop_size'
skyscrapers = 1 # Population distribution of agents in each age/parity bin (skyscraper plot); 'skyscrapers'
first_birth = 1  # Age at first birth with standard deviation; 'age_first_birth'
birth_space = 1  # Birth spacing with standard deviation; 'spacing'
mcpr = 1  # Modern contraceptive prevalence; 'mcpr'
methods = 1 # Overall percentage of method use and method use among users; 'methods'
mmr = 1  # Maternal mortality ratio at end of sim in model vs data; 'maternal_mortality_ratio'
infant_m = 1  # Infant mortality rate at end of sim in model vs data; 'infant_mortality_rate'
cdr = 1  # Crude death rate at end of sim in model vs data; 'crude_death_rate'
cbr = 1  # Crude birth rate (per 1000 inhabitants); 'crude_birth_rate'
tfr = 0  # Need to write code for TFR calculation from model - age specific fertility rate to match over time; 'tfr'

pregnancy_parity_file = sp.abspath('dropbox/SNIR80FL.DTA')  # DHS Senegal 2018 file
pop_pyr_year_file = sp.abspath('dropbox/Population_Pyramid_-_All.csv')
skyscrapers_file = sp.abspath('dropbox/Skyscrapers-All-DHS.csv')
methods_file = sp.abspath('dropbox/Method_v312.csv')
spacing_file = sp.abspath('dropbox/BirthSpacing.csv')
popsize_file = sp.abspath('dropbox/senegal-popsize.csv')
barriers_file = sp.abspath('dropbox/DHSIndividualBarriers.csv')

min_age = 15
max_age = 50
bin_size = 5
year_str = '2017'
mpy = 12

class Calibration:
    '''
    Class for running calibration to data
    '''

    def __init__(self):
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
        self.dhs_data['total fertility rate'] = 4.62  # From World Bank

        return

    def extract_dhs_data(self):

        # Extract ages, currently pregnant, and parity in 2018 in dataframe
        dhs_pregnancy_parity = pd.read_stata(pregnancy_parity_file, convert_categoricals=False)
        dhs_pregnancy_parity = dhs_pregnancy_parity[['v012', 'v213', 'v218']]
        dhs_pregnancy_parity = dhs_pregnancy_parity.rename(columns={'v012': 'Age', 'v213': 'Currently pregnant',
                                                                    'v218': 'Parity'})  # Parity means # of living children in DHS
        self.dhs_data['pregnancy_parity'] = dhs_pregnancy_parity

        # Extract population size over time
        pop_size = pd.read_csv(popsize_file, header=None)  # From World Bank
        self.dhs_data['pop_years'] = pop_size.iloc[0, :].to_numpy()
        self.dhs_data['pop_size'] = pop_size.iloc[1, :].to_numpy()

        return

    def run_model(self):

        self.init_dhs_data()
        self.extract_dhs_data()

        pars = sp.make_pars()
        sim = lfp.Sim(pars=pars)

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

        if popsize:
            self.model_pop_size()
        if mcpr:
            self.model_mcpr()
        if mmr:
            self.model_mmr()
        if infant_m:
            self.model_infant_mortality_rate()
        if cdr:
            self.model_crude_death_rate()
        if cbr:
            self.model_crude_birth_rate()
        if tfr:
            self.model_tfr()

        return

    def model_pop_size(self):

        self.model_to_calib['pop_size'] = self.model_results['pop_size']
        self.model_to_calib['pop_years'] = self.model_results['t']

        return

    def model_mcpr(self):

        self.model_to_calib['mcpr'] = self.model_results['mcpr']

        return

    def model_mmr(self):

        maternal_deaths = pl.cumsum(self.model_results['maternal_deaths'][-mpy * 3:])
        births_last_3_years = pl.cumsum(self.model_results['births'][-mpy * 3:])
        self.model_to_calib['maternal_mortality_ratio'] = maternal_deaths[-1] / (births_last_3_years[-1] * 100000)

        return

    def model_infant_mortality_rate(self):

        infant_deaths = pl.cumsum(self.model_results['infant_deaths'][-mpy:])
        births_last_year = pl.cumsum(self.model_results['births'][-mpy:])
        self.model_to_calib['infant_mortality'] = infant_deaths[-1] / (births_last_year[-1] * 1000)

        return

    def model_crude_death_rate(self):

        total_deaths = pl.cumsum(self.model_results['deaths'][-mpy:]) + \
                       pl.cumsum(self.model_results['infant_deaths'][-mpy:]) + \
                       pl.cumsum(self.model_results['maternal_deaths'][-mpy:])
        self.model_to_calib['crude_death_rate'] = total_deaths[-1] / (self.model_results['pop_size'][-1] * 1000)

        return

    def model_crude_birth_rate(self):

        births_last_year = pl.cumsum(self.model_results['births'][-mpy:])
        self.model_to_calib['crude_birth_rate'] = births_last_year[-1] / (self.model_results['pop_size'][-1] * 1000)

        return

    def model_tfr(self):
        pass

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
        sky_parity = sky_raw_data[2].to_numpy()
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

        right_year = data['SurveyYear'] == '2010-11'   #TODO - Should be 2017?
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

    def run(self):

        self.run_model()
        self.extract_model()
        self.extract_dhs_data()
        if skyscrapers:
            self.extract_skyscrapers()
        if birth_space:
            self.extract_birth_order_spacing()
        if methods:
            self.extract_methods()

        # Store model_to_calib and dhs_data dictionaries in preferred way

        return

calibrate = Calibration()
calibrate.run()

if do_save:
    sc.saveobj('senegal_calibration.obj', calibrate)

sc.toc()

print('Done.')
