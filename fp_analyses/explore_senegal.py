

import os
import pylab as pl
import pandas as pd
import sciris as sc
import seaborn as sns
import fpsim as lfp
import senegal_parameters as sp

sc.tic()
pl.rcParams['font.size'] = 10

# Files
def datapath(path):
    ''' Return the path of the parent folder -- TODO: remove duplication with calibration.py'''
    return sc.thisdir(__file__, os.pardir, 'dropbox', path)

pregnancy_parity_file = datapath('SNIR80FL.DTA')  # DHS Senegal 2018 file
pop_pyr_year_file = datapath('Population_Pyramid_-_All.csv')
skyscrapers_file = datapath('Skyscrapers-All-DHS.csv')
methods_file = datapath('Method_v312.csv')
spacing_file = datapath('BirthSpacing.csv')
popsize_file = datapath('senegal-popsize.csv')
barriers_file = datapath('DHSIndividualBarriers.csv')
mcpr_file = datapath('mcpr_senegal.csv')
tfr_file = datapath('senegal-tfr.csv')

tfr = 0
postpartum = 0
demographics = 0
birth_spacing = 1


pars = sp.make_pars()
sim = lfp.Sim(pars=pars)

sim.run()
sim.plot()
results = sim.store_results()  # Stores dictionary of results
people = list(sim.people.values())

if tfr:

    tfr = pd.read_csv(tfr_file, header=None)  # From DHS
    data_tfr_years = tfr.iloc[:,0].to_numpy()
    data_tfr = tfr.iloc[:,1].to_numpy()

    x = results['tfr_years']
    y = results['tfr_rates']

    pl.plot(x, y, label = 'Total fertility rates')
    pl.scatter(data_tfr_years, data_tfr)

    pl.xlabel('Year')
    pl.ylabel('Total fertility rate - children per woman')
    pl.title('Total fertility rate in model compared data - Senegal', fontweight = 'bold')

    pl.show()

if postpartum:

    '''Creates a plot over time of various postpartum states in the model'''

    x = results['t']
    y1 = results['pp0to5']
    y2 = results['pp6to11']
    y3 = results['pp12to23']
    y4 = results['nonpostpartum']

    pl.plot(x, y1, label = 'Postpartum 0-5 months')
    pl.plot(x, y2, label = 'Postpartum 6-11 months')
    pl.plot(x, y3, label = 'Postpartum 12-23 months')
    pl.plot(x, y4, label = 'Non-postpartum')

    pl.xlabel('Year')
    pl.ylabel('Percent')
    pl.legend(loc = 'best')
    pl.title('Percent women age 15-49 in postpartum states', fontweight = 'bold')

    pl.show()

if demographics:

    total_deaths = pl.sum(results['deaths'][-12:]) +\
                    pl.sum(results['infant_deaths'][-12:]) +\
                    pl.sum(results['maternal_deaths'][-12:])
    print(f'Crude death rate per 1,000 inhabitants: {(total_deaths/results["pop_size"][-1])*1000}')
    infant_deaths = pl.sum(results['infant_deaths'][-12:])
    maternal_deaths = pl.sum(results['maternal_deaths'][-36:])
    births_last_year = pl.sum(results['births'][-12:])
    births_last_3_years = pl.sum(results['births'][-36:])
    print(f'Total infant mortality rate in model: {(infant_deaths/births_last_year)*1000}.  Infant mortality rate 2015 Senegal: 36.4')
    print(f'Total maternal death rate in model: {(maternal_deaths/births_last_3_years)*100000}.  Maternal mortality ratio 2015 Senegal: 315 ')
    print(f'Crude birth rate per 1000 inhabitants in model: {(births_last_year/results["pop_size"])*1000}.  Crude birth rate Senegal 2018: 34.52 per 1000 inhabitants')
    print(f'Final percent non-postpartum : {results["nonpostpartum"][-1]}')
    print(f'TFR rates over last 10 years: {results["tfr_rates"][-10:]}.  TFR in Senegal in 2015: 4.84')

if birth_spacing:

        spacing_bins = sc.odict({'0-12': 0, '12-24': 1, '24-48': 2, '>48':4})  # Spacing bins in years

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

        #sb = list(spacing_bins.values())
        #data_spacing_counts = sc.odict().make(keys=spacing_bins.keys(), vals=0.0)

        #self.dhs_data['spacing_counts'] = data_spacing_counts

        #Save to dictionary
        #self.dhs_data['spacing'] = sorted_spacing
        #self.dhs_data['age_first_birth'] = sorted_first

        data_spacing_counts = sc.odict().make(keys=spacing_bins.keys(), vals=0.0)

        # Spacing bins from model
        for s in range(len(spacing)):
            i = sc.findinds(spacing[s] > spacing_bins[:])[-1]
            data_spacing_counts[i] += 1

        data_spacing_counts[:] /= data_spacing_counts[:].sum()
        data_spacing_counts[:] *= 100

        # From model
        model_age_first = []
        model_spacing = []
        model_spacing_counts = sc.odict().make(keys=spacing_bins.keys(), vals=0.0)
        for person in people:
            if len(person.dobs):
                model_age_first.append(person.dobs[0])
            if len(person.dobs) > 1:
                for d in range(len(person.dobs) - 1):
                    space = person.dobs[d + 1] - person.dobs[d]
                    ind = sc.findinds(space > spacing_bins[:])[-1]
                    model_spacing_counts[ind] += 1

                    model_spacing.append(space)

        model_spacing_counts[:] /= model_spacing_counts[:].sum()
        model_spacing_counts[:] *= 100

        print(f'\Data Spacing Counts: {data_spacing_counts}.')
        print(f'\Model Spacing Counts: {model_spacing_counts}')


