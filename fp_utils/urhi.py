import os
import itertools
import unidecode
import numpy as np
from multiprocessing import Pool
from pathlib import Path
import pandas as pd

from fp_utils.base import Base

class URHI(Base):

    yeardict = {
        'Baseline':    os.path.join('Baseline', 'SEN_base_wm_20160427.dta'),
        'Midline':     os.path.join('Midline', 'SEN_mid_wm_match_20160609.dta'),
        'Endline':     os.path.join('Endline', 'SEN_end_wm_match_20160505.dta'),
    }

    indicators = {
        'Baseline': {'location_code':'Cluster', 'line':'Line', 'hhnum': 'HHNUM', 'iyear':'Year', 'imon':'Month', 'w102':'Age', 'w208':'Parity', 'method':'Method', 'methodtype':'MethodType', 'wm_allcity_wt':'Weight', 'city':'City', 'unmet_cmw': 'Unmet', 'w512': 'Married'},
        'Midline': {'location_code':'Cluster', 'line':'Line', 'hhnum': 'HHNUM', 'mwiyear':'Year', 'mwimon':'Month', 'mw102':'Age', 'mw208b':'Parity', 'mmethod':'Method', 'mmethodtype':'MethodType', 'mwm_allcity_wt':'Weight', 'mcity':'City', 'munmet_cmw': 'Unmet', 'mw512': 'Married'},
        'Endline': {'location_code':'Cluster', 'line':'Line', 'hhnum': 'HHNUM', 'ewiyear':'Year', 'ewimon':'Month', 'ew102':'Age', 'ew208':'Parity', 'emethod':'Method', 'emethodtype':'MethodType', 'ewoman_weight_6city':'Weight', 'ecity':'City', 'eunmet_cmw': 'Unmet', 'ew512': 'Married'},
    }

    def __init__(self, foldername, force_read=False, cores=8):
        self.cachefn = os.path.join('cache', 'urhi.hdf')
        self.results_dir = os.path.join('results', 'URHI')

        super().__init__(foldername, force_read, cores)

        self.cache_read()
        self._clean()

        surveyname_to_date = self.data.groupby('SurveyName')[['Year', 'Month']].mean().apply(lambda x: x['Year'] + x['Month']/12, axis=1)
        surveyname_to_date.name = 'Date'
        self.data = pd.merge(self.data, surveyname_to_date, on='SurveyName')

        self.create_bins()
        self.data['Survey'] = 'URHI'


    def cache_read(self):
        # Read in the URHI data, from file if necessary or requested
        if self.force_read:
            self.data = self.read()
        else:
            try:
                store = pd.HDFStore(self.cachefn)
                self.data = store['data'] #pd.read_hdf(cachefn, key='data')
                store.close()
            except:
                store.close()
                self.data = self.read()


    def load(self, x):
        year, path = x
        filename = os.path.join(self.foldername, path)
        print(f'Loading {year} from file {filename}')

        fn = Path(filename).resolve().stem
        print(f'File: {filename} ...')
        data = pd.read_stata(filename, convert_categoricals=False)

        data['SurveyName'] = year
        found_keys = []
        for k in self.indicators[year].keys():
            if k not in data.columns:
                print(f'SurveyName {year} is missing {k}')
            else:
                found_keys.append(k)


        data = data[['SurveyName'] + found_keys]

        '''
        for k in replace_map.keys():
            if self.indicators[year][k] == 'Parity':
                print('Skipping parity')
                continue

            if 0 in replace_map[k]:
                # zero-based
                data[k] = data[k].fillna(-1)
            else:
                # assume one-based
                data[k] = data[k].fillna(0) - 1

            try:
                data[k] = pd.Categorical.from_codes(data[k], categories = [unidecode.unidecode(v[1]) for v in sorted(replace_map[k].items(), key = lambda x: x[0])] )
            except:
                print('Difficulty:', year, k, data[k].unique(), replace_map[k])
                data[k] = data[k].replace(replace_map[k]).map(str) #.astype('category')
                print(data[k])
        '''
        def remove_dup_replacement_values(d, rm):
            l = list( rm.values() )
            lsl = list(set(l))
            if len(l) == len(lsl): # No dups
                return d, rm

            print('FIXING DUPs!')
            #print('Unequal lengths', len(l), len(lsl))
            #print('RM:', rm)
            #P = pd.DataFrame({'Keys': list(rm.keys()), 'Values': list(rm.values())})
            #print('PPP:\n', P)

            # Find keys associates with repeated values
            unique = {}
            dup = {}
            for kk,v in rm.items():
                if v not in unique.keys():
                    # New value!
                    unique[v] = kk
                else:
                    dup[kk] = unique[v]

            #print('U:', unique)
            #print('D:', dup)
            #print('Data unique before:', data[k].unique())
            #print('Data unique after:', data[k].replace(dup).unique())

            d = d.replace(dup)
            for kk in dup.keys(): # Could reverse unique
                #print(f'Removing {kk} from replace_map[{k}]')
                del rm[kk]

            return d, rm

        def fill_replacement_keys(d, rm):
            #print( 'U:', sorted(d.unique()) )
            #print( 'RM:', rm )
            #print( 'RM Keys:', list(set(rm.keys())) )
            all_keys_in_replace_map = all([(kk in rm) or (kk==-1) for kk in d.unique()])
            #print(all_keys_in_replace_map)
            if all_keys_in_replace_map: # and largest_data_index > len(d.unique()):
                print('FIXING REPLACEMENT!')
                # OK, we can fix it - just add the missing entries to the replace_map[k], that way codes are preserved
                largest_index = int(d.unique().max())
                #print('LI:', largest_index)
                for i in range(largest_index+1):
                    if i not in rm:
                        rm[i] = f'Dummy{i}'
                return d, rm
            return d, rm


        values = pd.io.stata.StataReader(filename).value_labels()
        replace_map = {k: values[k.upper()] if k.upper() in values else values[k] for k in found_keys if k in values or k.upper() in values}

        # Ugh
        if year == 'Midline':
            replace_map['mmethodtype'] = values['methodtype'] # Doesn't appear to be an entry for mmethodtype?
        elif year == 'Endline':
            replace_map['emethod'] = values['method'] # Doesn't appear to be an entry for emethod?a # emethodvl is ~same (icud vs iud)
            replace_map['emethodtype'] = values['methodtype'] # Doesn't appear to be an entry for emethodtype? # emethodtypevl is ~same (icud vs iud)

        for k in replace_map.keys():
            if self.indicators[year][k] == 'Parity':
                print('Skipping parity')
                continue

            data[k] = data[k].fillna(-1)

            data[k], replace_map[k] = remove_dup_replacement_values(data[k], replace_map[k])
            data[k], replace_map[k] = fill_replacement_keys(data[k], replace_map[k])
            '''
            # -1 should get mapped to NaN in the Categorical below
            if 0 in replace_map[k]:
                # zero-based
                data[k] = data[k].fillna(-1)
            else:
                # assume one-based
                min_key = sorted(replace_map[k].items(), key = lambda x: x[0])[0][0] # Lame
                if min_key == 1:
                    data[k] = data[k].fillna(0) - 1
                else:
                    print('MIN KEY NOT 1:\n', sorted(replace_map[k].items(), key = lambda x: x[0]))
                    print('UNIQUE BEFORE:', data[k].unique())
                    data[k] = data[k].replace(replace_map[k]).map(str) #.astype('category')
                    print('UNIQUE AFTER:', data[k].unique())
                    continue
                '''

            try:
                data[k] = pd.Categorical.from_codes(data[k], categories = [unidecode.unidecode(v[1]) for v in sorted(replace_map[k].items(), key = lambda x: x[0])] )
                print(year, k, [unidecode.unidecode(v[1]) for v in sorted(replace_map[k].items(), key = lambda x: x[0])])
            except Exception as e:
                print('Difficulty:', year, k, data[k].unique(), replace_map[k])
                print('--> ', e)
                print('--> ', 'Handling via simple replacement')
                data[k] = data[k].replace(replace_map[k]).map(str) #.astype('category')

        data.rename(columns=self.indicators[year], inplace=True)

        age_edges = list(range(15,55,5)) + [99]
        a,b = itertools.tee(age_edges)
        a = list(a)[:-1]
        next(b)
        labels = [f'{c}-{d}' for c,d in zip(a,b)]
        data['AgeBin'] = pd.cut(data['Age'], bins = age_edges, labels=labels, right=False)

        parity_edges = list(range(6+1)) + [99]
        a,b = itertools.tee(parity_edges)
        a = list(a)[:-1]
        next(b)
        labels = [f'{c}-{d}' for c,d in zip(a,b)]
        data['ParityBin'] = pd.cut(data['Parity'], bins = parity_edges, labels=labels, right=False)

        if True:
            values = pd.io.stata.StataReader(filename).value_labels()
            codebook = pd.io.stata.StataReader(filename).variable_labels()

            pd.DataFrame({'keys': list(codebook.keys()), 'values': list(codebook.values())}).set_index('keys').to_csv(f'codebook_{fn}.csv')

        return data


    def read(self):
        with Pool(self.cores) as p:
            data_list = p.map(self.load, self.yeardict.items())

        data = pd.concat(data_list)
        data['UID'] = data.apply(lambda x: str(x['Cluster']) + ' ' + str(x['HHNUM']) + ' ' + str(x['Line']), axis=1)
        data.set_index('UID', inplace=True)

        data.drop(['Cluster', 'HHNUM', 'Line'], axis=1, inplace=True)

        print(data.dtypes)
        print(sorted(data['Method'].head(25)))
        print(sorted(data['MethodType'].unique()))
        data.to_hdf(self.cachefn, key='data', format='t')

        return data


    def _clean(self):
        self.raw = self.data

        # Parity is all [NaN or 0] at Midline?!  Causes seaborn ploting problems, so fill -1.
        self.data.loc[self.data['SurveyName']=='Midline', 'Parity'] = \
            self.data.loc[self.data['SurveyName']=='Midline', 'Parity'].fillna(0)

        NO_METHOD = 'No method'
        SHORT = 'Short-term'
        LONG = 'Long-term'
        INJECTION = 'Injection'
        OTHER = 'Other'

        self.data.loc[:,'MethodDurability'] = self.data['Method']

        self.data.loc[:,'Unmet'].fillna('Missing', inplace=True) # Needed for plotting

        self.data.replace(
            {
                'Method': {
                    #'Female sterilization': '',
                    #'No method': '',
                    #'Daily pill': '',

                    'Injectables': 'Injectable',
                    'Breastfeeding/LAM': 'LAM',
                    'iucd': 'IUD',
                    'Female condom': 'Condom',
                    'Male condom': 'Condom',
                    'Implants': 'Implant',

                    'Natural methods': 'Traditional',
                    'Other traditional method': 'Traditional',

                    'sdm': 'Other modern',
                    'Other modern method': 'Other modern',
                    'Emergency pill': 'Other modern',
                },

                'MethodDurability': {
                    #'No method': NO_METHOD, # Causes...  "ValueError: Replacement not allowed with overlapping keys and values"

                    'Daily pill': SHORT,
                    'Female condom': SHORT,
                    'Male condom': SHORT,

                    'Injectables': INJECTION,

                    'Female sterilization': LONG,
                    'iucd': LONG,
                    'Implants': LONG,

                    'Natural methods': OTHER,
                    'Breastfeeding/LAM': OTHER,
                    'Other traditional method': OTHER,
                    'sdm': OTHER,
                    'Other modern method': OTHER,
                    'Emergency pill': OTHER,
                },

                'Unmet': {
                    '-1.0': 'Unknown',
                    'No unmet need': 'No',
                    'Unmet need': 'Yes',
                    'Missing': 'Unknown',
                }
            },
            inplace=True
        )

        self.data.reset_index(inplace=True)
