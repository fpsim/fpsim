import os
import unidecode
import itertools
import numpy as np
from multiprocessing import Pool
from pathlib import Path
import pandas as pd

from fp_utils.base import Base

class DHS(Base):

    yeardict = {
        '1986':     os.path.join('1986',    'SNIR02DT', 'SNIR02FL.DTA'), # Missing v025
        '1992-93':  os.path.join('1992-93', 'SNIR21DT', 'SNIR21FL.DTA'),
        '1997':     os.path.join('1997',    'SNIR32DT', 'SNIR32FL.DTA'),
        #'1999':     os.path.join('1999',    'SNIQ41DT', 'SNIQ41FL.DTA'),
        '2005':     os.path.join('2005',    'SNIR4HDT', 'SNIR4HFL.dta'),
        '2010-11':  os.path.join('2010-11', 'SNIR61DT', 'SNIR61FL.DTA'),
        '2012-13':  os.path.join('2012-13', 'SNIR6DDT', 'SNIR6DFL.DTA'),
        '2014':     os.path.join('2014',    'SNIR70DT', 'SNIR70FL.DTA'), # SNIR6RFL
        '2015':     os.path.join('2015',    'SNIR7HDT', 'SNIR7HFL.DTA'),
        '2016':     os.path.join('2016',    'SNIR7IDT', 'SNIR7IFL.DTA'), # SNIRG0FL
        '2017':     os.path.join('2017',    'SNIR7ZDT', 'SNIR7ZFL.DTA'),
    }

    indicators = {
        'caseid': 'Case identification',
        'v005': "Weight",
        'v006': "Month of survey",
        'v007': "Year of survey",
        'v011': "Date of birth (cmc)",
        'v012': "Respondent's current age",
        'v013': "Age in 5-year groups",
        'v008': "Date of interview (cmc)",
        'v101': "De facto region of residence",
        'v102': "De facto type of place of residence",
        #'v106': "Highest educational level",
        'v190': "wealth index combined",
        'v201': "total children ever born",
        'v213': "Currently pregnant",
        'v225': "current pregnancy wanted",
        'v228': "ever had a terminated pregnancy",
        'v301': "knowledge of any method # Lists most sophisticate method type",
        #'v304 - v307 is the contraceptive table, interesting!",
        'v312': "current contraceptive method",
        'v313': "current use by method type",
        'v317': "date of start of use of method (cmc)",
        'v364': "contraceptive use and intention",
        'v367': "wanted last child",
        'v384a': "heard family planning on radio last few months",
        'v384b': "heard family planning on tv last few months",
        'v384c': "heard family planning in newspaper/magazine last few months",
            #'v384d': "heard family planning by text messages on mobile phone", # Missing in 1984!
        #'v393': "visited by fieldworker in last 12 months",
        #'v393a': "did fieldworker talk about family planning",
        #'v394': "visited health facility last 12 months",
        #'v395': "at health facility, told of family planning",
        'v3a08a': "reason not using: not married",
        'v3a08b': "reason not using: not having sex",
        'v3a08c': "reason not using: infrequent sex",
        'v3a08d': "reason not using: menopausal/hysterectomy",
        'v3a08e': "reason not using: subfecund/infecund",
        'v3a08f': "reason not using: postpartum amenorrheic",
        'v3a08g': "reason not using: breastfeeding",
        'v3a08h': "reason not using: fatalistic",
        'v3a08i': "reason not using: respondent opposed",
        'v3a08j': "reason not using: husband/partner opposed",
        'v3a08k': "reason not using: others opposed",
        'v3a08l': "reason not using: religious prohibition",
        'v3a08m': "reason not using: knows no method",
        'v3a08n': "reason not using: knows no source",
        'v3a08o': "na-reason not using: health concerns",
        'v3a08p': "reason not using: fear of side effects/health concerns",
        'v3a08q': "reason not using: lack of access/too far",
        'v3a08r': "reason not using: costs too much",
        'v3a08s': "reason not using: inconvenient to use",
        'v3a08t': "reason not using: interferes with body's processes",
        'v3a08u': "reason not using: preferred method not available",
        'v3a08v': "reason not using: no method available",
        'v3a08w': "na-reason not using: cs",
        'v3a08aa': "na-reason not using: cs",
        'v3a08ab': "na-reason not using: cs",
        'v3a08ac': "na-reason not using: cs",
        'v3a08ad': "na-reason not using: cs",
        'v3a08x': "reason not using: other",
        'v3a08z': "reason not using: don't know",

        'v404': "currently breastfeeding",
        'v405': "currently amenorrheic",
        'v406': "currently abstaining",

        'v501': "current marital status",
        'v525': "age at first sex",
        'v602': "fertility preference",
        #'v603': "preferred waiting time for birth of a/another child", # Requires special handling because some -1, some numeric, and some from dictionary {996: 'non-numeric', 997: 'inconsistent', 998: "don't know"}
        'v604': "preferred waiting time for birth of a/another child (grouped)",
        'v605': "desire for more children",
        #'v613': "ideal number of children", # Requires special handling because some numeric and others from {96: 'as god wills', 97: 'other', 98: 'don t know'}
        'v614': "ideal number of children (grouped)",

        'v623': "exposure",
        'v624': "unmet need",
        'v625': "exposure (definition 2)",
        'v626': "unmet need (definition 2)",
        'v625a': "exposure to need for contraception (definition 3)",
        'v626a': "unmet need for contraception (definition 3)",

        # vcol_1 vcol_2 vcol_3 vcol_4 vcol_5 vcol_6 vcol_7 vcol_8 vcol_9 vcal_1 vcal_2 vcal_3 vcal_4 vcal_5 vcal_6 vcal_7 vcal_8 vcal_9

        'b3_01': 'date of birth (cmc)',
        'b3_02': 'date of birth (cmc)',
        'b3_03': 'date of birth (cmc)',
        'b3_04': 'date of birth (cmc)',
        'b3_05': 'date of birth (cmc)',
        'b3_06': 'date of birth (cmc)',
        'b3_07': 'date of birth (cmc)',
        'b3_08': 'date of birth (cmc)',
        'b3_09': 'date of birth (cmc)',
        'b3_10': 'date of birth (cmc)',
        'b3_11': 'date of birth (cmc)',
        'b3_12': 'date of birth (cmc)',
        'b3_13': 'date of birth (cmc)',
        'b3_14': 'date of birth (cmc)',
        'b3_15': 'date of birth (cmc)',
        'b3_16': 'date of birth (cmc)',
        'b3_17': 'date of birth (cmc)',
        'b3_18': 'date of birth (cmc)',
        'b3_19': 'date of birth (cmc)',
        'b3_20': 'date of birth (cmc)',
    }


    def __init__(self, foldername, force_read=False, cores=4):

        self.barrier_keys = [k for k in self.indicators.keys() if k[:5] == 'v3a08']
        self.barrier_map = {k: v.split(':')[1][1:] for k,v in self.indicators.items() if k in self.barrier_keys}

        self.cachefn = os.path.join('cache', 'dhs.hdf')
        self.results_dir = os.path.join('results', 'DHS')

        super().__init__(foldername, force_read, cores)

        self.cache_read()

        self._clean()

        year_to_date = self.data.groupby('SurveyName')['SurveyYear', 'InterviewDateCMC'].mean().apply(self.cmc_to_year, axis=1) # mean CMC
        year_to_date.name = 'Date'
        self.data = pd.merge(self.data, year_to_date, on='SurveyName')

        self.create_bins()
        self.data['Survey'] = 'DHS'

        print('UR:\n', self.data['v102'].unique())

        self.dakar_urban = self.data.loc[ (self.data['v101'].isin(['dakar'])) & (self.data['v102'] == 'urban') ]
        self.dakar_urban.loc[:,'Survey'] = 'DHS: Dakar-urban'
        self.urhi_like = self.data.loc[ (self.data['v101'].isin(['west', 'dakar', 'kaolack', 'thies'])) & (self.data['v102'] == 'urban') ]
        self.urhi_like.loc[:,'Survey'] = 'DHS: URHI-like'
        self.urban = self.data.loc[ (self.data['v102'] == 'urban') ]
        self.urban.loc[:,'Survey'] = 'DHS: Urban'
        print('UR:\n', self.data['v102'].unique())
        self.rural = self.data.loc[ (self.data['v102'] == 'rural') ]
        self.rural.loc[:,'Survey'] = 'DHS: Rural'


    def cache_read(self):
        # Read in the DHS data, from file if necessary or requested
        if self.force_read:
            self.data, self.barrier, self.birth_spacing = self.read()
        else:
            try:
                store = pd.HDFStore(self.cachefn)
                self.data = store['data'] #pd.read_hdf(cachefn, key='data')
                self.barriers = store['barriers'] #pd.read_hdf(cachefn, key='barriers')
                self.birth_spacing = store['birth_spacing'] #pd.read_hdf(cachefn, key='birth_spacing')
                store.close()
            except:
                store.close()
                self.data, self.barrier, self.birth_spacing = self.read()



    def load(self, x):
        year, path = x
        filename = os.path.join(self.foldername, path)
        print(f'Loading {year} from file {filename}')

        fn = Path(filename).resolve().stem
        print(f'File: {filename} ...')
        data = pd.read_stata(filename, convert_categoricals=False)

        data['SurveyName'] = year
        found_keys = []
        for k in self.indicators.keys():
            if k not in data.columns:
                print(f'SurveyName {year} is missing {k}')
            else:
                found_keys.append(k)
        data = data[['SurveyName'] + found_keys]

        def remove_dup_replacement_values(d, rm):
            l = list( rm.values() )
            lsl = list(set(l))
            if len(l) == len(lsl): # No dups
                return d, rm

            #print('FIXING DUPs!')
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
                #print('FIXING REPLACEMENT!')
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
        for k in replace_map.keys():
            #data[k] = data[k].replace(replace_map[k]).astype('category')

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
            except Exception as e:
                print('Difficulty:', year, k, data[k].unique(), replace_map[k])
                print('--> ', e)
                print('--> ', 'Handling via simple replacement')
                data[k] = data[k].replace(replace_map[k]).map(str) #.astype('category')


        if False: # TODO: Make argparse flag
            values = pd.io.stata.StataReader(filename).value_labels()
            codebook = pd.io.stata.StataReader(filename).variable_labels()
            pd.DataFrame({'keys': list(codebook.keys()), 'values': list(codebook.values())}).set_index('keys').to_csv(f'codebook_{fn}.csv')

        return data


    def read(self):
        with Pool(self.cores) as p:
            data_list = p.map(self.load, self.yeardict.items())

        data = pd.concat(data_list).set_index('caseid')
        data.to_hdf(self.cachefn, key='data', format='t')

        # Build barriers from reasons for non-use questions
        barriers = pd.DataFrame(index=self.yeardict.keys(), columns=self.barrier_keys).fillna(0)
        for year, dat in data.groupby('SurveyName'):
            for k in self.barrier_keys:
                #print(year, k, dat[k].unique())  # nan, no, yes, not married

                tmp = dat[[k,'v005']].dropna()
                if tmp.shape[0] == 0:
                    print('NaN', year, k, self.indicators[k], dat[k].unique())
                    continue
                ans = 100 * np.dot(~(tmp[k] == 'no'), tmp['v005']) / tmp['v005'].sum()
                barriers.loc[year, k] = ans
                print('%.2f'%ans, year, k, self.indicators[k], dat[k].unique())

        barriers.rename(columns=self.barrier_map, inplace=True)
        barriers = barriers.stack()
        barriers.name = 'Percent'
        barriers = barriers.reset_index().rename(columns={'level_0':'SurveyName','level_1':'Barrier'})

        barriers.to_hdf(self.cachefn, key='barriers', format='t')


        # BIRTH SPACING
        birth_keys = [f'b3_{i:02}' for i in range(20,0,-1)]
        data_births = data.loc[data['v201'] > 0]
        birth_spacing = np.full((data_births.shape[0],20), fill_value = np.NaN)
        for i, (caseid, d) in enumerate(data_births.iterrows()):
            a,b = itertools.tee(d[['v011'] + birth_keys].dropna())
            a = list(a)[:-1]
            next(b)
            bs = np.array([b-a for i, (a,b) in enumerate(zip(a,b))])
            birth_spacing[i,:len(bs)] = bs
        birth_spacing = pd.DataFrame(birth_spacing, columns = list(range(20)))
        birth_spacing.index = data_births.index

        birth_spacing.to_hdf(self.cachefn, key='birth_spacing', format='t')

        return data, barriers, birth_spacing


    def compute_individual_barriers(self):
        # INDIVIDUAL BARRIERS
        barrier_map = {
            "not married": "No need",
            "not having sex": "No need",
            "infrequent sex": "No need",
            "menopausal/hysterectomy": "No need",
            "subfecund/infecund": "No need",
            "postpartum amenorrheic": "No need",
            "breastfeeding": "No need",
            "fatalistic": "Opposition",
            "respondent opposed": "Opposition",
            "husband/partner opposed": "Opposition",
            "others opposed": "Opposition",
            "religious prohibition": "Opposition",
            "knows no method": "Knowledge",
            "knows no source": "Access",
            "health concerns": "Health",
            "fear of side effects/health concerns": "Health",
            "lack of access/too far": "Access",
            "costs too much": "Access",
            "inconvenient to use": "Health",
            "interferes with body's processes": "Health",
            "preferred method not available": "Access",
            "no method available": "Access",
            "cs": "N/A",
            "cs": "N/A",
            "cs": "N/A",
            "cs": "N/A",
            "cs": "N/A",
            "other": "N/A",
            "don't know": "N/A",
        }

        coarse_barrier_map = {k:barrier_map[v] for k,v in self.barrier_map.items()}
        dat = self.data.set_index(['UID', 'SurveyName', 'Age', 'Parity', 'Weight', 'Unmet', 'Method', 'MethodType'])

        dat = dat.loc[dat.index.get_level_values('MethodType') == 'No method'] # Only keep barriers for "No method"

        coarse_barriers = list(set(coarse_barrier_map.values()))

        # Code goes row-by-row, very slow and inefficient
        # Due to handling of NaN below
        def dan_sum(d):
            e = d.rename(coarse_barrier_map)
            ans = np.zeros(6)
            for i, cb in enumerate(coarse_barriers):
                vec = e.loc[[cb]]
                if all(vec.isna()):
                    #All NaN, take NaN
                    ans[i] = np.NaN
                else:
                    #Some non-nan, taking sum of number of entries outside set ['no', '-1.0', np.NaN]
                    ans[i] = (~(vec.isin(['no', '-1.0', np.NaN]))).sum()
            ret = pd.Series(ans, index=coarse_barriers)
            return ret

        individual_barriers = dat[self.barrier_keys].apply(dan_sum, axis=1)

        # Might as well cache
        individual_barriers.to_hdf(self.cachefn, key='individual_barriers', format='t')

        return individual_barriers


    def _clean(self):
        self.raw = self.data
        self.data = self.raw.reset_index()\
            .rename(columns = {
                'v312': 'Method',
                'v313': 'MethodType',
                'v005': 'Weight',
                'v006': 'SurveyMonth',
                'v007': 'SurveyYear',
                'v008': 'InterviewDateCMC',
                'v012': 'Age',
                'v201': 'Parity',
                'v624': 'Unmet',
                'caseid': 'UID',
            })

        self.data.loc[:,'MethodDurability'] = self.data['Method']

        NO_METHOD = 'No method'
        SHORT = 'Short-term'
        LONG = 'Long-term'
        INJECTION = 'Injection'
        OTHER = 'Other'

        self.data.replace(
            {
                'MethodType': {
                    'no method': 'No method',
                    '-1.0': 'No method',
                    'traditional method': 'Traditional',
                    'folkloric method': 'Traditional',
                    'modern method': 'Modern',
                },
                'Method': {
                    '-1.0': 'No method',
                    'not using': 'No method',
                    'iud': 'IUD',
                    'pill': 'Daily pill',
                    'condom': 'Condom',
                    'injections': 'Injectable',
                    'lactational amenorrhea': 'LAM',
                    'norplant': 'Implant',
                    'implants/norplant': 'Implant',
                    'lactational amenorrhea (lam)': 'LAM',
                    'male condom': 'Condom',
                    'female condom': 'Condom',
                    'female sterilization': 'Female sterilization',
                    'male sterilization': 'Male sterilization',

                    'other traditional': 'Traditional',
                    'withdrawal': 'Traditional',
                    'gris-gris': 'Traditional',
                    'gris - gris': 'Traditional',
                    'abstinence': 'Traditional',
                    'periodic abstinence': 'Traditional',
                    'other': 'Traditional',
                    'medicinal plants': 'Traditional',

                    'other modern method': 'Other modern',
                    'standard days method (sdm)': 'Other modern',
                    'diaphragm': 'Other modern',
                    'emergency contraception': 'Other modern',
                    'foam or jelly': 'Other modern',
                    'diaphragm/foam/jelly': 'Other modern',
                    'diaphragm /foam/jelly': 'Other modern',
                    'oher modern method': 'Other modern', # Yes, oher
                    'collier (cs)': 'Other modern',
                },
                'MethodDurability': {
                    '-1.0': NO_METHOD,
                    'not using': NO_METHOD,
                    'gris-gris': NO_METHOD, # Short?
                    'gris - gris': NO_METHOD, # Short?
                    'abstinence': NO_METHOD, # Short?
                    'periodic abstinence': NO_METHOD, # Short?
                    'medicinal plants': NO_METHOD, # Short?

                    'male condom': SHORT,
                    'female condom':  SHORT,
                    'diaphragm/foam/jelly': SHORT,
                    'diaphragm /foam/jelly': SHORT,
                    'diaphragm': SHORT,
                    'foam or jelly': SHORT,
                    'pill': SHORT,
                    'condom': SHORT,

                    'iud': LONG,
                    'norplant': LONG,
                    'implants/norplant': LONG,
                    'female sterilization': LONG,
                    'male sterilization': LONG,

                    'injections': INJECTION,

                    'standard days method (sdm)': OTHER, # Could be SHORT
                    'collier (cs)': OTHER, # Maybe could be SHORT?
                    'withdrawal': OTHER,
                    'lactational amenorrhea': OTHER,
                    'other traditional': OTHER,
                    'other': OTHER,
                    'lactational amenorrhea (lam)': OTHER,
                    'other modern method': OTHER,
                    'emergency contraception': OTHER,
                    'oher modern method': OTHER, # Yes, oher
                },
                'Unmet': {
                    np.NaN: 'Unknown',
                    '0.0': 'No', # Likely never had sex.
                    'using to space': 'No',
                    'desire birth < 2 yrs': 'No',
                    '-1.0': 'Unknown',
                    'limiting failure': 'No',
                    'unmet need to space': 'Yes',
                    'unmet need to limit': 'Yes',
                    'infecund, menopausal': 'No',
                    'never had sex': 'No',
                    'no sex, want to wait': 'No',
                    'using to limit': 'No',
                    'no unmet need': 'No',
                    'unmet need for limiting': 'Yes',
                    'unmet need for spacing': 'Yes',
                    'using for limiting': 'No',
                    'not married and no sex in last 30 days': 'No', # Maybe Yes?
                    'using for spacing': 'No',
                    'spacing failure': 'No',
                }
            },
            inplace=True
        )
