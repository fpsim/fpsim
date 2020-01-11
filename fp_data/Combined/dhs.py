import os
import itertools
import numpy as np
from multiprocessing import Pool
from pathlib import Path
import pandas as pd

class DHS:

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
        'v007': "Year of survey",
        'v011': "Date of birth (cmc)",
        #'v012': "Respondent's current age",
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


    def __init__(self, foldername, force_read=False, cores=8):
        self.cores = cores
        self.foldername = foldername
        self.force_read = force_read
        Path("cache").mkdir(exist_ok=True)
        self.cachefn = os.path.join('cache', 'dhs.hdf')

        self.results_dir = os.path.join('results', 'DHS')
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)

        self.cache_read()

        self._clean()


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
        print('CASEID:\n', data['caseid'].head())

        data['SurveyName'] = year
        found_keys = []
        for k in self.indicators.keys():
            if k not in data.columns:
                print(f'SurveyName {year} is missing {k}')
            else:
                found_keys.append(k)
        data = data[['SurveyName'] + found_keys]

        values = pd.io.stata.StataReader(filename).value_labels()
        replace_dict = {k: values[k.upper()] if k.upper() in values else values[k] for k in found_keys if k in values or k.upper() in values}
        for k in replace_dict.keys():
            #data[k] = data[k].replace(replace_dict[k]).astype('category')
            if 0 in replace_dict[k]:
                # zero-based
                data[k] = data[k].fillna(-1)
            else:
                # assume one-based
                data[k] = data[k].fillna(0) - 1

            try:
                data[k] = pd.Categorical.from_codes(data[k], categories = [unidecode.unidecode(v[1]) for v in sorted(replace_dict[k].items(), key = lambda x: x[0])] )
            except:
                #print('Difficulty:', year, k, data[k].unique(), replace_dict[k])
                data[k] = data[k].replace(replace_dict[k]).map(str) #.astype('category')
                #print(data[k])

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
        keys = [k for k in self.indicators.keys() if k[:5] == 'v3a08']
        barriers = pd.DataFrame(index=self.yeardict.keys(), columns=keys).fillna(0)
        print(data[keys].describe())
        for year, dat in data.groupby('SurveyName'):
            ret = {}
            wsum = dat['v005'].sum()
            for k in keys:
                #print(year, k, dat[k].unique())  # nan, no, yes, not married

                tmp = dat[[k,'v005']].dropna()
                if tmp.shape[0] == 0:
                    print('NaN', year, k, self.indicators[k], dat[k].unique())
                    continue
                ans = 100 * np.dot(~(tmp[k] == 'no'), tmp['v005']) / tmp['v005'].sum()
                barriers.loc[year, k] = ans
                print('%.2f'%ans, year, k, self.indicators[k], dat[k].unique())


        new_colname_dict = {k: v.split(':')[1][1:] for k,v in self.indicators.items() if k in keys}
        barriers.rename(columns=new_colname_dict, inplace=True)
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

    def _clean(self):
        self.clean = self.data\
            .rename(columns={
                'v312':'Method',
                'v313':'MethodType'
            })

        self.clean.replace(
            {
                'MethodType': {
                    'no method': 'No method',
                    '-1.0': 'No method',
                    'traditional method': 'Traditional',
                    'folkloric method': 'Traditional',
                    'modern method': 'Modern',
                }
            },
            inplace=True
        )
