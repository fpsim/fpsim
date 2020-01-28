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
        'Midline':     os.path.join('Midline',  'SEN_mid_wm_match_20160609.dta'),
        'Endline':     os.path.join('Endline',  'SEN_end_wm_match_20160505.dta'),
    }

    shared_indicators = {
        'location_code':    'Cluster',
        'line':             'Line',
        'hhnum':            'HHNUM',
    }

    indicators = {
        'Baseline': {
            'iyear':               'Year',
            'imon':                'Month',
            'w102':                Base.AGE,
            'w208':                Base.PARITY,
            'method':              Base.METHOD,
            'methodtype':          Base.METHODTYPE,
            'wm_allcity_wt':       Base.WEIGHT,
            'city':                'City',
            'unmet_cmw':           Base.UNMET,
            'w512':                Base.MARRIED,

            'w347a':  "Raison n'utilise pas la PF: Peu/pas de rapports sexuels",
            'w347b':  "Raison n'utilise pas la PF: Partenaire/mari absent",
            'w347c':  "Raison n'utilise pas la PF: Ménopause/hystérectomie",
            'w347d':  "Raison n'utilise pas la PF: Déjà enceinte",
            'w347e':  "Raison n'utilise pas la PF: Allaitement",
            'w347f':  "Raison n'utilise pas la PF: Ne peut pas avoir d'enfants",
            'w347g':  "Raison n'utilise pas la PF: Souhaite avoir autant d'enfants que possible",
            'w347h':  "Raison n'utilise pas la PF: Souhaite tomber/essaye de tomber enceinte",
            'w347i':  "Raison n'utilise pas la PF: Amenorrhée postpartum",
            'w347j':  "Raison n'utilise pas la PF: L__enquêtée est opposée",
            'w347k':  "Raison n'utilise pas la PF: Le partenaire est opposée",
            'w347l':  "Raison n'utilise pas la PF: D__autres personnes sont opposées",
            'w347m':  "Raison n'utilise pas la PF: Interdiction religieuse",
            'w347n':  "Raison n'utilise pas la PF: Ne sait pas comment utiliser une methode",
            'w347o':  "Raison n'utilise pas la PF: Ne connait aucune source",
            'w347p':  "Raison n'utilise pas la PF: Problèmes de santé",
            'w347q':  "Raison n'utilise pas la PF: Peur des effets secondaires",
            'w347r':  "Raison n'utilise pas la PF: Manque d'accès/trop éloigné",
            'w347s':  "Raison n'utilise pas la PF: Coûte trop chère",
            'w347t':  "Raison n'utilise pas la PF: Pas pratique à utiliser",
            'w347u':  "Raison n'utilise pas la PF: N__aime pas les méthodes existantes",
            'w347v':  "Raison n'utilise pas la PF: Expérience malheureuse avec méth. existantes",
            'w347w':  "Raison n'utilise pas la PF: Fataliste: dépend de Dieu",
            'w347x':  "Raison n'utilise pas la PF: Autre réponse",
            'w347z':  "Raison n'utilise pas la PF: Ne sait pas",
            'w347xs': "Raison n'utilise pas la PF: Autre réponse spécifiée",
            **shared_indicators
        },
        'Midline': {
            'mwiyear':             'Year',
            'mwimon':              'Month',
            'mw102':               Base.AGE,
            'mw208b':              Base.PARITY,
            'mmethod':             Base.METHOD,
            'mmethodtype':         Base.METHODTYPE,
            'mwm_allcity_wt':      Base.WEIGHT,
            'mcity':               'City',
            'munmet_cmw':          Base.UNMET,
            'mw512':               Base.MARRIED,


            'mw318a':   "Raison n'utilise pas la PF: Pas de rapports sexuels",
            'mw318b':   "Raison n'utilise pas la PF: Rapports sexuels non fréquents",
            'mw318c':   "Raison n'utilise pas la PF: Pas encore mariée/pas de partenaire",
            'mw318d':   "Raison n'utilise pas la PF: Partenaire/mari absent",
            'mw318e':   "Raison n'utilise pas la PF: Déjà enceinte",
            'mw318f':   "Raison n'utilise pas la PF: Allaitement",
            'mw318g':   "Raison n'utilise pas la PF: Récemment eu un bébé",
            'mw318h':   "Raison n'utilise pas la PF: Veut plus d'enfant/tomber enceinte",
            'mw318i':   "Raison n'utilise pas la PF: Ménopause/hystérectomie",
            'mw318j':   "Raison n'utilise pas la PF: Ne peut pas avoir d'enfants",
            'mw318k':   "Raison n'utilise pas la PF: L__enquêtée est opposée",
            'mw318l':   "Raison n'utilise pas la PF: La partenaire est opposée",
            'mw318m':   "Raison n'utilise pas la PF: D__autres personnes sont opposées",
            'mw318n':   "Raison n'utilise pas la PF: Interdiction religieuse",
            'mw318o':   "Raison n'utilise pas la PF: Ne sait pas quelle méthode utiliser",
            'mw318p':   "Raison n'utilise pas la PF: Ne sait pas comment utiliser une methode",
            'mw318q':   "Raison n'utilise pas la PF: Ne connait aucune source",
            'mw318r':   "Raison n'utilise pas la PF: Problèmes de santé",
            'mw318s':   "Raison n'utilise pas la PF: Peur des effets secondaires",
            'mw318t':   "Raison n'utilise pas la PF: Manque d'accès/trop éloigné",
            'mw318u':   "Raison n'utilise pas la PF: Coûte trop chère",
            'mw318v':   "Raison n'utilise pas la PF: Pas pratique à utiliser",
            'mw318w':   "Raison n'utilise pas la PF: N__aime pas les méthodes existantes",
            'mw318x':   "Raison n'utilise pas la PF: Expérience malheureuse avec méth. existantes",
            'mw318y':   "Raison n'utilise pas la PF: Fataliste: dépend de Dieu",
            'mw318ww':  "Raison n'utilise pas la PF: Autre réponse #1",
            'mw318xx':  "Raison n'utilise pas la PF: Autre réponse #2",
            'mw318yy':  "Raison n'utilise pas la PF: Autre réponse #3",
            'mw318zz':  "Raison n'utilise pas la PF: Ne sait pas",
            'mw318wws': "Raison n'utilise pas la PF: Autre réponse spécifiée #1",
            'mw318xxs': "Raison n'utilise pas la PF: Autre réponse spécifiée #2",
            'mw318yys': "Raison n'utilise pas la PF: Autre réponse spécifiée #3",
            **shared_indicators
        },
        'Endline': {
            'ewiyear':             'Year',
            'ewimon':              'Month',
            'ew102':               Base.AGE,
            'ew208':               Base.PARITY,
            'emethod':             Base.METHOD,
            'emethodtype':         Base.METHODTYPE,
            'ewoman_weight_6city': Base.WEIGHT,
            'ecity':               'City',
            'eunmet_cmw':          Base.UNMET,
            'ew512':               Base.MARRIED,


            'ew339a':   "Raison n'utilise pas la PF: Pas de rapports sexuels",
            'ew339b':   "Raison n'utilise pas la PF: Rapports sexuels non fréquents",
            'ew339c':   "Raison n'utilise pas la PF: Pas encore mariée/pas de partenaire",
            'ew339d':   "Raison n'utilise pas la PF: Partenaire/mari absent",
            'ew339e':   "Raison n'utilise pas la PF: Déjà enceinte",
            'ew339f':   "Raison n'utilise pas la PF: Allaitement",
            'ew339g':   "Raison n'utilise pas la PF: Récemment eu un bébé",
            'ew339h':   "Raison n'utilise pas la PF: Veut plus d'enfant/tomber enceinte",
            'ew339i':   "Raison n'utilise pas la PF: Ménopause/hystérectomie",
            'ew339j':   "Raison n'utilise pas la PF: Ne peut pas avoir d'enfants",
            'ew339k':   "Raison n'utilise pas la PF: L'__enquêtée est opposée",
            'ew339l':   "Raison n'utilise pas la PF: La partenaire est opposée",
            'ew339m':   "Raison n'utilise pas la PF: D__'autres personnes sont opposées",
            'ew339n':   "Raison n'utilise pas la PF: Interdiction religieuse",
            'ew339o':   "Raison n'utilise pas la PF: Ne connait pas de méthode",
            'ew339p':   "Raison n'utilise pas la PF: Ne sait pas quelle méthode utiliser",
            'ew339q':   "Raison n'utilise pas la PF: Ne sait pas comment utiliser une méthode",
            'ew339r':   "Raison n'utilise pas la PF: Ne connait aucune source",
            'ew339s':   "Raison n'utilise pas la PF: Problèmes de santé",
            'ew339t':   "Raison n'utilise pas la PF: Peur des effets secondaires",
            'ew339u':   "Raison n'utilise pas la PF: Manque d'accès/trop éloigné",
            'ew339v':   "Raison n'utilise pas la PF: Coûte trop chère",
            'ew339w':   "Raison n'utilise pas la PF: Pas pratique à utiliser",
            'ew339x':   "Raison n'utilise pas la PF: N'__aime pas les méthodes existantes",
            'ew339y':   "Raison n'utilise pas la PF: Expérience malheureuse avec méth. existantes",
            'ew339z':   "Raison n'utilise pas la PF: Fataliste: dépend de Dieu",
            'ew339ww':  "Raison n'utilise pas la PF: Autre réponse #1",
            'ew339xx':  "Raison n'utilise pas la PF: Autre réponse #2",
            'ew339yy':  "Raison n'utilise pas la PF: Autre réponse #3",
            'ew339zz':  "Raison n'utilise pas la PF: Ne sait pas",
            'ew339wws': "Raison n'utilise pas la PF: Autre réponse spécifiée #1",
            'ew339xxs': "Raison n'utilise pas la PF: Autre réponse spécifiée #2",
            'ew339yys': "Raison n'utilise pas la PF: Autre réponse spécifiée #3",
            **shared_indicators
        },
    }


    def __init__(self, foldername, force_read=False, cores=8):
        self.barrier_keys = [k for k in self.indicators.keys() if k[:4] == 'w347' or k[:5] == 'mw318' or k[:5]=='ew339']
        self.barrier_map = {k: v.split(':')[1][1:] for k,v in self.indicators.items() if k in self.barrier_keys}

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

        values = pd.io.stata.StataReader(filename).value_labels()
        replace_map = {k: values[k.upper()] if k.upper() in values else values[k] for k in found_keys if k in values or k.upper() in values}

        # Ugh
        if year == 'Midline':
            replace_map['mmethodtype'] = values['methodtype'] # Doesn't appear to be an entry for mmethodtype?
        elif year == 'Endline':
            replace_map['emethod'] = values['method'] # Doesn't appear to be an entry for emethod?a # emethodvl is ~same (icud vs iud)
            replace_map['emethodtype'] = values['methodtype'] # Doesn't appear to be an entry for emethodtype? # emethodtypevl is ~same (icud vs iud)

        for k in replace_map.keys():
            if self.indicators[year][k] == self.PARITY:
                print('Skipping parity')
                continue

            data[k] = data[k].fillna(-1)

            data[k], replace_map[k] = Base.remove_dup_replacement_values(data[k], replace_map[k])
            data[k], replace_map[k] = Base.fill_replacement_keys(data[k], replace_map[k])
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
        data['AgeBin'] = pd.cut(data[self.AGE], bins = age_edges, labels=labels, right=False)

        parity_edges = list(range(6+1)) + [99]
        a,b = itertools.tee(parity_edges)
        a = list(a)[:-1]
        next(b)
        labels = [f'{c}-{d}' for c,d in zip(a,b)]
        data['ParityBin'] = pd.cut(data[self.PARITY], bins = parity_edges, labels=labels, right=False)

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
        print(sorted(data[self.METHOD].head(25)))
        print(sorted(data[self.METHODTYPE].unique()))
        data.to_hdf(self.cachefn, key='data', format='t')

        return data


    def _clean(self):
        self.raw = self.data

        # Parity is all [NaN or 0] at Midline?!  Causes seaborn ploting problems, so fill -1.
        self.data.loc[self.data['SurveyName']=='Midline', self.PARITY] = \
            self.data.loc[self.data['SurveyName']=='Midline', self.PARITY].fillna(0)

        self.data.loc[:,self.METHODDURABILITY] = self.data[self.METHOD]

        self.data.loc[:,self.UNMET].fillna('Missing', inplace=True) # Needed for plotting

        self.data.replace(
            {
                self.METHOD: {
                    #'Female sterilization': '',
                    #'No method': '',
                    #'Daily pill': '',

                    'Injectables':              'Injectable',
                    'Breastfeeding/LAM':        'LAM',
                    'iucd':                     'IUD',
                    'Female condom':            'Condom',
                    'Male condom':              'Condom',
                    'Implants':                 'Implant',

                    'Natural methods':          'Traditional',
                    'Other traditional method': 'Traditional',

                    'sdm':                      'Other modern',
                    'Other modern method':      'Other modern',
                    'Emergency pill':           'Other modern',
                },

                self.METHODDURABILITY: {
                    #'No method': NO_METHOD, # Causes...  "ValueError: Replacement not allowed with overlapping keys and values"

                    'Daily pill':               self.SHORT,
                    'Female condom':            self.SHORT,
                    'Male condom':              self.SHORT,

                    'Injectables':              self.INJECTION,

                    'Female sterilization':     self.LONG,
                    'iucd':                     self.LONG,
                    'Implants':                 self.LONG,

                    'Natural methods':          self.OTHER,
                    'Breastfeeding/LAM':        self.OTHER,
                    'Other traditional method': self.OTHER,
                    'sdm':                      self.OTHER,
                    'Other modern method':      self.OTHER,
                    'Emergency pill':           self.OTHER,
                },

                self.UNMET: {
                    '-1.0':             'Unknown',
                    'No unmet need':    'No',
                    'Unmet need':       'Yes',
                    'Missing':          'Unknown',
                }
            },
            inplace=True
        )

        self.data.reset_index(inplace=True)
