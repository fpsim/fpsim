import numpy as np
import itertools
import pandas as pd
from functools import partial
from pathlib import Path


class Base:
    AGE                 = 'Age'
    PARITY              = 'Parity'
    METHOD              = 'Method'
    METHODTYPE          = 'MethodType'
    METHODDURABILITY    = 'MethodDurability'
    WEIGHT              = 'Weight'
    UNMET               = 'Unmet'
    MARRIED             = 'Married'


    NO_METHOD = 'No method'
    SHORT = 'Short-term'
    LONG = 'Long-term'
    INJECTION = 'Injection'
    OTHER = 'Other'


    def __init__(self, foldername, force_read=False, cores=8):
        self.cores = cores
        self.foldername = foldername
        self.force_read = force_read
        Path("cache").mkdir(exist_ok=True)
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)


    def add_date(self):
        # Calculate the mean time point of each survey in years
        '''
        # Accounts for weight, but is remarkably slow!
        data['Date'] = data.apply(cmc_to_year, axis=1)
        year_to_date = data.groupby('SurveyName').apply( partial(wmean, value='Date', weight='v005') )
        year_to_date.name = 'Date'
        '''
        # Faster, but un-weighted, year_to_date calculation
        self.year_to_date = self.data.groupby('SurveyName')['SurveyYear', 'InterviewDateCMC'].mean().apply(self.cmc_to_year, axis=1) # mean CMC
        self.year_to_date.name = 'Date'


    def create_bins(self):
        age_edges = list(range(15,45+1,15)) + [99]
        a,b = itertools.tee(age_edges)
        a = list(a)[:-1]
        next(b)
        labels = [f'{c}-{d}' for c,d in zip(a,b)]
        self.data['AgeBinCoarse'] = pd.cut(self.data['Age'], bins = age_edges, labels=labels, right=False)

        age_edges = list(range(15,55,5)) + [99]
        a,b = itertools.tee(age_edges)
        a = list(a)[:-1]
        next(b)
        labels = [f'{c}-{d}' for c,d in zip(a,b)]
        self.data['AgeBin'] = pd.cut(self.data['Age'], bins = age_edges, labels=labels, right=False)

        parity_edges = list(range(6+1)) + [99]
        a,b = itertools.tee(parity_edges)
        a = list(a)[:-1]
        next(b)
        #labels = [f'{c}-{d}' for c,d in zip(a,b)]
        #self.data['ParityBin'] = pd.cut(self.data['Parity'], bins = parity_edges, labels=labels, right=False)

        #self.data['ParityBin'] = pd.cut(self.data['Parity'], bins = parity_edges, right=False)

        labels = [f'{z}' for z in a]
        labels[-1] = f'{a[-1]}+'
        self.data['ParityBin'] = pd.cut(self.data['Parity'], bins = parity_edges, labels=labels, right=False)


    def cmc_to_year(self, data, survey_year_col='SurveyYear', cmc_col='InterviewDateCMC'): # v008 is date of interview
        survey_year = data[survey_year_col]
        cmc = data[cmc_col]

        if np.isnan(cmc):
            return np.nan

        if survey_year < 100:
            year = 1900 + int((cmc-1)/12)
            month = cmc - (survey_year*12)
        elif survey_year >= 2000:
            year = 1900 + int((cmc-1)/12)
            month = cmc - ((survey_year-1900)*12)
        else:
            raise Exception('Help!')

        return year + month/12


    @staticmethod
    def remove_dup_replacement_values(d, rm):
        l = list( rm.values() )
        lsl = list(set(l))
        if len(l) == len(lsl): # No dups
            return d, rm

        print('Fixing duplicates')

        # Find keys associates with repeated values
        unique = {}
        dup = {}
        for kk,v in rm.items():
            if v not in unique.keys():
                # New value!
                unique[v] = kk
            else:
                dup[kk] = unique[v]

        d = d.replace(dup)
        for kk in dup.keys(): # Could reverse unique
            #print(f'Removing {kk} from replace_map[{k}]')
            del rm[kk]

        return d, rm

    @staticmethod
    def fill_replacement_keys(d, rm):
        all_keys_in_replace_map = all([(kk in rm) or (kk==-1) for kk in d.unique()])
        if all_keys_in_replace_map:
            print('Fixing missing replacement keys')
            # OK, we can fix it - just add the missing entries to the replace_map[k], that way codes are preserved
            largest_index = int(d.unique().max())
            for i in range(largest_index+1):
                if i not in rm:
                    rm[i] = f'Dummy{i}'
            return d, rm
        return d, rm
