'''
Generate the population size data for Senegal.

Based on UN World Population Projections.
'''

import pylab as pl
import pandas as pd
import sciris as sc

uhri_file = '../fp_data/UHRI/senegal_women.obj' # Not Windows friendly but simpler and neither is later code

# UN population data
def UN_pop_data(init_pop=1000.0):
    years = pl.arange(1980,2021)
    popsize = pl.array([5583, 5740, 5910, 6090, 6277, 6471, 6671, 6876, 7087, 7304, \
                        7526, 7756, 7990, 8227, 8461, 8690, 8913, 9131, 9348, 9569, \
                        9798, 10036, 10284, 10541, 10810, 11090, 11382, 11687, 12005, \
                        12335, 12678, 13034, 13402, 13782, 14175, 14578, 14994, 15419, \
                        15854, 16296, 16744], dtype=float)
    popsize *= init_pop/popsize[0]
    return sc.objdict({'years':years, 'popsize':popsize})

# Age-parity data
def age_parity_data(uhri_file=uhri_file):
    data = sc.loadobj(uhri_file)
    
    age_edges = list(range(15,55,5)) + [99]
    parity_edges = list(range(6+1)) + [99]
    data['AgeBin'] = pd.cut(data['Age'], bins = age_edges, right=False)
    data['ParityBin'] = pd.cut(data['Parity'], bins = parity_edges, right=False)
    data['AgeBinCode'] = data['AgeBin'].cat.codes
    data['ParityBinCode'] = data['ParityBin'].cat.codes
    age_bin_codes = pl.array(sorted(list(data['AgeBinCode'].unique())))
    parity_bin_codes = pl.array(sorted(list(data['ParityBinCode'].unique())))
    age_parity_data = pl.zeros((len(age_bin_codes), len(parity_bin_codes)))
    
    for i,row in data.iterrows():
        age_index = row['AgeBinCode']
        parity_index = row['ParityBinCode']
        weight = row['Weight']
        age_parity_data[age_index, parity_index] += weight
    
    return sc.objdict({'data':age_parity_data, 'agebins':age_edges, 'paritybins':parity_edges})


if __name__ == '__main__':
    popdata = UN_pop_data()
    ageparitydata = age_parity_data()
    data = sc.objdict({'unpop':popdata, 'ageparity':ageparitydata})
    sc.saveobj('senegal_data.obj', data)
    print('Done.')