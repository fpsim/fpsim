import numpy as np
import pandas as pd
import sciris as sc

np.set_printoptions(suppress = True)

pma = pd.read_csv('../kenya/matrices_kenya_pma_2019_20.csv')

raw = {
    'annual': {},
    'pp1to6': {},
    'pp0to1': {},
}
keys = {'annual': 'No', 'pp1to6': '6', 'pp0to1': '1'}
bins = ['<18', '18-20', '20-25', '25-35', '>35']

for k,v in keys.items():
    for b in bins:
        matrix = pma[(pma['postpartum'] == v) & (pma['age_grp'] == b)]
        if b == '20-25':
            new = '21-25'
            raw[k][new] = matrix.to_numpy()[:, -10:].astype(float).round(4)
        elif b == '25-35':
            new = '26-35'
            raw[k][new] = matrix.to_numpy()[:, -10:].astype(float).round(4)
        else:
            raw[k][b] = matrix.to_numpy()[:, -10:].astype(float).round(4)

print(raw)


