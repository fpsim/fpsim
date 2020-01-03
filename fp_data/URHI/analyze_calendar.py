# DJK, Dec 20, 2019
# Purpose is to decode the calendars that appear in the women's report of the URHI final (2015) wave

from scipy.sparse import coo_matrix
import os
from pathlib import Path
import seaborn as sns
sns.set()
import glob
import itertools
import pylab as pl
import numpy as np
import pandas as pd
import sciris as sc

# Check versions
sciris_version = sc.__version__
min_version = '0.14.10'
assert sc.compareversions(sciris_version, min_version)>=0, f'Please upgrade Sciris from {sciris_version} to at least {min_version}'

T = sc.tic()

cachefn = 'calendar.hdf'
pickle_filename = 'senegal_calendear.obj'
store = pd.HDFStore(cachefn)
force_read = False

username = os.path.split(os.path.expanduser('~'))[-1]
folderdict = {'dklein': os.path.join( os.getenv("HOME"), 'Dropbox (IDM)', 'URHI', 'Senegal'),
              'cliffk': '/home/cliffk/idm/fp/data/Senegal',
             }
try:
    basedir = folderdict[username]
except:
    raise Exception(f'User {username} not found among users {list(folderdict.keys())}, cannot find data folder')

'''
cipher_method = {
    'B': 'Birth',
    'T': 'Terminated pregnancy/non-live birth',
    'P': 'Pregnancy',
    '0': 'Non-use of contraception',
    '1': 'Pill',
    '2': 'IUD',
    '3': 'Injectables',
    '4': 'Diaphragm',
    '5': 'Condom',
    '6': 'Female sterilization',
    '7': 'Male sterilization',
    '8': 'Periodic abstinence/rhythm',
    '9': 'Withdrawal',
    'W': 'Other traditional methods',
    'N': 'Implants', # Guessing this might instead be pregnancy?
#   'G': 'Pregnant', # <-- Total guess, see 'G' 8x after N (Implants???) and 2x after each F (Foam and Jelly)
    'A': 'Abstinence',
    'L': 'Lactational amenorrhea method (LAM)',
    'C': 'Female cggondom',
    'F': 'Foam and Jelly',
    'E': 'Emergency contraception (DHSVI)',
    'S': 'Standard days method (DHSVI)',
    'M': 'Other modern method (DHSVI)',
    'α2': 'Country-specific method 1',
    'ß2': 'Country-specific method 2',
    'τ2': 'Country-specific method 3',
    '?': 'Unknown method/missing data',
}
'''

cipher_method = {
    'N': 'Birth',
    'G': 'Pregnancy',
    'F': 'Miscarriage',
    'M': 'Stillbirth',
    'A': 'Abortion',

    '0': 'No Method',
    '1': 'Female sterilization',
    '2': 'Male sterilization',
    '3': 'Implant',
    '4': 'IUD',
    '5': 'Injectables',
    '6': 'Pill',
    '7': 'Emergency contraception',
    '8': 'Male condom',
    '9': 'Female condom',

    'H': 'Menstrual cycle collar',
    'L': 'Lactational amenorrhea method (LAM)',
    'X': 'Other modern method',

    'R': 'Rhythm method',
    'W': 'Withdrawl',
    'Y': 'Other traditional method',
}

cipher_discontinuation = {
    '1': 'Became pregnant while using',
    '2': 'Wanted to become pregnant',
    '3': 'Husband disapproved',
    '4': 'Side effects/health concerns',
    '5': 'Health concerns',
    '6': 'Access/availability',
    '7': 'Wanted a more effective method',
    '8': 'Inconvenient to use',
    '9': 'Infrequent sex/husband away',
    'C': 'Cost too much',
    'F': 'Up to God/fatalistic',
    'A': 'Difficult to get pregnant/menopausal',
    'D': 'Marital dissolution/separation',
    'W': 'Other',
    'K': 'Do not know',
}

cipher_marriage = {
    'X': 'In union (married or living together)',
    'O': 'Not in union',
}

def decrypt(dat):
    method  = dat['cal_1']
    source  = dat['cal_2']
    reason  = dat['cal_3']
    married = dat['cal_4']
    other1   = dat['cal_1o']
    other2   = dat['cal_2o']
    other3   = dat['cal_3o']

    print('-'*80)
    print('UID:', dat['UID'])
    print('METHOD :', method)
    print('SOURCE :', source)
    print('REASON :', reason)
    print('MARRIED:', married)
    print('OTHER1 :', other1)
    print('OTHER2 :', other2)
    print('OTHER3 :', other3)

    # TODO: More efficient...
    assert(len(method) == len(source))
    assert(len(method) == len(reason))
    assert(len(method) == len(married))
    #assert(len(method) == len(other1)) # <-- These seem to have a different length?
    #assert(len(method) == len(other2))
    #assert(len(method) == len(other3))

    # Look for something interesting in other
    #if any([True for o in other1 if o not in [' ', '?']]):
    #    print('OTHER  :', other1)

    # Go through one by one and decode using the cipher
    for mthd, rsn, mrd in zip(method, reason, married):
        if rsn != ' ':
            if rsn in cipher_discontinuation:
                print(rsn, ' ****** ', cipher_discontinuation[rsn])
            else:
                print(rsn, ' ****** ', f'{rsn} ???')

        if mthd in cipher_method:
            print(mthd, ' --> ', cipher_method[mthd])
        else:
            print(mthd, ' --> ', f'{mthd} ???')


if (not force_read) and os.path.isfile(cachefn) and 'cal' in store:
    data = store['cal']
else:
#filenames = glob.glob( os.path.join(basedir, 'Endline', '*.dta'))
    filename = os.path.join( basedir, 'Endline', 'SEN_end_wm_match_20160505.dta' )


    fn = Path(filename).resolve().stem
    print(f'File: {filename} ...')
    data = pd.read_stata(filename, convert_categoricals=False)

    if False:
        values = pd.io.stata.StataReader(filename).value_labels()
        codebook = pd.io.stata.StataReader(filename).variable_labels()

        pd.DataFrame({'keys': list(codebook.keys()), 'values': list(codebook.values())}).set_index('keys').to_csv(f'codebook_{fn}.csv')

    data['UID'] = data['location_code'].astype(str) + '.' + data['hhnum'].astype(str) + '.' + data['line'].astype(str)

    ''' # From the codebook...
    cal_1	Column 1 of calendar - Births, pregnancies, terminations, contraceptive use
    cal_2	Column 2 of calendar - Source of contraceptive
    cal_3	Column 3 of calendar - Discontinuation
    cal_4	Column 4 of calendar - Marital status
    cal_1o	Column 1 of calendar - Other contraceptive method specified
    cal_2o	Column 2 of calendar - Other source of contraceptive specified
    cal_3o	Column 3 of calendar - Other reason for discontinuation specified
    '''

    #print( '\n'.join(data.columns) )
    store['cal'] = data[['UID', 'cal_1', 'cal_2', 'cal_3', 'cal_4', 'cal_1o', 'cal_2o', 'cal_3o', 'ewoman_weight']]

store.close()

print(f'Data has {data.shape[0]} rows.')

chr_to_idx = {}
def build_chr_to_idx(dat):
    for c in dat['cal_1'].strip():
        if c not in chr_to_idx:
            chr_to_idx[c] = len(chr_to_idx)

def build_switching_matrix(dat):
    inds = [chr_to_idx[c] for c in dat['cal_1'].strip() ]

    row, col = itertools.tee(inds)
    next(col, None)

    rows = np.array(list(row)[:-1])
    cols = np.array(list(col))
    vals = np.ones_like(rows)
    vals_weighted = dat['ewoman_weight'] * np.ones_like(rows)

    #return coo_matrix((vals_weighted, (row, col)), shape=(row.max()+1, col.max()+1))
    return rows, cols, vals, vals_weighted


data.apply(build_chr_to_idx, axis=1)

rows = []
cols = []
vals = []
vals_weighted = []

for idx, dat in data.iterrows():
    r, c, v, vw = build_switching_matrix(dat)
    rows.append(r)
    cols.append(c)
    vals.append(v)
    vals_weighted.append(vw)


index = [cipher_method[k] for k in chr_to_idx.keys()]

rows = np.concatenate(rows)
cols = np.concatenate(cols)
vals_weighted = np.concatenate(vals_weighted)
def form_switching_matrix(rows,  cols, vals, fn = None):
    M = coo_matrix((vals_weighted, (rows, cols)), shape=(rows.max()+1, cols.max()+1)).todense()
    M = pd.DataFrame(M, index = index, columns = index)
    if fn is not None:
        M.to_csv(fn)

    return M

M = form_switching_matrix(rows, cols, vals, 'SwitchingMatrix_Unweighted.csv')
W = form_switching_matrix(rows, cols, vals_weighted, 'SwitchingMatrix_Weighted.csv')


with pd.option_context('display.max_colwidth', -1):
    print(M)
    exit()

    # Let's decode some record
    data.iloc[range(500,600)].apply(decrypt, axis=1)

sc.toc(start=T)

print('Done.')
