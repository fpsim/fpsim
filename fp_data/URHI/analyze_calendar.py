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

cipher_method = {
    'N': 'Birth',
    'G': 'Pregnancy',
    'F': 'Miscarriage',
    'M': 'Stillbirth',
    'A': 'Abortion',

    '0': 'No method',
    '1': 'Female sterilization',
    '2': 'Male sterilization',
    '3': 'Implant',
    '4': 'IUD',
    '5': 'Injectables',
    '6': 'Pill',
    '7': 'Emergency contraception',
    '8': 'Male condom',
    '9': 'Female condom',

    'H': 'Standard days method',
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
    # Note reverse because data starts in 2015 and goes backwards
    method  = dat['cal_1'][::-1]
    source  = dat['cal_2'][::-1]
    reason  = dat['cal_3'][::-1]
    married = dat['cal_4'][::-1]
    other1   = dat['cal_1o'][::-1]
    other2   = dat['cal_2o'][::-1]
    other3   = dat['cal_3o'][::-1]

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
    cal = store['cal']
    data = store['all']
else:
#filenames = glob.glob( os.path.join(basedir, 'Endline', '*.dta'))
    filename = os.path.join( basedir, 'Endline', 'SEN_end_wm_match_20160505.dta' )


    fn = Path(filename).resolve().stem
    print(f'File: {filename} ...')
    data = pd.read_stata(filename, convert_categoricals=False)

    if True:
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

    store['all'] = data

    #print( '\n'.join(data.columns) )
    cal = data[['UID', 'cal_1', 'cal_2', 'cal_3', 'cal_4', 'cal_1o', 'cal_2o', 'cal_3o', 'ewoman_weight_6city']]
    store['cal'] = cal

store.close()

print(f'Data has {data.shape[0]} rows.')

# Check age distribution
data['age_bin'] = pd.cut(data['ew102'], [15, 20, 25, 30, 35, 40, 55], right=False)
age_summary = data.groupby('age_bin')['ewoman_weight_6city'].sum()
print(100 * age_summary / age_summary.sum())

# Check pregnancies
def count_starts(x, code='G'):
    if 'G' not in x:
        return 0

    p,q = itertools.tee(x.strip()[::-1])
    p = list(p)[:-1]
    next(q, None)

    count = 1 if p[0] == code else 0
    for a,b in zip(p,q):
        if a != code and b == code:
            count += 1

    return count

from functools import partial
cal['Pregnancies'] = cal['cal_1'].apply(partial(count_starts, code='G'))
print(f'Data has {cal["Pregnancies"].sum()} pregnancies.')

cal['Failures'] = cal['cal_1'].apply(lambda x: len([c for c in x if c in ['A', 'F', 'M']]))
print(f'Data has {cal["Failures"].sum()} abortions, miscarriages, and stillbirths.')

cal['Births'] = cal['cal_1'].apply(lambda x: len([c for c in x if c == 'N']))
print(f'Data has {cal["Births"].sum()} births.')

cal['Non-Use'] = cal['cal_1'].apply(partial(count_starts, code='0'))
print(f'Data has {cal["Non-Use"].sum()} continuous periods of non-use.')
#with pd.option_context('display.max_colwidth', -1):
#    print(cal[['cal_1', 'Non-Use']])

chr_to_idx = {}
def build_chr_to_idx(dat):
    for c in dat['cal_1'].strip():
        if c not in chr_to_idx:
            chr_to_idx[c] = len(chr_to_idx)

def build_switching_matrix(dat):
    # Note reverse because data starts in 2015 and goes backwards
    inds = [chr_to_idx[c] for c in dat['cal_1'].strip()[::-1] ]

    method_from, method_to = itertools.tee(inds)
    next(method_to, None)

    method_from = np.array(list(method_from)[:-1])
    method_to = np.array(list(method_to))

    vals = np.ones_like(method_from)
    vals_weighted = dat['ewoman_weight_6city'] * np.ones_like(method_from)

    return method_from, method_to, vals, vals_weighted


cal.apply(build_chr_to_idx, axis=1)

method_from = [] # rows
method_to = [] # cols
vals = []
vals_weighted = []

for idx, dat in cal.iterrows():
    r, c, v, vw = build_switching_matrix(dat)
    method_from.append(r)
    method_to.append(c)
    vals.append(v)
    vals_weighted.append(vw)


index = [cipher_method[k] for k in chr_to_idx.keys()]

method_from = np.concatenate(method_from)
method_to = np.concatenate(method_to)
vals_weighted = np.concatenate(vals_weighted)
def form_switching_matrix(method_from,  method_to, vals, fn = None):
    # method_from on rows, method_to on columns
    M = coo_matrix((vals_weighted, (method_from, method_to)), shape=(method_from.max()+1, method_to.max()+1)).todense()
    M = pd.DataFrame(M, index = index, columns = index)
    if fn is not None:
        M.to_csv(fn)

    return M

W = form_switching_matrix(method_from, method_to, vals_weighted, 'SwitchingMatrix_Weighted.csv')

W['Condoms'] = W['Male condom'] + W['Female condom']
W.loc['Condoms'] = W.loc['Male condom'] + W.loc['Female condom']

W['Other modern'] = W['Other modern method'] + W['Standard days method'] + W['Emergency contraception']
W.loc['Other modern'] = W.loc['Other modern method'] + W.loc['Standard days method'] + W.loc['Emergency contraception']

W['Traditional'] = W['Other traditional method'] + W['Rhythm method'] + W['Withdrawl']
W.loc['Traditional'] = W.loc['Other traditional method'] + W.loc['Rhythm method'] + W.loc['Withdrawl']

W['Termination/Stillbirth'] = W['Miscarriage'] + W['Stillbirth'] + W['Abortion']

'''
diagW = pd.Series(np.diag(W), index=[W.index, W.columns]).to_frame().unstack().fillna(0)
print(diagW[0])
W.sort_index(inplace=True, axis=0)
W.sort_index(inplace=True, axis=1)
W = W - diagW[0]
'''

W = W.reindex(['Implant', 'IUD', 'Injectables', 'Pill', 'Condoms', 'Lactational amenorrhea method (LAM)', 'Other modern', 'Traditional'], axis=0)
W = W.reindex(['No method', 'Implant', 'IUD', 'Injectables', 'Pill', 'Condoms', 'Other modern', 'Traditional', 'Pregnancy', 'Termination/Stillbirth'], axis=1)

diag_names = [name for name in W.columns.values if name in W.index.values]
for name in diag_names:
    W.loc[name,name] = 0

w = 100*W.div(W.sum(axis=1), axis=0)
w['Total'] = W.sum(axis=1)
print(w)
w.to_csv('test.csv')



exit()
with pd.option_context('display.max_columns', -1):
    print(W)
    exit()

    # Let's decode some record
    cal.iloc[range(500,600)].apply(decrypt, axis=1)

sc.toc(start=T)

print('Done.')
