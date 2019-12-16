import pylab as pl
import sciris as sc
import fp_utils.calobj as co

dhs_file  = '/home/cliffk/idm/fp/fp_analyses/fp_data/DHS/senegal.cal'
dhs = co.load(dhs_file) # Create from saved data
dhs.make_results()

newkeys = ['none', 'lactation', 'implant', 'injectable', 'iud', 'pill', 'condom', 'other', 'traditional']

mapping = sc.odict({
        "B": 'n/a',
        "T": 'n/a',
        "P": 'n/a', # TODO, would make for great cross-validation!
        "0": 'none',
        "1": 'pill',
        "2": 'iud',
        "3": 'injectable',
        "4": 'other',
        "5": 'condom',
        "6": 'other',
        "7": 'other',
        "8": 'traditional',
        "9": 'traditional',
        "W": 'traditional',
        "N": 'implant',
        "A": 'traditional',
        "L": 'lactation',
        "C": 'condom',
        "F": 'other',
        "E": 'other',
        "S": 'other',
        "M": 'other',
        })

n_new = len(newkeys)
newdata = pl.zeros((n_new, n_new))

for i,newkey1 in mapping.enumvals():
    for j,newkey2 in mapping.enumvals():
        if newkey1 != 'n/a' and newkey2 != 'n/a':
            new_i = newkeys.index(newkey1)
            new_j = newkeys.index(newkey2)
            newdata[new_i, new_j] += dhs.results.counts[i,j]
newdata /= newdata.sum()
    

sc.bar3d(newdata)

print('New data, copy into model.py:')
print(newdata)


