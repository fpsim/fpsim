import sciris as sc
import fp_data.DHS.calobj as co

raw_dhs_file = '/home/cliffk/idm/fp/data/DHS/SNIR7ZDT/SNIR7ZFL.DTA' # TODO: make more generic

sc.tic()
try:
    print('Loading original data and saving...')
    calobj = co.CalObj(raw_dhs_file, which='DHS7')
    calobj.save('../data/senegal.cal')
except:
    print('Loading failed :(')
    sc.toc()
    raise

sc.toc()
print('Done.')


