import os
import sciris as sc
import fp_data.DHS.calobj as co

# Add your username here -- raw data file is 98 MB
raw_dhs_files = {'cliffk': '/home/cliffk/idm/fp/data/DHS/SNIR7ZDT/SNIR7ZFL.DTA',
                 }

sc.tic()
try:
    print('Loading original data and saving...')
    username = os.path.split(os.path.expanduser('~'))[-1]
    calobj = co.CalObj(raw_dhs_files[username], which='DHS7')
    calobj.save('../DHS/senegal.cal')
except:
    print('Loading failed :(')
    sc.toc()
    raise

sc.toc()
print('Done.')


