import os
import pandas as pd

def abspath(path):
    cwd = os.path.abspath(os.path.dirname(__file__))
    output = os.path.join(cwd, path)
    return output

pop_pyr_1982_fn = abspath('data/senegal-population-pyramid-1982.csv')
pop_pyr_2015_fn = abspath('data/senegal-population-pyramid-2015.csv')
popsize_tfr_fn = abspath('data/senegal-popsize-tfr.csv')

# Load data
pop_pyr_1982 = pd.read_csv(pop_pyr_1982_fn)
pop_pyr_2015 = pd.read_csv(pop_pyr_2015_fn)
popsize_tfr  = pd.read_csv(popsize_tfr_fn, header=None)

# Handle population size
scale_factor = 1000
years = popsize_tfr.iloc[0,:].to_numpy()
popsize = popsize_tfr.iloc[1,:].to_numpy() / scale_factor