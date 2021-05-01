'''
This script loads in the pregnancy parity file and saves it as a pickle for ease
of future loading.
'''

import pandas as pd
import sciris as sc
import fpsim as fp

infile  = fp.calibration.pregnancy_parity_file_raw # Name of the input file
outfile = fp.calibration.pregnancy_parity_file     # Name of the output file

def preprocess_dhs_file():
    '''
    Extract ages, currently pregnant, and parity in 2018 DHS data in dataframe
    '''
    print(f'Loading data from {infile}...')
    dhs_pp = pd.read_stata(infile, convert_categoricals=False) # dhs_pp = dhs_pregnancy_parity

    print('Processing data...')
    dhs_pp = dhs_pp[['v012', 'v213', 'v218']]
    dhs_pp = dhs_pp.rename(columns={'v012': 'Age', 'v213': 'Pregnant', 'v218': 'Parity'})  # Parity means # of living children in DHS

    print('Saving data...')
    sc.saveobj(outfile, dhs_pp)

    return dhs_pp


if __name__ == '__main__':

    sc.tic() # Start timing
    dhs_pp = preprocess_dhs_file() # Load, process, and save the file
    sc.toc() # Stop timing
    print('Done.')

