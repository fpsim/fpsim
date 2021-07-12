'''
This script loads in the pregnancy parity file and saves it as a pickle for ease
of future loading.
'''

import pandas as pd
import sciris as sc
import fpsim as fp

dhs_infile      = fp.experiment.pregnancy_parity_file_raw # Name of the input file
dhs_outfile     = fp.experiment.pregnancy_parity_file     # Name of the output file
spacing_infile  = fp.experiment.spacing_file_raw
spacing_outfile = fp.experiment.spacing_file



def preprocess_dhs_file():
    '''
    Extract ages, currently pregnant, and parity in 2018 DHS data in dataframe
    '''
    print(f'Loading data from {dhs_infile}...')
    dhs_pp = pd.read_stata(dhs_infile, convert_categoricals=False) # dhs_pp = dhs_pregnancy_parity

    print('Processing data...')
    dhs_pp = dhs_pp[['v012', 'v213', 'v218']]
    dhs_pp = dhs_pp.rename(columns={'v012': 'Age', 'v213': 'Pregnant', 'v218': 'Parity'})  # Parity means # of living children in DHS

    print(f'Saving data to {dhs_outfile}...')
    sc.saveobj(dhs_outfile, dhs_pp)

    return dhs_pp


def preprocess_spacing_file():
    '''
    Simply convert birth spacing data from a CSV to a pickle, to save on loading time.
    '''
    print('Converting and saving birth spacing file...')
    data = pd.read_csv(spacing_infile)
    right_year = data['SurveyYear'] == '2017'
    not_first = data['Birth Order'] != 0
    is_first = data['Birth Order'] == 0
    filtered = data[(right_year) & (not_first)]
    spacing = filtered['Birth Spacing'].to_numpy()
    first_filtered = data[(right_year) & (is_first)]
    first = first_filtered['Birth Spacing'].to_numpy()
    data = dict(spacing=spacing, first=first)
    sc.saveobj(spacing_outfile, data)
    return spacing


if __name__ == '__main__':

    sc.tic() # Start timing
    dhs_pp  = preprocess_dhs_file()     # Load, process, and save the DHS file
    spacing = preprocess_spacing_file() # Load, process, and save the spacing file
    sc.toc() # Stop timing
    print('Done.')

