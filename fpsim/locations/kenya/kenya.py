"""
Set the parameters for a location-specific FPsim model.
"""
import numpy as np
from pathlib import Path
from fpsim import defaults as fpd
import fpsim as fp
import fpsim.locations.data_utils as fpld
import starsim as ss

# %% Housekeeping

def filenames():
    """ Data files for use with calibration, etc -- not needed for running a sim """
    base_dir = Path(__file__).resolve().parent / 'data'
    files = {
        'base': base_dir,
        'basic_wb': base_dir / 'basic_wb.yaml', # From World Bank https://data.worldbank.org/indicator/SH.STA.MMRT?locations=ET
        'popsize': base_dir / 'popsize.csv', # From UN World Population Prospects 2022: https://population.un.org/wpp/Download/Standard/Population/
        'mcpr': base_dir / 'cpr.csv',  # From UN Population Division Data Portal, married women 1970-1986, all women 1990-2030
        'tfr': base_dir / 'tfr.csv',   # From World Bank https://data.worldbank.org/indicator/SP.DYN.TFRT.IN?locations=ET
        'asfr': base_dir / 'asfr.csv', # From UN World Population Prospects 2022: https://population.un.org/wpp/Download/Standard/Fertility/
        'ageparity': base_dir / 'ageparity.csv', # Choose from either DHS 2016 or PMA 2022
        'spacing': base_dir / 'birth_spacing_dhs.csv', # From DHS
        'methods': base_dir / 'mix.csv', # From PMA
        'afb': base_dir / 'afb.table.csv', # From DHS
        'use': base_dir / 'use.csv', # From PMA
        'education': base_dir / 'edu_initialization.csv', # From DHS
    }
    return files


# %% Make module parameters
def make_pars():
    """ Make a dictionary of location-specific parameters """
    # Create all parameters
    pars = fp.all_pars()

    # FP parameters - move to csv?
    exposure_age = np.array([
                [0, 5, 10, 12.5, 15, 18, 20, 25, 30, 35, 40, 45, 50],
                [1, 1, 1,  1 ,   .4, 1.3, 1.5 ,.8, .8, .5, .3, .5, .5]
    ])
    pars['exposure_age'] = fpld.data2interp(exposure_age, fpd.spline_preg_ages)

    # move to csv?
    exposure_correction_parity = np.array([
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                [1, 1, 1, 1, 1, 1, 1, 0.8, 0.5, 0.3, 0.15, 0.10, 0.05, 0.01]
    ])
    pars['exposure_parity'] = fpld.data2interp(exposure_correction_parity, fpd.spline_parities)

    return pars


def dataloader(location='kenya'):
    return fpld.DataLoader(location=location)
