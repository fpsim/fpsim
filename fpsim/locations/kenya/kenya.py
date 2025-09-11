"""
Set the parameters for a location-specific FPsim model.
"""
import numpy as np
from pathlib import Path
from fpsim import defaults as fpd
import fpsim as fp
import fpsim.locations.data_utils as fpld
import starsim as ss


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
