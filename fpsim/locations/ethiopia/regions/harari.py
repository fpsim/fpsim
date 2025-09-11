"""
Set the parameters for FPsim, specifically for Ethiopia.
"""


import numpy as np
import fpsim as fp
import starsim as ss
from fpsim import defaults as fpd
import fpsim.locations.data_utils as fpld


def make_calib_pars():
    pars = {}
    pars['fecundity'] = ss.uniform(0.99, 1.38)

    exposure_correction_age = np.array([[0, 5, 10, 12.5, 15, 18, 20, 25, 30, 35, 40, 45, 50],
                                        [1, 1, 1, 1, 1, 1, 1, 0.8, 0.8, 0.8, 1, 1, 1]])
    exposure_age_interp = fp.data2interp(exposure_correction_age, fpd.spline_preg_ages)
    pars['exposure_age'] = exposure_age_interp

    exposure_correction_parity = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                                           [1, 1, 1, 1, 1, 1, 1, 0.8, 0.5, 0.3, 0.15, 0.10, 0.05, 0.01]])
    exposure_parity_interp = fp.data2interp(exposure_correction_parity, fpd.spline_parities)

    pars['exposure_parity'] = exposure_parity_interp
    return pars


def dataloader(location='harari'):
    return fpld.DataLoader(location=location)

