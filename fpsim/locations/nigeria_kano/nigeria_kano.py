"""
Set the parameters for Nigeria Kano.
"""
import numpy as np
import starsim as ss
import fpsim.locations.data_utils as fpld

def make_calib_pars():
    pars = {}
    pars['fecundity'] = ss.uniform(low=0.35, high=1.1)
    pars['exposure_factor'] = 0.75
    pars['prob_use_year'] = 2020
    pars['prob_use_trend_par'] = 0.01
    pars['prob_use_intercept'] = 1.2
    pars['method_weights'] = np.array([0.0012, 0.07, 3.5, 0.4, 0.5, 0.1, 300, 7, 3])
    pars['dur_postpartum'] = 23

    spacing_pref_array = np.ones(18, dtype=float)  # Size based on n_bins from data files
    spacing_pref_array[:3] =  1
    spacing_pref_array[3:6] = 1
    spacing_pref_array[6:9] = 1
    spacing_pref_array[9:] =  1
    
    pars['spacing_pref'] = {
        'preference': spacing_pref_array
    }
    pars['exposure_age'] = np.array([[0,     5, 10, 12.5,   15, 18, 20,  25, 30,   35, 40, 45, 50],
                                        [0.5, 0.5, 0.5, 0.4 , 0.5,  1,  1,  1.1,1.7,  1.7, 0.8,0.5,0.5]])
    pars['exposure_parity'] = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                                           [1, 1, 1, 1, 1, 1, 1, 0.8, 0.5, 0.3, 0.15, 0.10, 0.05, 0.01]])

    return pars


def dataloader(location='nigeria_kano'):
    return fpld.DataLoader(location=location)