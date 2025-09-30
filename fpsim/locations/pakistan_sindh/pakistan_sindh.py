"""
Set the parameters for Pakistan Sindh.
"""
import numpy as np
import starsim as ss
import fpsim.locations.data_utils as fpld

def make_calib_pars():
    pars = {}
    pars['fecundity'] = ss.uniform(low=0.1, high=0.9)
    pars['exposure_factor'] = 4
    pars['prob_use_year'] = 2020
    pars['prob_use_trend_par'] = 0.01
    pars['prob_use_intercept'] = -2.7
    pars['method_weights'] = np.array([0.06, 2.3, 20, 2, 2.5, 1.5, 3, 0.1, 0.01])
    pars['dur_postpartum'] = 23

    pars['spacing_pref'] = {}
    pars['spacing_pref']['interval'] = 3.0
    pars['spacing_pref']['n_bins'] = 18
    pars['spacing_pref']['months'] = np.arange(0, 54, 3)
    pars['spacing_pref']['preference'] = np.ones(pars['spacing_pref']['n_bins'], dtype=float)
    pars['spacing_pref']['preference'][:3] =  1
    pars['spacing_pref']['preference'][3:6] = 1
    pars['spacing_pref']['preference'][6:9] = 5
    pars['spacing_pref']['preference'][9:] =  0.1
    pars['exposure_age'] = np.array([[0,     5, 10, 12.5, 15, 18, 20, 25, 30, 35, 40, 45, 50],
                                        [0.2, 0.2, 0.2,  1 ,0.9, 0.8,0.8,1.25,1, 0.7,0.5,0.5, 0.5]])
    pars['exposure_parity'] = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                                           [1, 1, 1, 1, 1, 1, 1, 0.8, 0.5, 0.3, 0.15, 0.10, 0.05, 0.01]])

    return pars


def dataloader(location='pakistan_sindh'):
    return fpld.DataLoader(location=location)