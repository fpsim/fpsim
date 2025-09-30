"""
Set the parameters for Nigeria Kaduna.
"""
import numpy as np
import starsim as ss
import fpsim.locations.data_utils as fpld

def make_calib_pars():
    pars = {}
    pars['fecundity'] = ss.uniform(low=0.01, high=1.5)
    pars['exposure_factor'] = 1
    pars['prob_use_year'] = 2020
    pars['prob_use_trend_par'] = 0.01
    pars['prob_use_intercept'] = 0.1
    pars['method_weights'] = np.array([0.005, 0.005, 200, 10, 1, 0.001, 0.003, 10, 100])
    pars['dur_postpartum'] = 23

    pars['spacing_pref'] = {}
    pars['spacing_pref']['interval'] = 3.0
    pars['spacing_pref']['n_bins'] = 18
    pars['spacing_pref']['months'] = np.arange(0, 54, 3)
    pars['spacing_pref']['preference'] = np.ones(pars['spacing_pref']['n_bins'], dtype=float)
    pars['spacing_pref']['preference'][:3] =  1
    pars['spacing_pref']['preference'][3:6] = 0.5
    pars['spacing_pref']['preference'][6:9] = 0.8
    pars['spacing_pref']['preference'][9:] =  2
    pars['exposure_age'] = np.array([[0,     5,  10, 12.5, 15, 18, 20, 25,  30, 35,  40, 45,    50],
                                        [0.1, 0.1, 0.5,  3,   2,  0.5,0.5,1.5, 1.5, 1,   1,  1,   0.5]])
    pars['exposure_parity'] = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                                           [5, 5, 5, 1, 1, 1, 1, 0.8, 0.5, 0.3, 0.15, 0.10, 0.05, 0.01]])

    return pars


def dataloader(location='nigeria_kaduna'):
    return fpld.DataLoader(location=location)