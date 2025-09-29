"""
Set the parameters for Cotedivoire.
"""
import numpy as np
import os
import pandas as pd
import starsim as ss
import fpsim.locations.data_utils as fpld

def birth_spacing_pref():
    """
    Returns an array of birth spacing preferences by closest postpartum month.
    If the CSV file is missing, a default table with equal weights is used.
    """
    # Try to read the CSV, fallback to dummy df if not found
    try:
        df = fpld.read_data('birth_spacing_pref.csv', os.path.abspath(__file__) / 'data' )
    except FileNotFoundError:
        print(f"birth_spacing_pref.csv not found, using default weights of 1.")
        months = np.arange(0, 39, 3)  # 0 to 36 months in 3-month intervals
        weights = np.ones_like(months, dtype=float)
        df = pd.DataFrame({'month': months, 'weights': weights})

    # Check uniform intervals
    intervals = np.diff(df['month'].values)
    interval = intervals[0]
    assert np.all(intervals == interval), (
        f"In order to be computed in an array, birth spacing preference bins must be uniform. Got: {intervals}"
    )

    pref_spacing = {
        'interval': interval,
        'n_bins': len(intervals),
        'months': df['month'].values,
        'preference': df['weights'].values
    }

    return pref_spacing

def make_calib_pars():
    pars = {}
    pars['fecundity'] = ss.uniform(low=0.7, high=1.1)
    pars['exposure_factor'] = 2
    pars['prob_use_year'] = 2020
    pars['prob_use_trend_par'] = 0.01
    pars['prob_use_intercept'] = 0.5
    pars['method_weights'] = np.array([8, 5, 5, 20, 2, 2, 0.015, 0.015, 1])
    pars['dur_postpartum'] = 23

    pars['spacing_pref'] = {}
    pars['spacing_pref']['interval'] = 3.0
    pars['spacing_pref']['n_bins'] = 18
    pars['spacing_pref']['months'] = np.arange(0, 54, 3)  # 0 to 51 months in 3-month intervals
    pars['spacing_pref']['preference'] = np.ones(pars['spacing_pref']['n_bins'], dtype=float)
    pars['spacing_pref']['preference'][:3] =  1  # Spacing of 0-6 months
    pars['spacing_pref']['preference'][3:6] = 0.5  # Spacing of 9-15 months
    pars['spacing_pref']['preference'][6:9] = 0.8  # Spacing of 18-24 months
    pars['spacing_pref']['preference'][9:] =  2  # Spacing of 27-36 months
    pars['exposure_age'] = np.array([[0,    5,  10, 12.5, 15, 18, 20, 25,  30,  35, 40, 45, 50],
                                        [0.1, 0.1, 0.5,  0.1,0.1, 0.6,1.5,0.6,0.6,0.2,  0.3,0.5,0.2]])  # <<< USER-EDITABLE: Can be modified for calibration
    pars['exposure_parity'] = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                                           [1, 1, 1, 1, 1, 1, 1, 0.8, 0.5, 0.3, 0.15, 0.10, 0.05, 0.01]])  # <<< USER-EDITABLE: Can be modified for calibration


    return pars



def dataloader(location='cotedivoire'):
    return fpld.DataLoader(location=location)
