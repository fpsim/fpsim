'''
Set the parameters for FPsim.  These parameters are NOT
meant to represent a real place, but to illustrate what
one can set and to make writing tests of them easy.

  To extract this into an editable set of parameters:

  import fp_analyses.test_parameters as tp
  import json

  params = tp.make_pars()
  with open("sample_parameters.json","w") as outfile:
    json.dump(params, outfile, indent=4, sort_keys=True)
'''

import os
import numpy as np
import sciris as sc
from scipy import interpolate as si
import fpsim.defaults as fpd


# %% Set parameters for the simulation

def default_age_pyramid(age_array=None):
    '''
    Starting age bin, male population, female population
    This method defines a perfectly flat age structure,
    with the same portions in each age bin for each sex.

    Args:
        age_array [age_bucket, male_weight, female_weight]
    '''

    if not age_array:
        age_array = [[x, 1, 1] for x in range(0, 81, 5)]
    pyramid = np.array(age_array)
    return pyramid


def default_age_mortality(is_bound, test_value=0.001):
    ''' Age-dependent mortality rates.

    Defines identical mortality rates for every age
    bin and sex.

    Args:
        is_bound if the mortality spline will be bound between 1 and 0
        test_value mortality rate for each age bucket
    '''
    ages = [float(i) for i in list(range(0, 96, 5))]

    mortality = {
        'bins': np.array(ages),
        'm': np.array([test_value] * len(ages)),
        'f': np.array([test_value] * len(ages))
    }

    mortality['years'] = np.array([float(i) for i in list(range(1950, 2031, 5))])
    mortality['trend'] = np.array([12.0] * len(mortality['years']))
    mortality['trend'] /= mortality['trend'][8]  # Normalize around 2000 for trending

    m_mortality_spline_model = si.splrep(x=mortality['bins'],
                                         y=mortality['m'])  # Create a spline of mortality along known age bins
    f_mortality_spline_model = si.splrep(x=mortality['bins'], y=mortality['f'])
    m_mortality_spline = si.splev(fpd.spline_ages,
                                  m_mortality_spline_model)  # Evaluate the spline along the range of ages in the model with resolution
    f_mortality_spline = si.splev(fpd.spline_ages, f_mortality_spline_model)
    if is_bound:
        m_mortality_spline = np.minimum(1, np.maximum(0, m_mortality_spline))
        f_mortality_spline = np.minimum(1, np.maximum(0, f_mortality_spline))

    mortality['m_spline'] = m_mortality_spline
    mortality['f_spline'] = f_mortality_spline

    return mortality


def default_female_age_fecundity(is_bound, test_fecundity=80):
    '''
    This method defines female fecundity by age, setting it to very
    high values (80 conceptions per hundred over 12 menstrual cycles
    after entering the 12.5 year age bucket.

    Args:
        is_bound if the fecundity will be bound between 1 and 0
        test_fecundity rate for women over 12.5 and under 50.0
    '''
    fecundity = {
        'bins': np.array(
            [0., 5.0, 10, 12.5, 15,
             20, 25, 28, 31, 34,
             37, 40, 45, 50, 55,
             60, 65, 70, 75, 80,
             85, 90, 95, 99]),
        'f': np.array(
            [0., 0.0, 0., test_fecundity, test_fecundity,
             test_fecundity, test_fecundity, test_fecundity, test_fecundity, test_fecundity,
             test_fecundity, test_fecundity, test_fecundity, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0])
    }
    fecundity['f'] /= 100  # Conceptions per hundred to conceptions per woman over 12 menstrual cycles of trying to conceive

    fecundity_interp_model = si.interp1d(x=fecundity['bins'], y=fecundity['f'])
    fecundity_interp = fecundity_interp_model(fpd.spline_ages)
    if is_bound:
        fecundity_interp = np.minimum(1, np.maximum(0, fecundity_interp))

    return fecundity_interp


def default_maternal_mortality(test_maternal_mortality_multiplier=1.0):
    '''
    Risk of maternal death assessed at each pregnancy.
    This method sets that value to a flat rate over time.

    Args:
        test_maternal_mortality_multiplier value to multiply over all mortality rates
    '''

    data = np.array(
        [[year, 0.002, 0.002, 0.002] for year in range(1990, 2020)]
    )

    maternal_mortality = {}
    maternal_mortality['year'] = data[:, 0]
    maternal_mortality['probs'] = data[:,
                                  3] * test_maternal_mortality_multiplier  ##select column of low, median, high estimates

    return maternal_mortality


def default_infant_mortality(test_rate=50):
    '''Infant mortality rates per 1000 live births per year

    Args:
        test_rate value to set as a constant for every year
    '''

    data = np.array([[year, test_rate] for year in range(1960, 2020)])

    infant_mortality = {}
    infant_mortality['year'] = data[:, 0]
    infant_mortality['probs'] = data[:, 1] / 1000  # Rate per 1000 live births

    return infant_mortality


def default_methods():
    '''Matrices to give transitional probabilities.
    This is set to an identity matrix. No agent ever changes
    method (all start at 'None' and stay there).
    '''
    methods = {}

    methods['map'] = {
        'None': 0,
        'Pill': 1,
        'IUDs': 2,
        'Injectables': 3,
        'Condoms': 4,
        'BTL': 5,
        'Rhythm': 6,
        'Withdrawal': 7,
        'Implants': 8,
        'Other': 9,
    }

    methods['names'] = list(methods['map'].keys())

    methods['probs_matrix'] = {
        '<18': np.eye(len(methods['map'])),
        '18-20': np.eye(len(methods['map'])),
        '21-25': np.eye(len(methods['map'])),
        '>25': np.eye(len(methods['map']))
    }

    methods['mcpr_years'] = np.array(
        [1950, 1980, 1986, 1992, 1997, 2005, 2010, 2012, 2014, 2015, 2016, 2017, 2018, 2019])

    mcpr_rates = np.array([0.50, 1.0, 2.65, 4.53, 7.01, 7.62, 8.85, 11.3, 14.7, 15.3, 16.5, 18.8, 19, 20])

    methods['trend'] = mcpr_rates[-2] / mcpr_rates  # normalize trend around 2018 so "no method to no method" matrix entry will increase or decrease based on mcpr that year, probs from 2018

    return methods


def default_methods_postpartum():
    '''Function to give probabilities postpartum.
    Probabilities are transitional probabilities over 3 month period.
    This method sets an identity matrix, method does not change postpartum.
    '''
    available_methods = {
        'None': 0,
        'Pill': 1,
        'IUDs': 2,
        'Injectables': 3,
        'Condoms': 4,
        'BTL': 5,
        'Rhythm': 6,
        'Withdrawal': 7,
        'Implants': 8,
        'Other': 9
    }
    methods_postpartum = {}

    methods_postpartum['map'] = available_methods
    methods_postpartum['names'] = list(available_methods.keys())
    all_none = np.zeros(len(available_methods))
    all_none[0] = 1.0 # Sets 1.0 probability of none
    methods_postpartum['probs_matrix_0-3'] = {
        '<18': sc.dcp(all_none),
        '18-20': sc.dcp(all_none),
        '21-25': sc.dcp(all_none),
        '>25': sc.dcp(all_none)
    }

    methods_postpartum['probs_matrix_4-6'] = np.eye(len(available_methods))

    methods_postpartum['mcpr_years'] = np.array(
        [1950, 1980, 1986, 1992, 1997, 2005, 2010, 2012, 2014, 2015, 2016, 2017, 2018, 2019])

    mcpr_rates = np.array(
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    methods_postpartum['trend'] = mcpr_rates[
                                      -2] / mcpr_rates  # normalize trend around 2018 so "no method to no method" matrix entry will increase or decrease based on mcpr that year, probs from 2018

    return methods_postpartum


def default_efficacy():
    ''' Test data only '''

    method_efficacy = sc.odict({
        "None": 0.0,
        "Pill": 100.0,
        "IUDs": 100.0,
        "Injectable": 80.0,
        "Condoms": 60.0,
        "BTL": 40.0,
        "Rhythm": 20.0,
        "Withdrawal": 10.0,
        "Implants": 5.0,
        "Other": 2.5,
    })

    method_efficacy = method_efficacy[:] / 100

    return method_efficacy


def default_barriers():
    barriers = sc.odict({
        'No need': 54.2,
        'Opposition': 30.5,
        'Knowledge': 1.7,
        'Access': 4.5,
        'Health': 12.9,
    })
    barriers[:] /= barriers[:].sum()  # Ensure it adds to 1
    return barriers


def default_sexual_activity(test_rate=100.0):
    '''
    Returns a linear interpolation of rates of female sexual activity,
    defined as percentage women who have had sex within the last four weeks.

    Args:
        test_rate percentage to apply to all ages over 18
    '''

    age_buckets = [0, 5, 10, 12.5, 15, 18, 20, 25, 30, 35, 40, 45, 50]
    rates_under18 = [0.0] * len(age_buckets[:age_buckets.index(18)])
    rates_over18 = [test_rate] * len(age_buckets[age_buckets.index(18):])
    sexually_active = np.array([age_buckets,
                                rates_under18 + rates_over18])
    sexually_active[1] /= 100  # Convert from percent to rate per woman
    activity_ages = sexually_active[0]
    activity_interp_model = si.interp1d(x=activity_ages, y=sexually_active[1])
    activity_interp = activity_interp_model(fpd.spline_preg_ages)  # Evaluate interpolation along resolution of ages

    return activity_interp


def default_sexual_activity_postpartum(
        test_first_two=1.0,
        test_rest=0.2
):
    '''
    Returns an array of monthly likelihood of having resumed sexual activity
    within 0-11 months postpartum

    Default is set to 2 months of no sexual activity and then fully resuming.
    '''

    postpartum_abstinence_rates = \
    [[month, test_first_two] for month in range(0, 2)] + \
    [[month, test_rest] for month in range(2, 36)]
    postpartum_abstinent = np.array(postpartum_abstinence_rates)

    postpartum_activity = {}
    postpartum_activity['month'] = postpartum_abstinent[:, 0]
    postpartum_activity['percent_active'] = 1 - postpartum_abstinent[:, 1]

    return postpartum_activity


def default_lactational_amenorrhea(test_value=0.1):
    '''
    Returns an array of the percent of women by month postpartum 0-11 months who meet criteria for LAM:
    Exclusively breastfeeding, menses have not returned.  Extended out 5-11 months to better match data
    as those women continue to be postpartum insusceptible.

    Args:
        test_value probability of LAM per month
    '''

    data = np.array([[month, test_value] for month in range(0,12)])

    lactational_amenorrhea = {}
    lactational_amenorrhea['month'] = data[:, 0]
    lactational_amenorrhea['rate'] = data[:, 1]

    return lactational_amenorrhea


def _data2interp(data, ages, normalize=False):
    ''' Convert unevenly spaced data into an even spline interpolation '''
    model = si.interp1d(data[0], data[1])
    interp = model(ages)
    if normalize:
        interp = np.minimum(1, np.maximum(0, interp))
    return interp


def default_miscarriage_rates(test_value=0):
    '''
    Returns a linear interpolation of the likelihood of a miscarriage by age.

    Args:
        test_value rate of miscarriage for every age bucket
    '''
    age_buckets = [0, 5, 10, 12.5, 15, 20, 25, 30, 35, 40, 45, 50]
    rates_by_age = [test_value] * len(age_buckets)
    miscarriage_rates = np.array([
        age_buckets,
        rates_by_age])
    miscarriage_interp = _data2interp(miscarriage_rates, fpd.spline_preg_ages)
    return miscarriage_interp


def default_fecundity_ratio_nullip(test_value=1.0):
    '''
    Returns an array of fecundity ratios for a nulliparous woman vs a gravid woman

    Default of 1.0 means that they are the same (nulliparous and gravid have
    the same fecundity)

    Args:
        test_value multiplier on fecundity ratio for all nulliparous agents
    '''

    age_buckets = [0, 5, 10, 12.5, 15, 18, 20, 25, 30, 34, 37, 40, 45, 50]
    age_ratios = [test_value] * len (age_buckets)
    fecundity_ratio_nullip = np.array([age_buckets,
                                       age_ratios])
    fecundity_nullip_interp = _data2interp(fecundity_ratio_nullip, fpd.spline_preg_ages)

    return fecundity_nullip_interp


def default_exposure_correction_age(test_value=1.0):
    '''
    Returns an array of experimental factors to be applied to account for
    residual exposure to either pregnancy or live birth by age.  Exposure to pregnancy will
    increase factor number and residual likelihood of avoiding live birth (mostly abortion,
    also miscarriage), will decrease factor number

    The default value is a multiplier of 1, equivalent to no effect.

    Args:
        test_value multiplier on exposure correction for all agents
    '''
    age_buckets = [0, 5, 10, 12.5, 15, 18, 20, 25, 30, 35, 40, 45, 50]
    default_corrections = [test_value] * len(age_buckets)

    exposure_correction_age = np.array([age_buckets,
                                        default_corrections])
    exposure_age_interp = _data2interp(exposure_correction_age, fpd.spline_preg_ages)

    return exposure_age_interp


def default_birth_spacing_preference(test_spacing_multiplier=1.0):
    '''
    Returns an array of birth spacing preferences by closest postpartum month.
    Applied to postpartum pregnancy likelihoods.

    NOTE: spacing bins must be uniform!
    Args:
        test_spacing_multiplier on sexual activity during postpartum months
    '''
    uniform_spacing = np.array([[month, test_spacing_multiplier] for month in range(0, 49, 12)])

    # Calculate the intervals and check they're all the same
    intervals = np.diff(uniform_spacing[:, 0])
    interval = intervals[0]
    assert np.all(intervals == interval), f'In order to be computed in an array, birth spacing preference bins must be equal width, not {intervals}'
    pref_spacing = {}
    pref_spacing['interval'] = interval # Store the interval (which we've just checked is always the same)
    pref_spacing['n_bins'] = len(intervals) # Actually n_bins - 1, but we're counting 0 so it's OK
    pref_spacing['preference'] = uniform_spacing[:, 1] # Store the actual birth spacing data

    return pref_spacing


def default_exposure_correction_parity(test_multiplier=1):
    '''
    Returns an array of experimental factors to be applied to account for residual exposure to either pregnancy
    or live birth by parity. The default value is a multiplier of 1, equivalent to no effect.

    Args:
        test_multiplier multiplier on exposure correction for every parity bucket
    '''

    parity_level = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20]
    parity_corrections = [test_multiplier] * len(parity_level)
    exposure_correction_parity = np.array([parity_level,
                                           parity_corrections])
    exposure_parity_interp = _data2interp(exposure_correction_parity, fpd.spline_parities)

    return exposure_parity_interp


def default_stillbirth(test_value=20.0):
    '''Sets stillbirth rate to constant

    Args:
        test_value rate per 1000 live births
    '''

    data = np.array([
        [2000, test_value],
        [2010, test_value],
        [2020, test_value]
    ])

    stillbirth_rate = {
        'year': data[:,0],
        'probs': data[:,1]/1000
    }

    return stillbirth_rate


def make_pars(configuration_file=None, defaults_file=None):
    #
    # User-tunable parameters
    #

    defaults = {
        'name': 'Default',
        'n': 5000,
        'start_year': 1960,
        'end_year': 2019,
        'timestep': 1,
        'verbose': 1,
        'seed': 1,
        'fecundity_variation_low': 0.4,
        'fecundity_variation_high': 1.4,
        'method_age': 15,
        'max_age': 99,
        'preg_dur_low': 9,
        'preg_dur_high': 9,
        'switch_frequency': 12,
        'breastfeeding_dur_low': 1,
        'breastfeeding_dur_high': 24,
        'age_limit_fecundity': 50,
        'postpartum_length': 24,
        'end_first_tri': 3,
        'abortion_prob': 0.1,
        'twins_prob': 0.018,
        'LAM_efficacy': 0.98,
        'maternal_mortality_multiplier': 1
    }

    pars = sc.dcp(defaults)

    # Complicated parameters
    pars['methods'] = default_methods()
    pars['methods_postpartum'] = default_methods_postpartum()
    pars['age_pyramid'] = default_age_pyramid()
    pars['age_mortality'] = default_age_mortality(is_bound=True)
    pars['age_fecundity'] = default_female_age_fecundity(
        is_bound=True)  # Changed to age_fecundity for now from age_fertility for use with LEMOD
    pars['method_efficacy'] = default_efficacy()
    pars['barriers'] = default_barriers()
    pars['maternal_mortality'] = default_maternal_mortality()
    pars['infant_mortality'] = default_infant_mortality()
    pars[
        'sexual_activity'] = default_sexual_activity()  # Returns linear interpolation of annual sexual activity based on age
    pars[
        'sexual_activity_postpartum'] = default_sexual_activity_postpartum()  # Returns array of likelihood of resuming sex per postpartum month
    pars['pref_spacing'] = default_birth_spacing_preference()
    pars['lactational_amenorrhea'] = default_lactational_amenorrhea()
    pars['miscarriage_rates'] = default_miscarriage_rates()
    pars['stillbirth_rate'] = default_stillbirth()
    pars['fecundity_ratio_nullip'] = default_fecundity_ratio_nullip()
    pars['exposure_correction_age'] = default_exposure_correction_age()
    pars['exposure_correction_parity'] = default_exposure_correction_parity()
    pars['exposure_correction'] = 1  # Overall exposure correction factor

    return pars
