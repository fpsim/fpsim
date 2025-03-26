"""
Process datafiles - this file contains functions common to all locations
"""

import numpy as np
import pandas as pd
import sciris as sc
from scipy import interpolate as si
from fpsim import defaults as fpd
from fpsim import utils as fpu
# %% Housekeeping

def this_dir():
    thisdir = sc.path(sc.thisdir(__file__))  # For loading CSV files
    return thisdir


def data2interp(data, ages, normalize=False):
    ''' Convert unevenly spaced data into an even spline interpolation '''
    model = si.interp1d(data[0], data[1])
    interp = model(ages)
    if normalize:
        interp = np.minimum(1, np.maximum(0, interp))
    return interp


def filenames(location):
    ''' Data files for use with calibration, etc -- not needed for running a sim '''
    files = {}
    files['base'] = sc.thisdir(aspath=True) / 'data'
    files['basic_dhs'] = 'basic_dhs.yaml' # From World Bank https://data.worldbank.org/indicator/SH.STA.MMRT?locations=KE
    files['popsize'] = 'popsize.csv' # Downloaded from World Bank: https://data.worldbank.org/indicator/SP.POP.TOTL?locations=KE
    files['mcpr'] = 'cpr.csv'  # From UN Population Division Data Portal, married women 1970-1986, all women 1990-2030
    files['tfr'] = 'tfr.csv'   # From World Bank https://data.worldbank.org/indicator/SP.DYN.TFRT.IN?locations=KE
    files['asfr'] = 'asfr.csv' # From UN World Population Prospects 2022: https://population.un.org/wpp/Download/Standard/Fertility/
    files['ageparity'] = 'ageparity.csv' # Choose from either DHS 2014 or PMA 2022
    files['spacing'] = 'birth_spacing_dhs.csv'
    files['methods'] = 'mix.csv'
    files['afb'] = 'afb.table.csv'
    files['use'] = 'use.csv'
    # files['empowerment'] = 'empowerment.csv'
    files['education'] = 'edu_initialization.csv'
    return files


# %% Demographics and pregnancy outcome

def urban_proportion(location):
    """Load information about the proportion of people who live in an urban setting"""
    urban_data = pd.read_csv(this_dir() / 'data' / 'urban.csv')
    return urban_data["mean"][0]  # Return this value as a float


def age_mortality(location):
    '''
    Age-dependent mortality rates taken from UN World Population Prospects 2022.  From probability of dying each year.
    https://population.un.org/wpp/
    Used CSV WPP2022_Life_Table_Complete_Medium_Female_1950-2021, Kenya, 2010
    Used CSV WPP2022_Life_Table_Complete_Medium_Male_1950-2021, Kenya, 2010
    Mortality rate trend from crude death rate per 1000 people, also from UN Data Portal, 1950-2030:
    https://population.un.org/dataportal/data/indicators/59/locations/404/start/1950/end/2030/table/pivotbylocation
    Projections go out until 2030, but the csv file can be manually adjusted to remove any projections and stop at your desired year
    '''
    data_year = 2010
    mortality_data = pd.read_csv(this_dir() / location / 'data' / 'mortality_prob.csv')
    mortality_trend = pd.read_csv(this_dir() / 'data' / 'mortality_trend.csv')

    mortality = {
        'ages': mortality_data['age'].to_numpy(),
        'm': mortality_data['male'].to_numpy(),
        'f': mortality_data['female'].to_numpy()
    }

    mortality['year'] = mortality_trend['year'].to_numpy()
    mortality['probs'] = mortality_trend['crude_death_rate'].to_numpy()
    trend_ind = np.where(mortality['year'] == data_year)
    trend_val = mortality['probs'][trend_ind]

    mortality['probs'] /= trend_val  # Normalize around data year for trending
    m_mortality_spline_model = si.splrep(x=mortality['ages'],
                                         y=mortality['m'])  # Create a spline of mortality along known age bins
    f_mortality_spline_model = si.splrep(x=mortality['ages'], y=mortality['f'])
    m_mortality_spline = si.splev(fpd.spline_ages,
                                  m_mortality_spline_model)  # Evaluate the spline along the range of ages in the model with resolution
    f_mortality_spline = si.splev(fpd.spline_ages, f_mortality_spline_model)
    m_mortality_spline = np.minimum(1, np.maximum(0, m_mortality_spline))  # Normalize
    f_mortality_spline = np.minimum(1, np.maximum(0, f_mortality_spline))

    mortality['m_spline'] = m_mortality_spline
    mortality['f_spline'] = f_mortality_spline

    return mortality


def _check_age_endpoints(df):
    """
    Add 'age endpoints' 0 and fpd.max_age+1 if they are not present in the
    add rows of the dataframe. This is needed for extrapolation if we want to
    have a complete representation of the probabilities across all possible
    ages.

    TODO:PSL: remove after we decide whether we want to have full arrays of
    data, meaning arrays that have data/probs/metric every age represented
    in the range [0-fpd.max_age],
    as opposed to arrays that have a subset of ages represented. The
    difference between the two approaches is how to correctly index
    into the data array.

    Pseudo-code:
    A: with age extrapolation
    age = ppl.age
    empwr_data[int(age)][empwr_metric]

    B: without age extrapolation, we don't need this function _check_age_endpoints()
    age = ppl.age
    age_cutofs = np.hstack(empwr_min_age, empwr_max_age) ## assumes contignuous age range between min, max age
    empwr_age_index = np.digitize(age, age_cuttofs)
    empwr_data[empwr_age_index][empwr_metric]

    A: can be faster than B as we don't have to digitize the agents ages each
    time we want to access the data, we only need to cast ages into integers.
    continous ages - > integer ages > index empowerment data

    B: removes the need for extrapolation and choosing the extrapolation function.
    But we won't be able to directly index using an integer version of the agent ages.

    continuous ages -> binned ages > bin indices -> index empowerment probs

    """

    rows_to_add = []
    for age in [0, fpd.max_age+1]:
        if age not in df['age'].values:
            new_row = pd.Series([0.0] * len(df.columns), index=df.columns)
            new_row['age'] = age
            rows_to_add.append(new_row)

    df = pd.concat([df, pd.DataFrame(rows_to_add)], ignore_index=True)
    df.sort_values('age', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def age_spline(which, location):
    d = pd.read_csv(this_dir() / 'data' / f'splines_{which}.csv')
    # Set the age as the index
    d.index = d.age
    return d


def age_partnership(location):
    """ Probabilities of being partnered at age X"""
    age_partnership_data = pd.read_csv(this_dir() / 'data' / 'age_partnership.csv')
    partnership_dict = {}
    partnership_dict["age"] = age_partnership_data["age_partner"].to_numpy()
    partnership_dict["partnership_probs"] = age_partnership_data["percent"].to_numpy()
    return  partnership_dict


def wealth(location):
    """ Process percent distribution of people in each wealth quintile"""
    cols = ["quintile", "percent"]
    wealth_data = pd.read_csv(this_dir() / 'data' / 'wealth.csv', header=0, names=cols)
    return wealth_data


# %% Education
def education_objective(df):
    """
    Transforms education objective data from a DataFrame into a numpy array. The DataFrame represents
    the proportion of women with different education objectives, stratified by urban/rural residence.

    The 'percent' column represents the proportion of women aiming for 'edu' years of education. The data
    is based on education completed by women over 20 with no children, stratified by urban/rural residence
    from the Demographic and Health Surveys (DHS).

    Args:
        df (pd.DataFrame): Contains 'urban', 'edu' and 'percent' columns, with 'edu' ranging from
                           0 to a maximum value representing years of education.

    Returns:
        arr (np.array): A 2D array of shape (2, n_edu_objective_years) containing proportions. The first
                        row corresponds to 'rural' women and the second row to 'urban' women.
    """
    arr = df["percent"].to_numpy().reshape(df["urban"].nunique(), df["edu"].nunique())
    return arr


def education_attainment(df):
    """
    Transforms education attainment data (from education_initialization.csv) from a DataFrame
    into a numpy array. The DataFrame represents the average (mean) number of years of
    education 'edu', a woman aged 'age' has attained/completed.

    Args:
        df (pd.DataFrame): Contains 'age', 'edu' columns.

    Returns:
        arr (np.array): A 1D with array with interpolated values of 'edu' years attained.

    NOTE: The data in education_initialization.csv have been extrapolated to cover the age range
    [0, 99], inclusive range. Here we only interpolate data for the group 15-49 (inclusive range).
    """
    # This df has columns
    # age:age in years and edu: mean years of education
    df.sort_values(by="age", ascending=True, inplace=True)
    ages = df["age"].to_numpy()
    arr  = df["edu"].to_numpy()

    # Get indices of those ages
    inds = np.array(sc.findinds(ages >= fpd.min_age, ages < fpd.max_age_preg+5)) # interpolate beyond DHS max age to avoid discontinuities
    arr[inds] = sc.smooth(arr[inds], 3)
    return arr


def education_dropout_probs(df):
    """
    Transforms education dropout probabilities (from edu_stop.csv) from a DataFrame
    into a dictionary. The dataframe will be used as the probability that a woman aged 'age'
    and with 'parity' number of children, would drop out of school if she is enrolled in
    education and becomes pregnant.

    The data comes from PMA Datalab and 'percent' represents the probability of stopping/droppping
    out of education within 1 year of the birth represented in 'parity'.
        - Of women with a first birth before age 18, 12.6% stopped education within 1 year of that birth.
        - Of women who had a subsequent (not first) birth before age 18, 14.1% stopped school within 1 year of that birth.

    Args:
        df (pd.DataFrame): Contains 'age', 'parity' and 'percent' columns.

    Returns:
       data (dictionary): Dictionary with keys '1' and '2+' indicating a woman's parity. The values are dictionaries,
           each with keys 'age' and 'percent'.
    """
    data = {}
    for k in df["parity"].unique():
        data[k] = {"age": None, "percent": None}
        data[k]["age"] = df["age"].unique()
        data[k]["percent"] = df["percent"][df["parity"] == k].to_numpy()
    return data


def education_distributions(location):
    """
    Loads and processes all education-related data files. Performs additional interpolation
    to reduce noise from empirical data.

    Returns:
        Tuple[dict, dict]: A tuple of two dictionaries:
            education_data (dict): Contains the unmodified empirical data from CSV files.
            education_dict (dict): Contains the processed empirical data to use in simulations.
    """

    # Load empirical data
    data_path = this_dir() / "data"

    education ={"edu_objective":
                    {"data_file": "edu_objective.csv",
                     "process_function": education_objective},
                "edu_attainment":
                    {"data_file": "edu_initialization.csv",
                     "process_function": education_attainment},
                "edu_dropout_probs":
                    {"data_file": "edu_stop.csv",
                     "process_function": education_dropout_probs}}

    education_data = dict()
    education_dict = dict()
    for edu_key in education.keys():
        education_data[edu_key] = pd.read_csv(data_path / education[edu_key]["data_file"])
        education_dict[edu_key] = education[edu_key]["process_function"](education_data[edu_key])
    education_dict.update({"age_start": 6.0})
    return education_dict, education_data


def process_contra_use_pars(location):
    raw_pars = pd.read_csv(this_dir() / 'data' / 'contra_coef.csv')
    pars = sc.objdict()
    for var_dict in raw_pars.to_dict('records'):
        var_name = var_dict['rhs'].replace('_0', '').replace('(', '').replace(')', '').lower()
        pars[var_name] = var_dict['Estimate']
    return pars

 
def process_contra_use(which, location):
    """
    Process cotraceptive use parameters.
    Args:
        which: either 'simple' or 'mid'
    """

    # Read in data
    alldfs = [
        pd.read_csv(this_dir() / 'data' / f'contra_coef_{which}.csv'),
        pd.read_csv(this_dir() / 'data' / f'contra_coef_{which}_pp1.csv'),
        pd.read_csv(this_dir() / 'data' / f'contra_coef_{which}_pp6.csv'),
    ]

    contra_use_pars = dict()

    for di, df in enumerate(alldfs):
        if which == 'mid':
            contra_use_pars[di] = sc.objdict(
                intercept=df[df['rhs'].str.contains('Intercept')].Estimate.values[0],
                age_factors=df[df['rhs'].str.contains('age') & ~df['rhs'].str.contains('prior_userTRUE')].Estimate.values,
                ever_used_contra=df[df['rhs'].str.contains('prior_userTRUE') & ~df['rhs'].str.contains('age')].Estimate.values[0],
                edu_factors=df[df['rhs'].str.contains('edu')].Estimate.values,
                parity=df[df['rhs'].str.contains('parity')].Estimate.values[0],
                urban=df[df['rhs'].str.contains('urban')].Estimate.values[0],
                wealthquintile=df[df['rhs'].str.contains('wealthquintile')].Estimate.values[0],
                age_ever_user_factors=df[df['rhs'].str.contains('age') & df['rhs'].str.contains('prior_userTRUE')].Estimate.values,
            )

        elif which == 'simple':
            contra_use_pars[di] = sc.objdict(
                intercept=df[df['rhs'].str.contains('Intercept')].Estimate.values[0],
                age_factors=df[df['rhs'].str.match('age') & ~df['rhs'].str.contains('fp_ever_user')].Estimate.values,
                fp_ever_user=df[df['rhs'].str.contains('prior_userTRUE') & ~df['rhs'].str.contains('age')].Estimate.values[0],
                age_ever_user_factors=df[df['rhs'].str.match('age') & df['rhs'].str.contains('prior_userTRUE')].Estimate.values,
            )

    return contra_use_pars


def process_markovian_method_choice(methods, location, df=None):
    """ Choice of method is age and previous method """
    if df is None:
        df = pd.read_csv(this_dir() / 'data' / 'method_mix_matrix_switch.csv', keep_default_na=False, na_values=['NaN'])
    csv_map = {method.csv_name: method.name for method in methods.values()}
    idx_map = {method.csv_name: method.idx for method in methods.values()}
    idx_df = {}
    for col in df.columns:
        if col in csv_map.keys():
            idx_df[col] = idx_map[col]

    mc = dict()  # This one is a dict because it will be keyed with numbers
    init_dist = sc.objdict()  # Initial distribution of method choice

    # Get end index
    ei = 4+len(methods)-1

    for pp in df.postpartum.unique():
        mc[pp] = sc.objdict()
        mc[pp].method_idx = np.array(list(idx_df.values()))
        for akey in df.age_grp.unique():
            mc[pp][akey] = sc.objdict()
            thisdf = df.loc[(df.age_grp == akey) & (df.postpartum == pp)]
            if pp == 1:  # Different logic for immediately postpartum
                mc[pp][akey] = thisdf.values[0][4:ei].astype(float)  # If this is going to be standard practice, should make it more robust
            else:
                from_methods = thisdf.From.unique()
                for from_method in from_methods:
                    from_mname = csv_map[from_method]
                    row = thisdf.loc[thisdf.From == from_method]
                    mc[pp][akey][from_mname] = row.values[0][4:ei].astype(float)

            # Set initial distributions by age
            if pp == 0:
                init_dist[akey] = thisdf.loc[thisdf.From == 'None'].values[0][4:ei].astype(float)
                init_dist.method_idx = np.array(list(idx_df.values()))

    return mc, init_dist


def process_dur_use(methods, location, df=None):
    """ Process duration of use parameters"""
    if df is None:
        df = pd.read_csv(this_dir() / 'data' / 'method_time_coefficients.csv', keep_default_na=False, na_values=['NaN'])
    for method in methods.values():
        if method.name == 'btl':
            method.dur_use = dict(dist='unif', par1=1000, par2=1200)
        else:
            mlabel = method.csv_name

            thisdf = df.loc[df.method == mlabel]
            dist = thisdf.functionform.iloc[0]
            method.dur_use = dict()
            age_ind = sc.findfirst(thisdf.coef.values, 'age_grp_fact(0,18]')
            method.dur_use['age_factors'] = thisdf.estimate.values[age_ind:]

            if dist in ['lognormal', 'lnorm']:
                method.dur_use['dist'] = 'lognormal_sps'
                method.dur_use['par1'] = thisdf.estimate[thisdf.coef == 'meanlog'].values[0]
                method.dur_use['par2'] = thisdf.estimate[thisdf.coef == 'sdlog'].values[0]
            elif dist in ['gamma']:
                method.dur_use['dist'] = dist
                method.dur_use['par1'] = thisdf.estimate[thisdf.coef == 'shape'].values[0]
                method.dur_use['par2'] = thisdf.estimate[thisdf.coef == 'rate'].values[0]
            elif dist == 'llogis':
                method.dur_use['dist'] = dist
                method.dur_use['par1'] = thisdf.estimate[thisdf.coef == 'shape'].values[0]
                method.dur_use['par2'] = thisdf.estimate[thisdf.coef == 'scale'].values[0]
            elif dist == 'weibull':
                method.dur_use['dist'] = dist
                method.dur_use['par1'] = thisdf.estimate[thisdf.coef == 'shape'].values[0]
                method.dur_use['par2'] = thisdf.estimate[thisdf.coef == 'scale'].values[0]
            elif dist == 'exponential':
                method.dur_use['dist'] = dist
                method.dur_use['par1'] = thisdf.estimate[thisdf.coef == 'rate'].values[0]
                method.dur_use['par2'] = None
            else:
                errormsg = f"Duration of use distribution {dist} not recognized"
                raise ValueError(errormsg)

    return methods


def mcpr(location):

    mcpr = {}
    cpr_data = pd.read_csv(this_dir() / 'data' / 'cpr.csv')
    mcpr['mcpr_years'] = cpr_data['year'].to_numpy()
    mcpr['mcpr_rates'] = cpr_data['cpr'].to_numpy() / 100

    return mcpr

