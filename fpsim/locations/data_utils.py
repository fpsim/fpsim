"""
Process datafiles - this file contains functions common to all locations
"""
import os
import numpy as np
import pandas as pd
import sciris as sc
import yaml
from scipy import interpolate as si
from fpsim import defaults as fpd
from fpsim import utils as fpu
import fpsim.shared_data as sd

sd_dir = os.path.dirname(sd.__file__)  # path to the shared_data directory

# %% Housekeeping and utility functions

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

def load_age_adjustments():
    with open(os.path.join(sd_dir, 'age_adjustments.yaml'), 'r') as f:
        adjustments = yaml.safe_load(f)
    return adjustments

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

# %% Scalar pars
def bf_stats(location):
    """ Load breastfeeding stats """
    bf_data = pd.read_csv(this_dir() / location / 'data' / 'bf_stats.csv')
    bf_pars = {
        'breastfeeding_dur_mean' : bf_data.loc[0]['value'],  # Location parameter of truncated norm distribution. Requires children's recode DHS file, see data_processing/breastfeeding_stats.R
        'breastfeeding_dur_sd' : bf_data.loc[1]['value']     # Location parameter of truncated norm distribution. Requires children's recode DHS file, see data_processing/breastfeeding_stats.R
    }

    return bf_pars

def scalar_probs(location):
    """ Load abortion and twins probabilities """
    data = pd.read_csv(this_dir() / location / 'data' / 'scalar_probs.csv')
    abortion_prob = data.loc[data['param']=='abortion_prob', 'prob'].values[0]   # From https://bmcpregnancychildbirth.biomedcentral.com/articles/10.1186/s12884-015-0621-1, % of all pregnancies calculated
    twins_prob = data.loc[data['param']=='twins_prob', 'prob'].values[0]         # From https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0025239

    return abortion_prob, twins_prob


# %% Demographics
def age_spline(which):
    d = pd.read_csv(os.path.join(sd_dir, f'splines_{which}.csv'))
    # Set the age as the index
    d.index = d.age
    return d


def age_partnership(location):
    """ Probabilities of being partnered at age X"""
    age_partnership_data = pd.read_csv(this_dir() / location / 'data' / 'age_partnership.csv')
    partnership_dict = {}
    partnership_dict["age"] = age_partnership_data["age_partner"].to_numpy()
    partnership_dict["partnership_probs"] = age_partnership_data["percent"].to_numpy()
    return  partnership_dict


def wealth(location):
    """ Process percent distribution of people in each wealth quintile"""
    cols = ["quintile", "percent"]
    wealth_data = pd.read_csv(this_dir() / location / 'data' / 'wealth.csv', header=0, names=cols)
    return wealth_data


def urban_proportion(location):
    """Load information about the proportion of people who live in an urban setting"""
    urban_data = pd.read_csv(this_dir() / location / 'data' / 'urban.csv')
    return urban_data["mean"][0]  # Return this value as a float


def age_pyramid(location):
    """Load age pyramid data"""
    age_pyramid_data = pd.read_csv(this_dir() / location / 'data' / 'age_pyramid.csv')
    pyramid = age_pyramid_data.to_numpy()
    return pyramid


# %% Mortality
def age_mortality(location, data_year=None):
    """
    Age-dependent mortality rates taken from UN World Population Prospects 2022.  From probability of dying each year.
    https://population.un.org/wpp/
    Used CSV WPP2022_Life_Table_Complete_Medium_Female_1950-2021, Kenya, 2010
    Used CSV WPP2022_Life_Table_Complete_Medium_Male_1950-2021, Kenya, 2010
    Mortality rate trend from crude death rate per 1000 people, also from UN Data Portal, 1950-2030:
    https://population.un.org/dataportal/data/indicators/59/locations/404/start/1950/end/2030/table/pivotbylocation
    Projections go out until 2030, but the csv file can be manually adjusted to remove any projections and stop at your desired year
    """
    mortality_data = pd.read_csv(this_dir() / location / 'data' / 'mortality_prob.csv')
    mortality_trend = pd.read_csv(this_dir() / location / 'data' / 'mortality_trend.csv')

    if data_year is None:
        error_msg = "Please provide a data year to calculate mortality rates"
        raise ValueError(error_msg)

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


def maternal_mortality(location):
    """
    From World Bank indicators for maternal mortality ratio (modeled estimate) per 100,000 live births:
    https://data.worldbank.org/indicator/SH.STA.MMRT?locations=ET

    For Senegal, data is risk of maternal death assessed at each pregnancy. Data from Huchon et al. (2013)
    prospective study on risk of maternal death in Senegal and Mali.
    Maternal deaths: The annual number of female deaths from any cause related to or aggravated by pregnancy
    or its management (excluding accidental or incidental causes) during pregnancy and childbirth or within
    42 days of termination of pregnancy, irrespective of the duration and site of the pregnancy,
    expressed per 100,000 live births, for a specified time period.
    """
    df = pd.read_csv(this_dir() / location / 'data' / 'maternal_mortality.csv')
    maternal_mortality = {}
    maternal_mortality['year'] = df['year'].values
    maternal_mortality['probs'] = df['probs'].values / 100000  # ratio per 100,000 live births
    return maternal_mortality


def infant_mortality(location):
    '''
    From World Bank indicators for infant mortality (< 1 year) for Kenya, per 1000 live births
    From API_SP.DYN.IMRT.IN_DS2_en_excel_v2_1495452.numbers
    '''
    df = pd.read_csv(this_dir() / location / 'data' / 'infant_mortality.csv')

    infant_mortality = {}
    infant_mortality['year'] = df['year'].values
    infant_mortality['probs'] = df['probs'].values / 1000  # Rate per 1000 live births, used after stillbirth is filtered out

    # Try to load age adjustments from YAML
    adjustments = load_age_adjustments()
    if 'infant_mortality' in adjustments:
        infant_mortality['ages'] = np.array(adjustments['infant_mortality']['ages'])
        infant_mortality['age_probs'] = np.array(adjustments['infant_mortality']['odds_ratios'])

    return infant_mortality


# %% Pregnancy and birth outcomes
def miscarriage():
    """
    Returns a linear interpolation of the likelihood of a miscarriage
    by age, taken from data from Magnus et al BMJ 2019: https://pubmed.ncbi.nlm.nih.gov/30894356/
    Data to be fed into likelihood of continuing a pregnancy once initialized in model
    Age 0 and 5 set at 100% likelihood.  Age 10 imputed to be symmetrical with probability at age 45 for a parabolic curve
    """
    df = pd.read_csv(os.path.join(sd_dir, 'miscarriage.csv'))

    # Extract data and interpolate
    miscarriage_rates = np.array([df['age'].values, df['prob'].values])
    miscarriage_interp = data2interp(miscarriage_rates, fpd.spline_preg_ages)
    return miscarriage_interp


def stillbirth(location):
    '''
    From Report of the UN Inter-agency Group for Child Mortality Estimation, 2020
    https://childmortality.org/wp-content/uploads/2020/10/UN-IGME-2020-Stillbirth-Report.pdf
    '''
    df = pd.read_csv(this_dir() / location / 'data' / 'stillbirths.csv')
    stillbirth_rate = {}
    stillbirth_rate['year'] = df['year'].values
    stillbirth_rate['probs'] =df['probs'].values / 1000  # Rate per 1000 total births

    # Try to load age adjustments from YAML
    adjustments = load_age_adjustments()
    if 'stillbirth' in adjustments:
        stillbirth_rate['ages'] = np.array(adjustments['stillbirth']['ages'])
        stillbirth_rate['age_probs'] = np.array(adjustments['stillbirth']['odds_ratios'])

    return stillbirth_rate


# %% Fecundity and conception
def female_age_fecundity():
    '''
    Use fecundity rates from PRESTO study: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5712257/
    Fecundity rate assumed to be approximately linear from onset of fecundity around age 10 (average age of menses 12.5) to first data point at age 20
    45-50 age bin estimated at 0.10 of fecundity of 25-27 yr olds
    '''
    df = pd.read_csv(os.path.join(sd_dir, 'age_fecundity.csv'))

    # Extract bins and fecundity values
    bins = df['bin'].values
    f = df['f'].values / 100    # Convert from per 100 to proportion

    fecundity_interp_model = si.interp1d(x=bins, y=f)
    fecundity_interp = fecundity_interp_model(fpd.spline_preg_ages)
    fecundity_interp = np.minimum(1, np.maximum(0, fecundity_interp))  # Normalize to avoid negative or >1 values

    return fecundity_interp


def fecundity_ratio_nullip():
    '''
    Returns an array of fecundity ratios for a nulliparous woman vs a gravid woman
    from PRESTO study: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5712257/
    Approximates primary infertility and its increasing likelihood if a woman has never conceived by age
    '''
    df = pd.read_csv(os.path.join(sd_dir, 'fecundity_ratio_nullip.csv'))

    # Extract data and interpolate
    fecundity_ratio_nullip = np.array([df['age'].values, df['prob'].values])
    fecundity_nullip_interp = data2interp(fecundity_ratio_nullip, fpd.spline_preg_ages)

    return fecundity_nullip_interp


def lactational_amenorrhea(location):
    '''
    Returns an array of the percent of breastfeeding women by month postpartum 0-11 months who meet criteria for LAM:
    Exclusively breastfeeding (bf + water alone), menses have not returned.  Extended out 5-11 months to better match data
    as those women continue to be postpartum insusceptible.
    Kenya: From DHS Kenya 2014 calendar data
    Ethiopia: From DHS Ethiopia 2016 calendar data
    Senegal: From DHS Senegal calendar data
    '''
    df = pd.read_csv(this_dir() / location / 'data' / 'lam.csv')
    lactational_amenorrhea = {}
    lactational_amenorrhea['month'] = df['month'].values
    lactational_amenorrhea['rate'] = df['rate'].values
    return lactational_amenorrhea


def sexual_activity(location):
    '''
    Returns a linear interpolation of rates of female sexual activity, defined as
    percentage women who have had sex within the last four weeks.
    From STAT Compiler DHS https://www.statcompiler.com/en/
    Using indicator "Timing of sexual intercourse"
    Includes women who have had sex "within the last four weeks"
    Excludes women who answer "never had sex", probabilities are only applied to agents who have sexually debuted
    Data taken from DHS, no trend over years for now
    Onset of sexual activity probabilities assumed to be linear from age 10 to first data point at age 15
    '''
    df = pd.read_csv(this_dir() / location / 'data' / 'sexually_active.csv')
    sexually_active = df['probs'].values / 100  # Convert from percent to rate per woman
    activity_ages = df['age'].values
    activity_interp_model = si.interp1d(x=activity_ages, y=sexually_active)
    activity_interp = activity_interp_model(fpd.spline_preg_ages)  # Evaluate interpolation along resolution of ages

    return activity_interp


def sexual_activity_pp(location):
    '''
    Returns an array of monthly likelihood of having resumed sexual activity within 0-35 months postpartum
    Uses 2014 Kenya DHS individual recode (postpartum (v222), months since last birth, and sexual activity within 30 days.
    Data is weighted.
    Limited to 23 months postpartum (can use any limit you want 0-23 max)
    Postpartum month 0 refers to the first month after delivery
    TODO-- Add code for processing this for other countries to data_processing
    '''
    df = pd.read_csv(this_dir() / location / 'data' / 'sexually_active_pp.csv')
    postpartum_activity = {}
    postpartum_activity['month'] = df['month'].values
    postpartum_activity['percent_active'] = df['probs'].values
    return postpartum_activity


def debut_age(location):
    """
    Returns an array of weighted probabilities of sexual debut by a certain age 10-45.
    Data taken from DHS variable v531 (imputed age of sexual debut, imputed with data from age at first union)
    Use sexual_debut_age_probs.py under locations/data_processing to output for other DHS countries
    """
    df = pd.read_csv(this_dir() / location / 'data' / 'debut_age.csv')
    debut_age = {}
    debut_age['ages'] = df['age'].values
    debut_age['probs'] = df['probs'].values

    return debut_age


def birth_spacing_pref(location):
    '''
    Returns an array of birth spacing preferences by closest postpartum month.
    Applied to postpartum pregnancy likelihoods.

    NOTE: spacing bins must be uniform!
    '''
    df = pd.read_csv(this_dir() / location / 'data' / 'birth_spacing_pref.csv')

    # Calculate the intervals and check they're all the same
    intervals = np.diff(df['month'].values)
    interval = intervals[0]
    assert np.all(
        intervals == interval), f'In order to be computed in an array, birth spacing preference bins must be equal width, not {intervals}'
    pref_spacing = {}
    pref_spacing['interval'] = interval  # Store the interval (which we've just checked is always the same)
    pref_spacing['n_bins'] = len(intervals)  # Actually n_bins - 1, but we're counting 0 so it's OK
    pref_spacing['months'] = df['month'].values
    pref_spacing['preference'] = df['weights'].values  # Store the actual birth spacing data

    return pref_spacing


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
    data_path = this_dir() / location / "data"

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

 
def process_contra_use(which, location):
    """
    Process cotraceptive use parameters.
    Args:
        which: either 'simple' or 'mid'
    """

    # Read in data
    alldfs = [
        pd.read_csv(this_dir() / location / 'data' / f'contra_coef_{which}.csv'),
        pd.read_csv(this_dir() / location / 'data' / f'contra_coef_{which}_pp1.csv'),
        pd.read_csv(this_dir() / location / 'data' / f'contra_coef_{which}_pp6.csv'),
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
            age_factors = df[df['rhs'].str.match('age') & ~df['rhs'].str.contains('prior_userTRUE')].Estimate.values
            age_factors = np.insert(age_factors, 0, 0)  # Add a zero for the 0-18 group
            contra_use_pars[di] = sc.objdict(
                intercept=df[df['rhs'].str.contains('Intercept')].Estimate.values[0],
                age_factors=age_factors,
                fp_ever_user=df[df['rhs'].str.contains('prior_userTRUE') & ~df['rhs'].str.contains('age')].Estimate.values[0],
                age_ever_user_factors=df[df['rhs'].str.match('age') & df['rhs'].str.contains('prior_userTRUE')].Estimate.values,
            )

    return contra_use_pars


def process_markovian_method_choice(methods, location, df=None):
    """ Choice of method is age and previous method """
    if df is None:
        df = pd.read_csv(this_dir() / location / 'data' / 'method_mix_matrix_switch.csv', keep_default_na=False, na_values=['NaN'])
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
        df = pd.read_csv(this_dir() / location / 'data' / 'method_time_coefficients.csv', keep_default_na=False, na_values=['NaN'])
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
    cpr_data = pd.read_csv(this_dir() / location / 'data' / 'cpr.csv')
    mcpr['mcpr_years'] = cpr_data['year'].to_numpy()
    mcpr['mcpr_rates'] = cpr_data['cpr'].to_numpy() / 100

    return mcpr


