"""
9/19/23, Emily Driano

This script can be run to pull the country data from the UN Data Portal used for calibration. If the 'get' options below
are set to True, this script will utilize the UN Data Portal API to scrape the data - and create files for -
contraceptive prevalence rate (cpr.csv), mortality probability (mortality_prob.csv), mortality trend (mortality_trend.csv), and
age-specific fertility rate (asfr.csv). It also scrapes the World Population Prospects (WPP) for the annual world pop data, which
is stored in a WPP csv in data_processing (which is used in the {country}.py file to create the pop pyramid).

The only setup required is to set the country name (all lower-case) and location ID (which can be found in the file
country_codes.csv, also located in the data_processing folder). The script will store any created files in a temporary
location for each respective country, in data_processing/scraped_data/{country_name}. As the WPP files are location-agnostic,
they are stored in data_processing/scraped_data.

For more information about the UN Data Portal API, visit this site: https://population.un.org/dataportal/about/dataapi
For more information about the UN World Population Prospects: https://population.un.org/wpp/Download/Standard/CSV/

"""

import os
import pandas as pd
import requests
import sciris as sc
import zipfile

# COUNTRY CODES FOR REFERENCE
# Senegal ID = 686
# Kenya ID = 404
# Ethiopia ID = 231
# India ID = 356

# Country name and location id(s) (per UN Data Portal); User must update these two variables prior to running.
country = 'ethiopia'
location_id = 686

# Default global variables
startYear = 1960
endYear = 2030

# Variables used in pulling WPP data
years = ['1950-2021']
pop_stem = 'WPP2022_Population1JanuaryByAge5GroupSex_Medium'
female_mort_stem = 'WPP2022_Life_Table_Complete_Medium_Female_1950-2021'
male_mort_stem = 'WPP2022_Life_Table_Complete_Medium_Male_1950-2021'
thisdir = sc.path(sc.thisdir())
filesdir = thisdir / 'scraped_data'

# Set options for web scraping of data; all True by default
get_cpr = True  # Contraceptive prevalence rate
get_mortality_prob = True  # Mortality prob
get_mortality_trend = True  # Mortality trend
get_asfr = True  # Age-specific fertility rate
get_pop = True  # Population pyramid (5-year age groups for both male/female sex)

# API Base URL
base_url = "https://population.un.org/dataportalapi/api/v1"
wpp_base_url = "https://population.un.org/wpp/Download/Files/1_Indicators%20(Standard)/CSV_FILES/"

# If country folder doesn't already exist, create it (location in which created data files will be stored)
if not os.path.exists(f'{filesdir}/{country}'):
    os.makedirs(f'{filesdir}/{country}')

# Function that calls a GET request to the UN Data Portal API given the target/uri specified
def get_data(target):
    '''
    Scrapes data from UN data portal using the UN Data Portal API
    https://population.un.org/dataportal/about/dataapi
    '''
    # Get the response, which includes the first page of data as well as information on pagination and number of records
    response = requests.get(target)

    # Converts call into JSON
    j = response.json()

    # Converts JSON into a pandas DataFrame.
    df = pd.json_normalize(
        j['data'])  # pd.json_normalize flattens the JSON to accommodate nested lists within the JSON structure

    # Loop until there are new pages with data
    while j['nextPage'] != None:
        # Reset the target to the next page
        target = j['nextPage']

        # call the API for the next page
        response = requests.get(target)

        # Convert response to JSON format
        j = response.json()

        # Store the next page in a data frame
        df_temp = pd.json_normalize(j['data'])

        # Append next page to the data frame
        df = pd.concat([df, df_temp])
        print("writing file...")

    return df


def get_UN_data(label='', file_stem=None, outfile=None, columns=None, force=None, tidy=None):
    ''' Download data from UN Population Division '''
    if force is None: force = False
    if tidy  is None: tidy  = True

    sc.heading(f'Downloading {label} data...')
    T = sc.timer()

    # Download data if it's not already in the directory
    url = f'{wpp_base_url}{file_stem}.zip'
    local_csv = f'{filesdir}/{file_stem}.csv'
    local_zip = f'{filesdir}/{file_stem}.zip'
    if force or not os.path.exists(local_csv):
        print(f'\nDownloading from {url}, this may take a while...')
        sc.download(url, filename=local_zip)
        zip_file_object = zipfile.ZipFile(local_zip, 'r')
        zip_file_object.extractall(filesdir)
        zip_file_object.close()

        # Extract the parts used in the model and save
        df = pd.read_csv(local_csv, usecols=columns)
        df.to_csv(f'{local_csv}')
        if tidy:
            print(f'Removing {local_zip}')
            os.remove(local_zip)
    else:
        print(f'Skipping {local_csv}, already downloaded')

    T.toctic(label=f'  Done with {label}')

    T.toc(doprint=False)
    print(f'Done with {label}: took {T.timings[:].sum():0.1f} s.')

    return


def get_pop_data(force=None, tidy=None):
    ''' Import population sizes by age from UNPD '''
    columns = ["Location", "Time", "AgeGrpStart", "PopMale", "PopFemale"]
    outfile = f'{pop_stem}.csv'
    kw = dict(label='pop', file_stem=pop_stem, outfile=outfile, columns=columns, force=force, tidy=tidy)
    return get_UN_data(**kw)


def get_age_data(force=None, tidy=None, file_stem=None, outfile=None):
    ''' Import population sizes by age from UNPD '''
    columns = ["Location", "Time", "AgeGrpStart", "qx"]
    kw = dict(label='age', file_stem=file_stem, outfile=outfile, columns=columns, force=force, tidy=tidy)
    return get_UN_data(**kw)


# Called if creating country file cpr.csv
if get_cpr:
    # Set params used in API
    cpr_ind = 1
    mcpr_ind = 2

    # Call API
    target = base_url + f"/data/indicators/{cpr_ind},{mcpr_ind}/locations/{location_id}/start/{startYear}/end/{endYear}"
    df = get_data(target)

    # Reformat data
    df_cpr = df.loc[(df['category'] == 'All women') & (df['variantLabel'] == 'Median') & (df['indicatorId'] == cpr_ind)]
    df_mcpr = df.loc[(df['category'] == 'All women') & (df['variantLabel'] == 'Median') & (df['indicatorId'] == mcpr_ind)]
    df_cpr = df_cpr.filter(['timeLabel', 'value'])
    df_mcpr = df_mcpr.filter(['timeLabel', 'value'])
    df_cpr.rename(columns={'timeLabel': 'year', 'value': 'cpr'}, inplace=True)
    df_mcpr.rename(columns={'timeLabel': 'year', 'value': 'mcpr'}, inplace=True)
    df2 = pd.merge(df_cpr, df_mcpr, on='year')

    # Save file
    df2.to_csv(f'{filesdir}/{country}/cpr.csv', index=False)

# Called if creating country file mortality_prob.csv
if get_mortality_prob:
    get_age_data(file_stem=female_mort_stem, outfile=f'{female_mort_stem}.csv')
    get_age_data(file_stem=male_mort_stem, outfile=f'{male_mort_stem}.csv')

    # Load female data from scraped data
    df = pd.read_csv(f'{filesdir}/{female_mort_stem}.csv')
    df_female = df.loc[(df['Location']==country.capitalize()) & (df['Time']==df['Time'].max())]
    df_female = df_female.filter(["AgeGrpStart", "qx"])
    df_female.rename(columns={'AgeGrpStart': 'age', 'qx': 'female'}, inplace=True)

    # Load male data from scraped data
    df = pd.read_csv(f'{filesdir}/{male_mort_stem}.csv')
    df_male = df.loc[(df['Location']==country.capitalize()) & (df['Time']==df['Time'].max())]
    df_male = df_male.filter(["AgeGrpStart", "qx"])
    df_male.rename(columns={'AgeGrpStart': 'age', 'qx': 'male'}, inplace=True)

    # Combine two dataframes
    df_combined = pd.merge(df_female, df_male, on="age")

    # Save file
    df_combined.to_csv(f'{filesdir}/{country}/mortality_prob.csv', index=False)

# Called if creating country file mortality_trend.csv
if get_mortality_trend:
    # Set params used in API
    ind = 59
    startYear = 1950

    # Call API
    target = base_url + f"/data/indicators/{ind}/locations/{location_id}/start/{startYear}/end/{endYear}"
    df = get_data(target)

    # Reformat data
    df = df.filter(['timeLabel', 'value'])
    df.rename(columns={'timeLabel': 'year', 'value': 'crude_death_rate'}, inplace=True)

    # Save file
    df.to_csv(f'{filesdir}/{country}/mortality_trend.csv', index=False)

# Called if creating country file asfr.csv
if get_asfr:
    # Set params used in API
    ind = 17
    startYear = 1950
    endYear = 2022

    # Call API
    target = base_url + f"/data/indicators/{ind}/locations/{location_id}/start/{startYear}/end/{endYear}"
    df = get_data(target)

    # Reformat data
    df = df.loc[(df['variantLabel'] == 'Median')]
    df.rename(columns={'timeLabel': 'year'}, inplace=True)
    df = df.pivot(index='year', columns='ageLabel', values='value')
    df = df.round(decimals=3).drop('50-54', axis=1)

    # Save file
    df.to_csv(f'{filesdir}/{country}/asfr.csv')

# Called if scraping population data from WPP, which is used by the {country}.py file in creating the population pyramid
if get_pop:
    get_pop_data()
