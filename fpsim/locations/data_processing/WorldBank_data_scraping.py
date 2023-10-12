'''
10/1/23, Emily Driano #TODO: Rewrite

This script can be run to pull the country data from the UN Data Portal used for calibration. If the 'get' options below
are set to True, this script will utilize the UN Data Portal API to scrape the data - and create files for -
contraceptive prevalence rate (cpr.csv), mortality probability (mortality_prob.csv), mortality trend (mortality_trend.csv),
age-specific fertility rate (asfr.csv), and the population pyramid (which is used in the {country}.py file).

The only setup required is to set the country name (all lower-case) and location ID (which can be found in the file
un_data_country_codes.csv, also located in the data_processing folder). If the country folder already exists, the script
will create the respective files and overwrite or create the files in the existing country folder. If the country folder
does not exist, the script will create a relevant country folder and create/store the newly-created files in this folder.

For more information about the UN Data Portal API, visit this site: https://population.un.org/dataportal/about/dataapi

'''
import os
import pandas as pd
import requests

# ISO2 COUNTRY CODES FOR REFERENCE
# Senegal ID = SN
# Kenya = KE
# Ethiopia = ET
# India ID = IN

# Country name and location id(s) (per UN Data Portal); User must update these two variables prior to running.
country = 'ethiopia'
iso_code = 'ET'

# Default global variables
startYear = 1960
endYear = 2030

# Set options for web scraping of data; all True by default
get_pop = True  # Annual population size
get_tfr = True # Total fertility rate
get_maternal_mortality = False # Maternal mortality
get_infant_mortality = False # Infant mortality
get_basic_dhs = False # Includes maternal mortality, infant mortality, crude birth rate, & crude death rate

# API Base URL
base_url = "https://api.worldbank.org/v2"
url_suffix = '?format=json'


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
        j[1])  # pd.json_normalize flattens the JSON to accommodate nested lists within the JSON structure

    # Loop until there are new pages with data
    while j[0]['page'] < j[0]['pages']:
        # Reset the target to the next page
        next = j[0]['page']+1
        target = target + f"&page={next}"

        # call the API for the next page
        response = requests.get(target)

        # Convert response to JSON format
        j = response.json()

        # Store the next page in a data frame
        df_temp = pd.json_normalize(j[1])

        # Append next page to the data frame
        df = pd.concat([df, df_temp])
        print("writing file...")

    # If country folder doesn't already exist, create it (location in which created data files will be stored)
    if not os.path.exists(f'../{country}'):
        os.mkdir(f'../{country}')

    return df


if get_pop:
    # Set params used in API
    ind = 'SP.POP.TOTL'

    # Call API
    target = base_url + f"/country/{iso_code}/indicators/{ind}" + url_suffix
    df = get_data(target)

    df = df.filter(['date', 'value'])
    df = df.rename(columns={'date': 'year', 'value': 'population'})
    df_sorted = df.sort_values(by=['year'])

    df.to_csv('popsize.csv', index=False)


if get_tfr:
    # Set params used in API
    ind = 'SP.DYN.TFRT.IN'

    # Call API
    target = base_url + f"/country/{iso_code}/indicators/{ind}" + url_suffix
    df = get_data(target)

    df = df.filter(['date', 'value'])
    df = df.rename(columns={'date': 'year', 'value': 'tfr'})
    df_sorted = df.sort_values(by=['year'])

    df.to_csv('tfr.csv', index=False)

if get_infant_mortality:
    mat_mort_ind = 'SH.STA.MMRT'
    inf_mort_ind = 'SP.DYN.IMRT.IN'
    cdr_ind = 'SP.DYN.CDRT.IN'
    cbr_ind = 'SP.DYN.CBRT.IN'