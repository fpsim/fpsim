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
import json

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
get_maternal_mortality = True # Maternal mortality
get_infant_mortality = True # Infant mortality
get_basic_dhs = True # Includes maternal mortality, infant mortality, crude birth rate, & crude death rate
#mortality_prob = False
#mortality_rate = False

# API Base URL
base_url = "https://api.worldbank.org/v2"
url_suffix = 'format=json'


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
    target = base_url + f"/country/{iso_code}/indicators/{ind}?" + url_suffix
    df = get_data(target)

    df = df.filter(['date', 'value'])
    df = df.rename(columns={'date': 'year', 'value': 'population'})
    df.dropna(subset=['population'], inplace=True)
    df = df.sort_values('year')

    df.to_csv('popsize.csv', index=False)


if get_tfr:
    # Set params used in API
    ind = 'SP.DYN.TFRT.IN'

    # Call API
    target = base_url + f"/country/{iso_code}/indicators/{ind}?" + url_suffix
    df = get_data(target)

    df = df.filter(['date', 'value'])
    df = df.rename(columns={'date': 'year', 'value': 'tfr'})
    df.dropna(subset=['tfr'], inplace=True)
    df = df.sort_values('year')

    df.to_csv('tfr.csv', index=False)

if get_maternal_mortality:
    # Set params used in API
    ind = 'SH.STA.MMRT'

    # Call API
    target = base_url + f"/country/{iso_code}/indicators/{ind}?" + url_suffix
    df = get_data(target)

    df = df.filter(['date', 'value'])
    df = df.rename(columns={'date': 'year', 'value': 'mmr'})
    df.dropna(subset=['mmr'], inplace=True)
    df['mmr'] = df['mmr'].astype(int)
    df = df.sort_values('year')

    df.to_csv('maternal_mortality.csv', index=False)

if get_infant_mortality:
    # Set params used in API
    ind = 'SP.DYN.IMRT.IN'

    # Call API
    target = base_url + f"/country/{iso_code}/indicators/{ind}?" + url_suffix
    df = get_data(target)

    df = df.filter(['date', 'value'])
    df = df.rename(columns={'date': 'year', 'value': 'imr'})
    df.dropna(subset=['imr'], inplace=True)
    df = df.sort_values('year')

    df.to_csv('infant_mortality.csv', index=False)

if get_basic_dhs:
    # Get the most recent year of data in infant & maternal mortality data
    im_data = pd.read_csv(f'infant_mortality.csv')
    mm_data = pd.read_csv(f'maternal_mortality.csv')

    latest_data_year = im_data.iloc[-1]['year']
    if mm_data.iloc[-1]['year'] < latest_data_year:
        latest_data_year = mm_data.iloc[-1]['year']

    imr = im_data.loc[im_data['year'] == latest_data_year, 'imr'].iloc[0]
    mmr = float(mm_data.loc[mm_data['year'] == latest_data_year, 'mmr'].iloc[0])

    # Crude birth and death rate indicators
    cbr_ind = 'SP.DYN.CBRT.IN'
    cdr_ind = 'SP.DYN.CDRT.IN'

    # Get CBR for latest_data_year
    target = base_url + f"/country/{iso_code}/indicators/{cbr_ind}?date={latest_data_year}&" + url_suffix
    df = get_data(target)
    cbr = df['value'].values[0]

    # Get CDR for latest_data_year
    target = base_url + f"/country/{iso_code}/indicators/{cdr_ind}?date={latest_data_year}&" + url_suffix
    df = get_data(target)
    cdr = df['value'].values[0]

    # Define dictionary for basic_dhs_csv
    basic_dhs_data = {
        'maternal_mortality_ratio': mmr, # Per 100,000 live births, (2017) From World Bank: https://data.worldbank.org/indicator/SH.STA.MMRT?locations=KE
        'infant_mortality_rate': imr,  # Per 1,000 live births, From World Bank
        'crude_death_rate': cdr,  # Per 1,000 inhabitants, From World Bank
        'crude_birth_rate': cbr,  # Per 1,000 inhabitants, From World Bank
    }

    with open('basic_dhs.yaml', 'w') as file:
        file.write(json.dumps(basic_dhs_data))





