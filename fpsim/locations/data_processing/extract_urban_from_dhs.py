"""
This script shows how to extract the mean proportion of women
living in urban areas, as well as the standard error of the mean,
from a DHS DTA file, Individual Women's Data - Individual Recode (IR).

This dataset has one record for every eligible woman
as defined by the household schedule.

NOTE: The Python ecosystem does not seem to have an equivalent to R's survey package.
Thus, results from the calculations below should be used with a clear understanding
of their statistical limitations as we do not perform the full range of survey
adjustments as carried out by svydesign in R.
"""

import pathlib

import pandas as pd
import sciris as sc

import fpsim.locations as fplocs

# Assumes datasets are stored in /home/user/DHS/...
# Replace with relevant directories and filenames
home_dir = pathlib.Path.home()
dhs_dir = "DHS"
country_dir = "KEIR8BDT"
filename = "KEIR8BFL.DTA"

# Path to DHS dataset stata file (.DTA)
filepath = sc.path(home_dir, dhs_dir, country_dir, filename)
data_raw = pd.read_stata(filepath, convert_categoricals=False)


cols_to_keep = ["v001", "v005", "v023", "v025"]
urban_dhs_var = "v025"  # DHS-8 variable name about type of place of residence

data = data_raw[cols_to_keep].copy()

#Take the 'women's individual sample weight' variable and divide by 1 million.
data["sample_weight"] = data["v005"]/1e6
data["urban"] = (data[urban_dhs_var] == 1).astype(float)
data["weighted_urban"] = data["urban"].multiply(data['sample_weight'], axis=0)

# Process the necessary information
data_processed = pd.DataFrame({
    "mean": [data["weighted_urban"].mean()],
    "urban.se": [data["weighted_urban"].sem()]
})


# If you want to write this DataFrame to a .csv file, in
# the corresponding locations folder you can do:
locations_path = sc.thisdir(fplocs)
country_dir = "kenya"
data_dir = "data"
output_filename = "urban.csv"

# Actually save the data
# NOTE that this will replace the default urban.csv
data_processed.to_csv(sc.path(locations_path,
                              country_dir,
                              data_dir,
                              output_filename),
                      index=False)
