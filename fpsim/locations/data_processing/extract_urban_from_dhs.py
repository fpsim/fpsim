"""
This script shows how to extract the mean proportion of women
living in urban areas, as well as the standard error of the mean,
from a DHS DTA file, Individual Women's Data - Individual Recode (IR).

This dataset has one record for every eligible woman
as defined by the household schedule.
"""

import pandas as pd
import sciris as sc
import pathlib
import fpsim.locations as fplocs

# Assumes datasets are stored in /home/user/DHS/etc
# Replace with relevant directory
home_dir = pathlib.Path.home()
dhs_dir = "DHS"
country_dir = "KEIR8ADT"
filename = "KEIR8AFL.DTA"

# Path to DHS dataset stata file (.DTA)
filepath = sc.path(home_dir, dhs_dir, country_dir, filename)
data_raw = pd.read_stata(filepath)

# Process the necessary information
urban_var = "v025"  # DHS-8 variable name about type of place of residence
data_processed = pd.DataFrame(columns=["mean", "urban.se"])
urban = (data_raw[urban_var] == 1).astype(float)
data_processed["mean"] = urban.mean()
data_processed["urban.se"] = urban.sem()

# If you want to write this DataFrame to a .csv file, in
# the corresponding locations folder you can do:
locations_path = sc.thisdir(fplocs)
country = "kenya"
output_filename = "urban.csv"

# Actually save the data
data_processed.to_csv(sc.path(locations_path, country,
                              output_filename),
                      index=False)
# NOTE that this will replace the default urban.csv
# TODO: maybe we automatically save a backup of default urban.csv?
