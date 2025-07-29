# -------------------------------
# User Configuration
# -------------------------------

# Output directory for processed data
# fpsim/fpsim/locations/niger/data
output_dir <- '../../locations/'

# Country label used for subdirectory name within output_dir defined above
# For example, the current configuration will store processed data in ./Model_Data/Kenya
country <- "pakistan"
# region_code: region integer from value_code v024; <int>
#            Examples: Pakistan Sindh (v024 2018) = 2; 
#            Nigeria Kano & Kaduna (sstate 2018) = 100 & 110, respectively
#            Nigeria Kano & Kaduna (v024 1990) = 12 & 11, respectively
region_variable <- "v024"
region <- "sindh"
region_code <- 2
country_region <- "pakistan_sindh"

# Base directories
# DHS location: Update path below relative to config file location
# config file location: fpsim/fpsim/data_processing/DHS_PMA_scripts/config_files
DHS_Data <- "../../../../data/DHS_katedownload/"  # Folder (relative to data_processing/ dir) where DHS DTA files are stored
PMA_Data <- file.path("../../../../data/PMA", 'india')# Folder (relative to data_processing/ dir) where PMA DTA files are stored

# Specific data file paths (relative to above folders or absolute)
dhs_file <- "Pakistan_Female_2018.DTA"  # female dhs file; needs to be individual recode (ir) dta file
dhs_household_file <- "Pakistan_Household_2018.DTA"  # female dhs file; needs to be individual recode (ir) dta file

pma_file1 <- "PMA2021_INP2_Rajasthan_HQFQ_v2.0_1Sep2024.dta"  # India PMA 
pma_file2 <- "PMA2022_INP3_Rajasthan_HQFQ_v2.0_1Sep2024.dta" # India PMA

# Contraceptive module for which to generate contraceptive coefficients in contra_coeffs.R
model_type <- "both"  # Options: "simple", "standard", "both"

# ------------------------------

# Full paths constructed here for use in other scripts
dhs_path <- file.path(DHS_Data, dhs_file)
dhs_household_path <- file.path(DHS_Data, dhs_household_file)
pma1_path <- file.path(PMA_Data, pma_file1)
pma2_path <- file.path(PMA_Data, pma_file2)
