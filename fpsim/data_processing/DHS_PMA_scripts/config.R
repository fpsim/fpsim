# -------------------------------
# User Configuration
# -------------------------------

# Output directory for processed data
# fpsim/fpsim/locations/nigeria_kano/data
output_dir <- '../../locations/'

#### Country label used for subdirectory name within output_dir defined above
# For example, the current configuration will store processed data in ./Model_Data/Kenya
country <- "nigeria"
#### Region Label: For countries with region specific data, specify region and region integer. Stored in DHS variable v024.
# region_code: region integer from value_code v024; <int>
#            Examples: Pakistan Sindh (v024 2018) = 2; 
#            Nigeria Kano & Kaduna (sstate 2018) = 100 & 110, respectively
#            Nigeria Kano & Kaduna (v024 1990) = 12 & 11, respectively
#region_variable <- "sstate"
#region <- "kaduna"
#region_code <- 110
#country_region <- "nigeria_kaduna"

# Base directories
# DHS location: Update path below relative to config file location
# config file location: fpsim/fpsim/data_processing/DHS_PMA_scripts/config_files
DHS_Data <- "../../../../data/DHS_katedownload/"  # Folder (relative to data_processing/ dir) where DHS DTA files are stored
PMA_Data <- file.path("../../../data/PMA", country)# Folder (relative to data_processing/ dir) where PMA DTA files are stored

# Specific data file paths (relative to above folders or absolute)
dhs_file <- "Nigeria_Female_2018.DTA"  # Female DHS file; needs to be Individual Recode (IR) DTA file

pma_file1 <- "PMA2020_NGP1_Kano_HQFQ_v3.0_1Sep2024.dta"  # Example PMA file, 2020 Phase 1 (Use the PMA 'Household and Female' datasets DTA files)
pma_file2 <- "PMA2021_NGP2_Kano_HQFQ_v2.0_1Sep2024.dta"  # Example PMA file, 2022 Phase 2 (Use the PMA 'Household and Female' datasets DTA files)
pma_file3 <- "PMA2022_NGP3_Kano_HQFQ_v3.0_1Sep2024.dta" # Example PMA file, 2022 Phase 3 (Use the PMA 'Household and Female' datasets DTA files)
pma_file4 <- "PMA2024_NGP4_Kano_HQFQ_v1.0_30Aug2024.dta" # Example PMA file, 2024 Phase 4 (Use the PMA 'Household and Female' datasets DTA files)

# Contraceptive module for which to generate contraceptive coefficients in contra_coeffs.R
model_type <- "both"  # Options: "simple", "standard", "both"

# ------------------------------

# Full paths constructed here for use in other scripts
dhs_path <- file.path(DHS_Data, dhs_file)
pma1_path <- file.path(PMA_Data, pma_file1)
pma2_path <- file.path(PMA_Data, pma_file2)
pma3_path <- file.path(PMA_Data, pma_file3)
pma4_path <- file.path(PMA_Data, pma_file4)
