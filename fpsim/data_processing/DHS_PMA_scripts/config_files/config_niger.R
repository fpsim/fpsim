# -------------------------------
# User Configuration
# -------------------------------

# Output directory for processed data
# fpsim/fpsim/locations/niger/data
output_dir <- '../../locations/'

# Country label used for subdirectory name within output_dir defined above
# For example, the current configuration will store processed data in ./Model_Data/Kenya
country <- "niger"

# Base directories
# DHS location: Update path below relative to config file location
# config file location: fpsim/fpsim/data_processing/DHS_PMA_scripts/config_files
DHS_Data <- "../../../../data/DHS_katedownload/"  # Folder (relative to data_processing/ dir) where DHS DTA files are stored
PMA_Data <- file.path("../../../../data/PMA", country)# Folder (relative to data_processing/ dir) where PMA DTA files are stored

# Specific data file paths (relative to above folders or absolute)
# For all but breastfeeding because missing b19 columns = "Niger_Female_2012.DTA"  # Female DHS file; needs to be Individual Recode (IR) DTA file
dhs_file <- "Niger_Female_2012.DTA"
dhs_household_file <- "Niger_Household_2012.DTA"
  
pma_file1 <- "PMA2021_NEP1_HQFQ_v2.0_1Sep2024.dta"  # Example PMA file, 2020 Phase 1 (Use the PMA 'Household and Female' datasets DTA files)
pma_file2 <- "PMA2022_NEP2_HQFQ_v2.0_1Sep2024.dta"  # Example PMA file, 2022 Phase 2 (Use the PMA 'Household and Female' datasets DTA files)
pma_file3 <- "PMA2023_NEP3_HQFQ_1Sep2023.dta" # Example PMA file, 2022 Phase 3 (Use the PMA 'Household and Female' datasets DTA files)
pma_file4 <- "PMA2024_NEP4_HQFQ_v1.0_1Sep2024.dta" # Example PMA file, 2024 Phase 4 (Use the PMA 'Household and Female' datasets DTA files)

# Contraceptive module for which to generate contraceptive coefficients in contra_coeffs.R
model_type <- "both"  # Options: "simple", "standard", "both"

# ------------------------------

# Full paths constructed here for use in other scripts
dhs_path <- file.path(DHS_Data, dhs_file)
dhs_household_path <- file.path(DHS_Data, dhs_household_file)
pma1_path <- file.path(PMA_Data, pma_file1)
pma2_path <- file.path(PMA_Data, pma_file2)
pma3_path <- file.path(PMA_Data, pma_file3)
pma4_path <- file.path(PMA_Data, pma_file4)
