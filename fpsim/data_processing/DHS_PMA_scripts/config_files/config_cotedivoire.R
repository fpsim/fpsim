# -------------------------------
# User Configuration
# -------------------------------

# Output directory for processed data: Update path below relative to config file location
output_dir <- '../../../locations/'

# Country label used for subdirectory name within output_dir defined above
# For example, the current configuration will store processed data in ./Model_Data/Kenya
country <- "cotedivoire"

# Base directories
# DHS location: Update path below relative to config file location
# config file location: fpsim/fpsim/data_processing/DHS_PMA_scripts/config_files
DHS_Data <- "../../../../../data/DHS_katedownload/"  # Folder (relative to data_processing/ dir) where DHS DTA files are stored
PMA_Data <- "../../../../../data/PMA/" + country + "/"  # Folder (relative to data_processing/ dir) where PMA DTA files are stored

# Specific data file paths (relative to above folders or absolute)
dhs_file <- "CDI_Female_2021.DTA"  # Female DHS file; needs to be Individual Recode (IR) DTA file

pma_file1 <- "PMA2020_CIP1_HQFQ_v2.0_1Sep2024.dta"  # Example PMA file, 2020 Phase 1 (Use the PMA 'Household and Female' datasets DTA files)
pma_file2 <- "PMA2022_CIP2_HQFQ_v2.0_1Sep2024.dta"  # Example PMA file, 2022 Phase 2 (Use the PMA 'Household and Female' datasets DTA files)
pma_file3 <- "PMA2022_CIP3_HQFQ_v3.0_1Sep2024.dta" # Example PMA file, 2022 Phase 3 (Use the PMA 'Household and Female' datasets DTA files)
pma_file4 <- "PMA2022_CIP3_HQFQ_v3.0_1Sep2024.dta" # Example PMA file, 2024 Phase 4 (Use the PMA 'Household and Female' datasets DTA files)

# Contraceptive module for which to generate contraceptive coefficients in contra_coeffs.R
model_type <- "both"  # Options: "simple", "standard", "both"

# ------------------------------

# Full paths constructed here for use in other scripts
dhs_path <- file.path(DHS_Data, dhs_file)
pma1_path <- file.path(PMA_Data, pma_file1)
pma2_path <- file.path(PMA_Data, pma_file2)
pma3_path <- file.path(PMA_Data, pma_file3)
pma4_path <- file.path(PMA_Data, pma_file4)
