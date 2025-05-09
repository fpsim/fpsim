# -------------------------------
# User Configuration
# -------------------------------

# Output directory for processed data
output_dir <- 'Model_Data'

# Country label used for subdirectory name within output_dir defined above
# For example, the current configuration will store processed data in ./Model_Data/Kenya
country <- "Kenya"

# Base directories
DHS_Data <- "DHS"  # Folder (relative to data_processing/ dir) where DHS DTA files are stored
PMA_Data <- "PMA"  # Folder (relative to data_processing/ dir) where PMA DTA files are stored

# Specific data file paths (relative to above folders or absolute)
dhs_file <- "KEIR8CFL.DTA"  # Example DHS file; needs to be Individual Recode (IR) DTA file

pma_file1 <- "PMA2019_KEP1_HQFQ_v4.0_1Sep2024.dta"  # Example PMA file, Phase 1 (Use the PMA 'Household and Female' datasets DTA files)
pma_file2 <- "PMA2020_KEP2_HQFQ_v4.0_1Sep2024.dta"  # Example PMA file, Phase 2 (Use the PMA 'Household and Female' datasets DTA files)
pma_file3 <- "PMA2022_KEP3_HQFQ_v4.0_12Jul2023.dta" # Example PMA file, Phase 3 (Use the PMA 'Household and Female' datasets DTA files)

# Contraceptive module for which to generate contraceptive coefficients in contra_coeffs.R
model_type <- "standard"  # Options: "simple", "standard", "both"

# ------------------------------

# Full paths constructed here for use in other scripts
dhs_path <- file.path(DHS_Data, dhs_file)
pma1_path <- file.path(PMA_Data, pma_file1)
pma2_path <- file.path(PMA_Data, pma_file2)
pma3_path <- file.path(PMA_Data, pma_file3)
