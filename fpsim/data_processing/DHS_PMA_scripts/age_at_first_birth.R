###############################################################################
# Calculate age at first birth for calibration
# Using DHS individual recode (IR) data
#
# Creates: afb.table.csv
# ---------------------------------------------------------------------------
# January 2023
###############################################################################

# -------------------------------
# 1. Setup
# -------------------------------

# Clear environment
rm(list = ls())

# Load user configuration
source("./config.R")

# Install and load required packages
required_packages <- c("tidyverse", "withr", "haven", "survey")
installed_packages <- rownames(installed.packages())

for (pkg in required_packages) {
  if (!pkg %in% installed_packages) {
    install.packages(pkg)
  }
  library(pkg, character.only = TRUE)
}

# -------------------------------
# 2. Load DHS Data
# -------------------------------
# Read relevant variables:
# - v005: sample weight
# - caseid: individual ID
# - v212: age at first birth
# - v201: number of children ever born
# - v012: current age

dhs_data <- read_dta(dhs_path,
                     col_select = c("v005", "caseid", "v212", "v201", "v012")) %>%
  mutate(
    wt = v005 / 1e6,
    afb = ifelse(v201 == 0, Inf, v212)  # Assign Inf for those with no children
  )

# -------------------------------
# 3. Prepare and Save Output
# -------------------------------

# Select key variables
afb_table <- dhs_data %>%
  dplyr::select(wt, age = v012, afb)

# Create country-based output directory if it doesn't exist
output_dir <- file.path(output_dir, country)
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Save CSV to ./<country>/afb.table.csv
write.csv(afb_table, file.path(output_dir, "afb.table.csv"), row.names = FALSE)
