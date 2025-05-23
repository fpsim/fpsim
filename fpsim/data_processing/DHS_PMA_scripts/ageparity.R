###############################################################################
# Age-Parity Table for Calibration
# Using DHS individual recode (IR) data
#
# Creates: ageparity.csv
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
# 2. Load and Prepare DHS Data
# -------------------------------

# Read relevant variables:
# - v005: sample weight
# - v012: age
# - v021: cluster ID (PSU)
# - v023: strata
# - v220: parity (number of live births)

dhs_data <- read_dta(dhs_path,
                     col_select = c("v005", "v012", "v021", "v023", "v220")) %>%
  mutate(
    wt = v005 / 1e6,
    age = as.numeric(v012),
    parity = as.numeric(v220),
    age = case_when(
      age > 14 & age <= 19 ~ "15-19",
      age > 19 & age <= 24 ~ "20-24",
      age > 24 & age <= 29 ~ "25-29",
      age > 29 & age <= 34 ~ "30-34",
      age > 34 & age <= 39 ~ "35-39",
      age > 39 & age <= 44 ~ "40-44",
      age > 44 & age <= 49 ~ "45-49",
      TRUE ~ NA_character_
    )
  )

# -------------------------------
# 3. Calculate Age-Parity Table
# -------------------------------

# Set up DHS survey design
design <- svydesign(
  id = ~v021,
  strata = ~v023,
  weights = ~wt,
  data = dhs_data
)

# Generate weighted age-parity cross-tabulation
age_parity_table <- svytable(~age + parity, design) %>%
  as.data.frame() %>%
  mutate(percentage = Freq / sum(Freq) * 100)

# -------------------------------
# 4. Save Output
# -------------------------------

# Create country-based output directory if it doesn't exist
output_dir <- file.path(output_dir, country)
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Save CSV to <output_dir>/<country>/ageparity.csv
write.table(age_parity_table,
            file = file.path(output_dir, "ageparity.csv"),
            sep = ",",
            row.names = FALSE)

