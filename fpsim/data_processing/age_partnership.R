###############################################################################
# Calculate the proportion of women entering partnership by age
# Using DHS individual recode (IR) data
#
# Creates: age_partnership.csv
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
# 2. Load and Clean Data
# -------------------------------
data.raw <- read_dta(dhs_path)

data <- data.raw %>%
  mutate(age_partner = v511)  # v511: age at first cohabitation

# -------------------------------
# 3. Create Survey Design Object
# -------------------------------
svydesign_obj = svydesign(id = data$v001, strata=data$v023, weights = data$v005/1000000, data=data)  # Ensure means and SEM are properly adjusted for the survey design

# -------------------------------
# 4. Calculate Proportion by Age at Partnership
# -------------------------------
table_partnership <- as.data.frame(svytable(~age_partner, svydesign_obj)) %>%
  mutate(percent = Freq/sum(Freq)) %>% dplyr::select(-Freq)

# -------------------------------
# 5. Save Output to Country Directory
# -------------------------------
output_dir <- file.path(output_dir, country)
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

write.csv(table_partnership, file.path(output_dir, "age_partnership.csv"), row.names = FALSE)
