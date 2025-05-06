###############################################################################
# Calculate the proportion of women living in urban areas
# Using DHS individual recode (IR) data
# User-configurable: Country and File Path
###############################################################################

# Clear environment
rm(list = ls())

# -------------------------------
# 1. User Configuration
# -------------------------------
country <- "Kenya"                 # Modify for labeling and output
dta_path <- "DHS/KEIR8CFL.DTA"        # Modify to path of your IR .DTA file

# -------------------------------
# 2. Setup
# -------------------------------
# Install and load required packages
required_packages <- c("tidyverse", "haven", "survey")
installed_packages <- rownames(installed.packages())

for (pkg in required_packages) {
  if (!pkg %in% installed_packages) {
    install.packages(pkg)
  }
  library(pkg, character.only = TRUE)
}

# -------------------------------
# 3. Load and Clean Data
# -------------------------------
data <- read_dta(dta_path) %>%
  mutate(urban = ifelse(v025 == 1, 1, 0))  # v025: 1 = urban, 2 = rural

# -------------------------------
# 4. Create Survey Design Object
# -------------------------------
svydesign_obj = svydesign(id = data$v001, strata=data$v023, weights = data$v005/1000000, data=data)

# -------------------------------
# 5. Calculate Proportion Urban
# -------------------------------
urban_table <- as.data.frame(svymean(~urban, svydesign_obj)) %>% rename(urban.se = urban)  # Rename standard error column for clarity

# -------------------------------
# 6. Save Output to Country Directory
# -------------------------------
output_dir <- file.path(".", country)
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

write.csv(urban_table, file.path(output_dir, "urban.csv"), row.names = FALSE)
