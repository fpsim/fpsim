###############################################################################
# Calculate Birth Spacing Intervals for Calibration
# Using Most Recent DHS IR Recode Data
# January 2023
###############################################################################

# Clear environment
rm(list = ls())

# -------------------------------
# 1. User Configuration
# -------------------------------
country <- "Kenya"                         # Used for labeling and output
dta_path <- "DHS/KEIR8CFL.DTA"             # Path to DHS IR .DTA file

# -------------------------------
# 2. Setup
# -------------------------------
# Install and load required packages
required_packages <- c("tidyverse", "haven", "survey", "withr")
installed_packages <- rownames(installed.packages())

for (pkg in required_packages) {
  if (!pkg %in% installed_packages) {
    install.packages(pkg)
  }
  library(pkg, character.only = TRUE)
}

# -------------------------------
# 3. Load and Prepare DHS Data
# -------------------------------

dhs_data <- read_dta(
  dta_path,
  col_select = c(
    "v005", "v021", "v023", "v102", "caseid",
    starts_with("b0"),  # multiple births
    starts_with("b11")  # preceding birth intervals (in months)
  )
) %>%
  gather(var, val, -v005, -v021, -v023, -v102, -caseid) %>%
  separate(var, into = c("var", "num"), sep = "_") %>%
  spread(var, val) %>%
  filter(b0 < 2) %>%  # remove multiples
  mutate(
    space_mo = b11,                         # birth interval in months
    wt = v005 / 1e6,                        # DHS sample weight
    urban_rural = factor(v102, levels = c(1, 2), labels = c("urban", "rural"))
  )

# -------------------------------
# 4. Calculate Weighted Birth Spacing Table
# -------------------------------
design <- svydesign(
  id = ~v021,
  strata = ~v023,
  weights = ~wt,
  data = dhs_data
)

spacing <- svytable(~space_mo + urban_rural, design) %>%
  as.data.frame()

# -------------------------------
# 5. Save Output to Country Directory
# -------------------------------
output_dir <- file.path(".", country)
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

write.csv(spacing, file.path(output_dir, "birth_spacing_dhs.csv"), row.names = FALSE)
