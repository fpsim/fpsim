###############################################################################
# Calculate Distribution of Age at Sexual Debut (v531) from DHS Data
# Using DHS individual recode (IR) data
#
# -----------------------------------------------------------------------------
# Creates: debut_age.csv (age-wise probabilities)
###############################################################################

# -------------------------------
# 1. Setup
# -------------------------------

rm(list = ls())

# Load user configuration
source("./config.R")

required_packages <- c("haven", "dplyr", "readr")
installed <- rownames(installed.packages())
for (pkg in required_packages) {
  if (!pkg %in% installed) install.packages(pkg)
  library(pkg, character.only = TRUE)
}

# -------------------------------
# 2. Load and Filter DHS Data
# -------------------------------
dhs <- read_dta(dhs_path) %>%
  mutate(wgt = v005 / 1e6) %>%
  filter(v531 >= 10 & v531 < 50)

# -------------------------------
# 3. Compute Weighted Probabilities
# -------------------------------
age_probs <- dhs %>%
  group_by(age = v531) %>%
  summarise(weight_sum = sum(wgt, na.rm = TRUE), .groups = "drop") %>%
  mutate(probs = weight_sum / sum(weight_sum, na.rm = TRUE))

# Sanity check: Probabilities should sum to 1
prob_sum <- sum(age_probs$probs)
if (!isTRUE(all.equal(prob_sum, 1, tolerance = 1e-6))) {
  stop(paste("Probabilities sum to", prob_sum, "instead of 1"))
}

# -------------------------------
# 4. Save Output
# -------------------------------
output_dir <- file.path(output_dir, country)
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

write_csv(age_probs %>% dplyr::select(age, probs), file.path(output_dir, "debut_age.csv"))
