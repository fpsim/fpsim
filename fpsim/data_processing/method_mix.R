###############################################################################
# Calculate Method Mix from DHS Data
# Using DHS individual recode (IR) data
# User-configurable: Country and File Path
###############################################################################

# Clear environment
rm(list = ls())

# -------------------------------
# 1. User Configuration
# -------------------------------
country <- "Kenya"                         # Modify for labeling and output
dta_path <- "DHS/KEIR8CFL.DTA"             # Modify to path of your IR .DTA file

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
  mutate(
    method = case_when(
      v312 == 1 ~ "Pill",
      v312 == 2 ~ "IUDs",
      v312 == 3 ~ "Injectables",
      v312 %in% c(4, 13, 15, 16, 17, 18) ~ "Other modern",
      v312 %in% c(5, 14) ~ "Condoms",
      v312 %in% c(6, 7) ~ "BTL",
      v312 %in% c(8, 10, 12) ~ "Other traditional",
      v312 == 9 ~ "Withdrawal",
      v312 == 11 ~ "Implants"
    ),
    wt = v005 / 1e6
  ) %>%
  filter(!is.na(method))

# -------------------------------
# 4. Create Survey Design Object
# -------------------------------
svydesign_obj <- svydesign(id = ~v021, strata = ~v023, weights = ~wt, data = data)

# -------------------------------
# 5. Calculate Method Mix
# -------------------------------
method_mix <- svytable(~method, svydesign_obj) %>%
  as.data.frame() %>%
  mutate(perc = Freq / sum(Freq) * 100)

# -------------------------------
# 6. Save Output to Country Directory
# -------------------------------
output_dir <- file.path(".", country)
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

write.csv(method_mix, file.path(output_dir, "mix.csv"), row.names = FALSE)
