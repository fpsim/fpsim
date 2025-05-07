###############################################################################
# Estimate Breastfeeding Duration from DHS Data (IR Recode Data)
# Using survey-weighted analysis and truncated normal distribution
# User-configurable: Country and File Path
###############################################################################

# Clear environment
rm(list = ls())

# -------------------------------
# 1. User Configuration
# -------------------------------
country <- "Kenya"                         # Modify for labeling and output
dta_path <- "DHS/KEIR8CFL.DTA"             # Path to DHS IR .DTA file

# -------------------------------
# 2. Setup
# -------------------------------
# Install and load required packages
required_packages <- c("tidyverse", "survey", "fitdistrplus", "truncdist", "readstata13")
installed_packages <- rownames(installed.packages())

for (pkg in required_packages) {
  if (!pkg %in% installed_packages) {
    install.packages(pkg)
  }
  library(pkg, character.only = TRUE)
}

# -------------------------------
# 3. Load and Clean DHS Data
# -------------------------------

# Read DHS individual recode file (must include m4_* and b19_* for breastfeeding analysis)
dat <- read.dta13(dta_path)

# Select and process relevant variables
dat_nonmiss <- dat %>%
  # select categories of breastfeeding, months breastfed, age of baby in months, and survey variables
  dplyr::select(starts_with("m4_"), starts_with("b19_"), v005, v021, v023, caseid) %>%
  # make long format with a row for each birth
  gather(var, val, -v005, -v021, -v023, -caseid) %>% separate(var, into = c("var", "birth")) %>% mutate(birth = as.numeric(birth)) %>% spread(var, val) %>%
  mutate(age_months = as.numeric(b19), # ensure age is numeric
         wt = v005/1e6, # calucalte survey weight
         still_bf = ifelse(m4 == "still breastfeeding", 1, 0), # create indicator for still breastfeeding
         age_grp = floor(age_months / 2) * 2) %>% # 2 month age bins (it's too noisy with individual age bins)
  filter(age_months < 36 & !is.na(m4))  # include babies less than 3 years old and non-missing data

# -------------------------------
# 4. Calculate Survey-Weighted BF Proportions
# -------------------------------

# survey weighted table of proportion still breastfeeding of that age group
summary_table <- svyby(~still_bf, ~age_grp,
                       svydesign(id=~v021, weights=~wt, strata=~v023, data = dat_nonmiss),
                       FUN = svymean) %>%
  arrange(age_grp) %>% mutate(p_stop = lag(still_bf) - still_bf,  # estimate of stopping probability, basically the PDF
                              counts = ifelse(p_stop<0, 0, round(p_stop*10000))) %>% # create pseudo-counts to simulate a sample size, change and negatives to 0
  filter(!is.na(p_stop)) # remove month 0

# -------------------------------
# 5. Fit Truncated Normal Distribution
# -------------------------------

# Create a vector of repeated ages
ages_rep <- rep(summary_table$age_grp, summary_table$counts)

# Fit truncated normal distribution
fit_tnorm <- fitdist(ages_rep, "norm", start = list(mean = mean(ages_rep), sd = sd(ages_rep)))
summary(fit_tnorm)

# -------------------------------
# 6. Save Output to Country Directory
# -------------------------------
output_dir <- file.path(".", country)
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

estimates <- data.frame(
  parameter = names(fit_tnorm$estimate),
  value = as.numeric(fit_tnorm$estimate)
)

write.csv(estimates, file.path(output_dir, "bf_stats.csv"), row.names = FALSE)
