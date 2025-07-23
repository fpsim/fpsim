# Master script to control which data processing scripts to run
if (!require("survey")) {
  options(repos = c(CRAN = "https://cloud.r-project.org/"))
  install.packages("survey")
}

# Run scripts conditionally
# Set to TRUE to run, FALSE to skip
print("run_age_at_first_birth")
run_age_at_first_birth <- TRUE
if (run_age_at_first_birth) source("age_at_first_birth.R")

print("run_age_partnership")
run_age_partnership <- TRUE
if (run_age_partnership) source("age_partnership.R")

print("run_ageparity")
run_ageparity <- TRUE
if (run_ageparity) source("ageparity.R")

print("run_birth_spacing")
run_birth_spacing <- TRUE
if (run_birth_spacing) source("birth_spacing.R")

print("run_breastfeeding_stats")
run_breastfeeding_stats <- TRUE
if (run_breastfeeding_stats) source("breastfeeding_stats.R")

print("run_contra_coeffs")
run_contra_coeffs <- TRUE
if (run_contra_coeffs) source("contra_coeffs.R")

print("run_education")
run_education <- TRUE
if (run_education) source("education_dhs_pma.R")

print("run_extract_urban")
run_extract_urban <- TRUE
if (run_extract_urban) source("extract_urban.R")

print("run_make_matrices")
run_make_matrices <- TRUE
if (run_make_matrices) source("make_matrices.R")

print("run_method_mix")
run_method_mix <- TRUE
if (run_method_mix) source("method_mix.R")

print("run_postpartum_recode")
run_postpartum_recode <- TRUE
if (run_postpartum_recode) source("postpartum_recode.R")

print("run_sexual_activity")
run_sexual_activity <- TRUE
if (run_sexual_activity) source("sexual_activity.R")

print("run_sexual_debut_age_probs")
run_sexual_debut_age_probs <- TRUE
if (run_sexual_debut_age_probs) source("sexual_debut_age_probs.R")

print("run_time_on_method")
run_time_on_method <- TRUE
if (run_time_on_method) source("time_on_method.R")

print("run_wealth")
run_wealth <- TRUE
if (run_wealth) source("wealth.R")

## Only for special cases where all subregions 
## should be run and aggregated into single file
run_subnational <- FALSE
if (run_subnational) source("subnational.R")















