### Data to process DHS data for breastfeeding duration

library(tidyverse)
library(survey)
library(fitdistrplus)
library(truncdist)

# Get DHS data
dir <- "directory" ## directory with data
filename <- "filename" ## must be a IR (individual recode) file from the DHS for automatic processing

setwd(dir)
dat <- readstata13::read.dta13(filename)

# Data cleaning
dat_nonmiss <- dat %>%
  # select categories of breastfeeding, months breastfed, age of baby in months, and survey variables
  select(starts_with("m4_"), starts_with("b19_"), v005, v021, v023, caseid) %>% 
  # make long format with a row for each birth
  gather(var, val, -v005, -v021, -v023, -caseid) %>% separate(var, into = c("var", "birth")) %>% mutate(birth = as.numeric(birth)) %>% spread(var, val) %>% 
  mutate(age_months = as.numeric(b19), # ensure age is numeric
         wt = v005/1e6, # calucalte survey weight
         still_bf = ifelse(m4 == "still breastfeeding", 1, 0), # create indicator for still breastfeeding
         age_grp = floor(age_months / 2) * 2) %>% # 2 month age bins (it's too noisy with individual age bins)
  filter(age_months < 36 & !is.na(m4))  # include babies less than 3 years old and non-missing data


# survey weighted table of proportion still breastfeeding of that age group  
summary_table <- svyby(~still_bf, ~age_grp, 
                       svydesign(id=~v021, weights=~wt, strata=~v023, data = dat_nonmiss), 
                       FUN = svymean) %>%
  arrange(age_grp) %>% mutate(p_stop = lag(still_bf) - still_bf,  # estimate of stopping probability, basically the PDF
                              counts = ifelse(p_stop<0, 0, round(p_stop*10000))) %>% # create pseudo-counts to simulate a sample size, change and negatives to 0
  filter(!is.na(p_stop)) # remove month 0 


# Create a vector of repeated ages
ages_rep <- rep(summary_table$age_grp, summary_table$counts)

# Fit truncated normal distribution
fit_tnorm <- fitdist(ages_rep, "norm", start = list(mean = mean(ages_rep), sd = sd(ages_rep)))
summary(fit_tnorm)

# Save estimates in a csv in relevant country folder
estimates <- as.data.frame(fit_tnorm$estimate)
write.csv(estimates, file = "bf_stats.csv")


