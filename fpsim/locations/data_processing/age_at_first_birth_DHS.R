################################################################
# Calculate age at first birth for calibration
# Using most recent DHS data
# Kenya
# January 2023
################################################################

rm(list=ls())

# -- Libraries -- #
library(tidyverse)      
library(withr)  
library(haven)   
library(survey)

# -- DHS data -- #
# Source https://dhsprogram.com/data/dataset_admin/index.cfm
# recode https://www.dhsprogram.com/publications/publication-dhsg4-dhs-questionnaires-and-manuals.cfm
dhs.data <- with_dir(normalizePath(file.path(Sys.getenv("ONEDRIVE"), "DHS/IR_all"), "/"),   # read individual recode data
                     {read_dta("KEIR72DT/KEIR72FL.DTA", col_select = c("v005", "caseid", "v212", "v201", "v012"))}) %>% # weight, individual id, age at first birth, number of children, current age
  mutate(wt = v005/1000000, afb = ifelse(v201==0, Inf, v212))                                                           # calculate individual weight, calucate age at first birth by including infinity for those who have yet to have a birth

# Weighted table of age at first birth by current age
afb <- svytable(~ v012 + afb, svydesign(id=~caseid, weights=~wt, data =dhs.data))

# Calculate the median age at first birth, from the weighted table
# For women currently age 25-49 only (can't include younger women because then less than 50% have had a birth yet so there is no median)
# Linearly interpolate between who age at first birth years to get the median
afb.median <- as.data.frame(afb) %>%
  mutate_at(c("v012", "afb"), ~as.numeric(as.character(.))) %>%       # Make age and age at first birth numeric
  filter(v012>=25) %>%                                                # Only include women age 25-49
  group_by(afb) %>% summarize(Freq = sum(Freq)) %>% arrange(afb) %>%  # sum number of women in each ag at first birth category
  mutate(cumsum = cumsum(Freq),                                       # sum total women who have given birth by that age
         dif = max(cumsum/2) - cumsum,                                # find where then median is (where this value become negative)
         perc = lag(dif,1)/Freq,                                      # calculate the percentage within the median year for interpolation
         median = afb + perc) %>%                                     # calculate the interpolated median
  filter(dif >0) %>% filter(dif == min(dif)) %>% select(median)       # keep only the median result
write.csv(afb.median, "locations/kenya/afb.median.csv", row.names = F)

# Created a weighted table of all ages at first birth
afb.table <- as.data.frame(afb) %>%
  mutate_at(c("v012", "afb"), ~as.numeric(as.character(.))) %>%       # Make age and age at first birth numeric
  group_by(afb) %>% summarize(Freq = sum(Freq)) %>% arrange(afb)      # sum number of women in each ag at first birth category
write.csv(afb.table, "locations/kenya/afb.table.csv", row.names = F)


