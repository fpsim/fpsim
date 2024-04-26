#########################################
# -- Empowerment data from DHS       - ##
# -- DHS data
# -- Marita Zimmermann
# -- August 2023
#########################################

rm(list=ls())

# -- Libraries -- #
library(tidyverse)      
library(haven)   
library(survey)
library(withr)

# -- Data -- #

# Kenya 2022 individual recode
home_dir <- path.expand("~")
dhs_dir <- "DHS"
survey_dir <- "KEIR8BDT"
filename <- "KEIR8BFL.DTA"
filepath <- file.path(home_dir, dhs_dir, survey_dir, filename)  # or replace with your own path to the DTA file

# -- Load all the data
data.raw <- read_dta(filepath)

# -- Extract empowerment-related variables -- #
data <- data.raw %>%
  mutate(age = v012,
         parity = v220,
         paidwork = case_when(v731 %in% c(1,2,3) & v741 %in% c(1,2,3) ~ 1, # done work in the past year and paid in cash or in kind
                              v731 == 0 | v741 == 0 ~ 0),
         decision.wages = case_when(v739 %in% c(1,2,3) ~ 1, v739 %in% c(4,5) ~ 0), # 1 she decides with or without someone else, 0 someone else decides
         decisionwages  = case_when(paidwork == 1 & decision.wages == 1 ~ 1, paidwork == 0 | decision.wages == 0 ~ 0), # create a combined variable for paid work and decision making autonomy
         refusesex = case_when(v850a == 1 ~ 1, v850a == 8 ~ 0.5, v850a == 0 ~ 0), # 1 she can refuse sex, 0.5 don't know/it depends, 0 no
         decisionpurchase = case_when(v743b %in% c(1,2,3) ~ 1, v739 %in% c(4,5) ~ 0), # large purchases, 1 she decides with or without someone else, 0 someone else decides
         decisionhealth = case_when(v743a %in% c(1,2,3) ~ 1, v739 %in% c(4,5) ~ 0), # health, 1 she decides with or without someone else, 0 someone else decides
         decisionfamily = case_when(v743d %in% c(1,2,3) ~ 1, v739 %in% c(4,5) ~ 0)) # visiting family, 1 she decides with or without someone else, 0 someone else decides

svydesign_obj = svydesign(id = data$v001,
                          strata = data$v023,
                          weights = data$v005/1000000,
                          data=data)


# -- Empowerment metrics -- #

# Table of the three empowerment outcomes by age
table.emp <- as.data.frame(svyby(~paidwork, ~age, svydes1, svymean)) %>% rename(paidwork.se = se) %>%
  left_join(as.data.frame(svyby(~decisionwages, ~age, svydes1, svymean, na.rm = T)) %>% rename(decisionwages.se = se)) %>%
  left_join(as.data.frame(svyby(~refusesex, ~age, svydes1, svymean, na.rm = T)) %>% rename(refusesex.se = se)) %>%
  left_join(as.data.frame(svyby(~decisionhealth, ~age, svydes1, svymean, na.rm = T)) %>% rename(decisionhealth.se = se))


# TODO: Add low and upper age bounds: 0 and 100 years old to table, fpsim uses this information to interpolate the
# data across all possible ages.


# -- Write table with empowerment data  -- #
fpsim_dir <- "fpsim"  # path to root directory of fpsim
locations_dir <- "fpsim/locations"
country_dir <- "kenya"
country_path <-
  file.path(home_dir, fpsim_dir, locations_dir, country_dir)

write.csv(stop.school, file.path(country_path, 'empowerment_.csv'), row.names = F)

