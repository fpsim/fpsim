#############################################
# -- Analysis on PMA data
# -- For empowerment implementation in FPsim
# -- Marita Zimmermann
# -- October 2023

# Use this to create coefficients for the not empowerment version of the module
# Using PMA data
# This script needs cleaning
#############################################

rm(list=ls())

library(lavaan)
library(tidySEM)
library(ggpubr)
library(extrafont)
library(ggalluvial)
library(survey)
library(lavaan.survey)
library(tidyverse)      
library(haven)  
library(withr)  
library(splines)
library(glmnet)
library(boot)

# -- To get formatted PMA dataset, run recode_pma_datasets.R in data processing folder in FPsim

# duplicate dataset to increase sample size... so we have a 1-2 time and 2-3 time
data.edit <- recoded.datasets %>% # filtered dataset from just wave 2-3 and rename to 1-2 and label time b
  filter(wave != 1) %>% mutate(wave = case_when(wave == 2 ~ 1, wave == 3 ~ 2), time = "b")
filter_data <- All_data %>%
  filter(wave != 3) %>% mutate(time = "a") %>% # filter to timepoint 102 and label a
  bind_rows(data.edit) %>% # add back in other timepoint
  # to wide 
  gather(var, val, -wave, -female_ID, -time) %>% unite("var", c(var, wave)) %>% spread(var, val, convert = T, drop = F, fill = NA) %>%
  mutate(FQweight = ifelse(is.na(FQweight_2), FQweight_1, FQweight_2),
         strata = ifelse(is.na(strata_2), strata_1, strata_2),
         EA_ID = ifelse(is.na(EA_ID_2), EA_ID_1, EA_ID_2)) %>%
  filter(!is.na(EA_ID)) 
  # refuse sex only waves 1 and 3, and no variation, so taking that one out

# create dataset for postpartum and not
filter.data.notpp <- filter_data %>% filter((pp.time_2>6 | is.na(pp.time_2)) & pregnant_2 == 0)
filter.data.pp1 <- filter_data %>% filter(pp.time_2<2)
filter.data.pp6 <- filter_data %>% filter(pp.time_2<6)

# create survey objects
options(survey.lonely.psu="adjust")
svydes.full <- svydesign(id = ~EA_ID, strata = ~strata, weights =  ~FQweight, data = filter_data , nest = T)
svydes <- svydesign(id = ~EA_ID, strata = ~strata, weights =  ~FQweight, data = filter.data.notpp , nest = T)
svydes.pp1 <- svydesign(id = ~EA_ID, strata = ~strata, weights =  ~FQweight, data = filter.data.pp1 , nest = T)
svydes.pp6 <- svydesign(id = ~EA_ID, strata = ~strata, weights =  ~FQweight, data = filter.data.pp6 , nest = T)


      



# contraception function

# Contraception mid function (only demographics, no empowerment or history)... no longitudinal or empowerment data needed, could be done with DHS
model.mid <- svyglm(current_contra_2 ~ ns(age_2, knots = c(25,40))*fp_ever_user_2 + yrs.edu_2 + live_births_2 + urban_2 + wealthquintile_2, 
                     family = quasibinomial(), 
                     design = svydes)
                     #design = svydes.pp1)
                     #design = svydes.pp6)
contra_coef.mid <- as.data.frame(summary(model.mid)$coefficients) %>% 
  mutate(rhs = rownames(.)) %>%
  mutate(rhs = gsub("_2", "", gsub("live_births", "parity",                                                   
                                   gsub("yrs.edu","edu_attainment",
                                        gsub("current_contra", "contraception", rhs)))))

# write.csv(contra_coef.mid, "fpsim/locations/kenya/contra_coef_mid.csv", row.names = F)
# write.csv(contra_coef.mid, "fpsim/locations/kenya/contra_coef_mid_pp1.csv", row.names = F)
# write.csv(contra_coef.mid, "fpsim/locations/kenya/contra_coef_mid_pp6.csv", row.names = F)




# create csv for age splines
splines <- as.data.frame(ns(c(15:49), knots = c(25,40)))
names(splines) <- c("knot_1", "knot_2","knot_3")
write.csv(splines, "splines_25_40.csv", row.names = F)
write.csv(splines, "C:/Users/maritazi/Documents/Projects/fpsim/fpsim/locations/kenya/data/splines_25_40.csv", row.names = F)







