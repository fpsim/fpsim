#############################################
# -- Analysis on PMA data
# -- For empowerment implementation in FPsim
# -- Marita Zimmermann
# -- October 2023

# Use this to create coefficients for the not empowerment version of the module
# Using DHS data
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


# -- Data -- #
# Senegal
filepath <- file.path("C:/Users/maritazi/OneDrive - Bill & Melinda Gates Foundation/DHS/IR_all/SNIR8BDT/SNIR8BFL.DTA")
#Ethiopia
filepath <- file.path("C:/Users/maritazi/OneDrive - Bill & Melinda Gates Foundation/DHS/IR_all/ETIR71DT/ETIR71FL.DTA")

# -- Load all the data
data.raw <- read_dta(filepath)


# data manipulation
All_data <- data.raw %>%
  mutate(age = as.numeric(v012),
         age_grp = cut(age, c(0, 18, 20, 25, 35, 50)),
         live_births = as.numeric(v219),
         urban = ifelse(v025 == 1, 1, 0), 
         pp.time = v222, # months since last birth
         pregnant = v213,
         fp_ever_user = ifelse(v302a %in% c(1,2), 1, 0),
         yrs.edu = ifelse(v133 == 98, NA, v133),
         wealthquintile = v190,
         current_contra = ifelse(v312 == 0, 0, 1))


# create dataset for postpartum and not
filter.data.notpp <- All_data %>% filter((pp.time>6 | is.na(pp.time)) & pregnant == 0)
filter.data.pp1 <- All_data %>% filter(pp.time<2)
filter.data.pp6 <- All_data %>% filter(pp.time<6)

# create survey objects
options(survey.lonely.psu="adjust")
svydes = svydesign(id = ~v001, strata = ~v023, weights = filter.data.notpp$v005/1000000, data=filter.data.notpp, nest = T)
svydes.pp1 = svydesign(id = ~v001, strata = ~v023, weights = filter.data.pp1$v005/1000000, data=filter.data.pp1, nest = T)
svydes.pp6 = svydesign(id = ~v001, strata = ~v023, weights = filter.data.pp6$v005/1000000, data=filter.data.pp6, nest = T)




# contraception functions

# Contraception simple function
model.simple <- svyglm(current_contra ~ age_grp * fp_ever_user, 
                       family = quasibinomial(), 
                       #design = svydes)
                       #design = svydes.pp1)
                       design = svydes.pp6)
contra_coef.simple <- as.data.frame(summary(model.simple)$coefficients) %>% 
  mutate(rhs = rownames(.))

# write.csv(contra_coef.simple, "C:/Users/maritazi/Documents/Projects/fpsim/fpsim/locations/senegal/data/contra_coef_simple.csv", row.names = F)
# write.csv(contra_coef.simple, "C:/Users/maritazi/Documents/Projects/fpsim/fpsim/locations/senegal/data/contra_coef_simple_pp1.csv", row.names = F)
# write.csv(contra_coef.simple, "C:/Users/maritazi/Documents/Projects/fpsim/fpsim/locations/senegal/data/contra_coef_simple_pp6.csv", row.names = F)

# write.csv(contra_coef.simple, "C:/Users/maritazi/Documents/Projects/fpsim/fpsim/locations/ethiopia/data/contra_coef_simple.csv", row.names = F)
# write.csv(contra_coef.simple, "C:/Users/maritazi/Documents/Projects/fpsim/fpsim/locations/ethiopia/data/contra_coef_simple_pp1.csv", row.names = F)
# write.csv(contra_coef.simple, "C:/Users/maritazi/Documents/Projects/fpsim/fpsim/locations/ethiopia/data/contra_coef_simple_pp6.csv", row.names = F)


# Contraception mid function (only demographics, no empowerment or history)... no longitudinal or empowerment data needed, could be done with DHS
model.mid <- svyglm(current_contra ~ ns(age, knots = c(25,40))*fp_ever_user + yrs.edu + live_births + urban + wealthquintile, 
                     family = quasibinomial(), 
                     design = svydes)
                     #design = svydes.pp1)
                     #design = svydes.pp6)
contra_coef.mid <- as.data.frame(summary(model.mid)$coefficients) %>% 
  mutate(rhs = rownames(.)) %>%
  mutate(rhs = gsub("live_births", "parity",                                                   
                    gsub("yrs.edu","edu_attainment",
                         gsub("current_contra", "contraception", rhs))))

# write.csv(contra_coef.mid, "C:/Users/maritazi/Documents/Projects/fpsim/fpsim/locations/senegal/data/contra_coef_mid.csv", row.names = F)
# write.csv(contra_coef.mid, "C:/Users/maritazi/Documents/Projects/fpsim/fpsim/locations/senegal/data/contra_coef_mid_pp1.csv", row.names = F)
# write.csv(contra_coef.mid, "C:/Users/maritazi/Documents/Projects/fpsim/fpsim/locations/senegal/data/contra_coef_mid_pp6.csv", row.names = F)

# write.csv(contra_coef.mid, "C:/Users/maritazi/Documents/Projects/fpsim/fpsim/locations/ethiopia/data/contra_coef_mid.csv", row.names = F)
# write.csv(contra_coef.mid, "C:/Users/maritazi/Documents/Projects/fpsim/fpsim/locations/ethiopia/data/contra_coef_mid_pp1.csv", row.names = F)
# write.csv(contra_coef.mid, "C:/Users/maritazi/Documents/Projects/fpsim/fpsim/locations/ethiopia/data/contra_coef_mid_pp6.csv", row.names = F)

# create csv for age splines
splines <- as.data.frame(ns(c(15:49), knots = c(25,40)))
names(splines) <- c("knot_1", "knot_2","knot_3")
write.csv(splines, "splines_25_40.csv", row.names = F)
write.csv(splines, "C:/Users/maritazi/Documents/Projects/fpsim/fpsim/locations/senegal/data/splines_25_40.csv", row.names = F)
write.csv(splines, "C:/Users/maritazi/Documents/Projects/fpsim/fpsim/locations/ethiopia/data/splines_25_40.csv", row.names = F)










