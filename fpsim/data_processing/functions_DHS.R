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
#Kenya
filepath <- file.path("C:/Users/maritazi/OneDrive - Bill & Melinda Gates Foundation/DHS/IR_all/KEIR72DT/KEIR72FL.DTA")

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
         edu.level = factor(case_when(v106 == 0 ~"None", v106 == 1 ~"Primary", v106 %in% c(2,3) ~"Secondary"), 
                               levels = c("None", "Primary", "Secondary")), # cuts are generally around 5/6 and 12/13
         wealthquintile = v190,
         current_contra = ifelse(v312 == 0, 0, 1),
         # find if women used a method prior to current
         cal = str_squish(substr(vcal_1, 1, 36)), # contraceptive calendar (3 years only)
         first_char = substr(cal, 1, 1), # current method
         chars = strsplit(cal, ""),
         repeat_end = map2_int(chars, first_char, ~ {match(TRUE, .x != .y, nomatch = length(.x) + 1)}), # how many months back does current method end
         remainder = ifelse(repeat_end > str_length(cal), "", map2_chr(chars, repeat_end, ~ paste0(.x[.y:length(.x)], collapse = ""))), # string of remaining methods prior to current
         prior_user = ifelse(vcal_1 == "", NA, 
                             ifelse(repeat_end > str_length(cal) & first_char != "0", T, # true if she has been using the same method for 5 years
                             map_lgl(remainder, ~ str_detect(.x, "[^0LPTB]"))))) # does remainder include any method aside from none, LAM, pregnancy, termination, birth


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
simple.function <- function(svychoice){
  model.simple <- svyglm(current_contra ~ age_grp * prior_user, 
                         family = quasibinomial(), 
                         design = svychoice)
  contra_coef.simple <- as.data.frame(summary(model.simple)$coefficients) %>% 
    mutate(rhs = rownames(.))
  return(contra_coef.simple)
}
  

# write.csv(simple.function(svydes), "C:/Users/maritazi/Documents/Projects/fpsim/fpsim/locations/senegal/data/contra_coef_simple.csv", row.names = F)
# write.csv(simple.function(svydes.pp1), "C:/Users/maritazi/Documents/Projects/fpsim/fpsim/locations/senegal/data/contra_coef_simple_pp1.csv", row.names = F)
# write.csv(simple.function(svydes.pp6), "C:/Users/maritazi/Documents/Projects/fpsim/fpsim/locations/senegal/data/contra_coef_simple_pp6.csv", row.names = F)

# write.csv(simple.function(svydes), "C:/Users/maritazi/Documents/Projects/fpsim/fpsim/locations/ethiopia/data/contra_coef_simple.csv", row.names = F)
# write.csv(simple.function(svydes.pp1), "C:/Users/maritazi/Documents/Projects/fpsim/fpsim/locations/ethiopia/data/contra_coef_simple_pp1.csv", row.names = F)
# write.csv(simple.function(svydes.pp6), "C:/Users/maritazi/Documents/Projects/fpsim/fpsim/locations/ethiopia/data/contra_coef_simple_pp6.csv", row.names = F)

# write.csv(simple.function(svydes), "C:/Users/maritazi/Documents/Projects/fpsim/fpsim/locations/kenya/data/contra_coef_simple.csv", row.names = F)
# write.csv(simple.function(svydes.pp1), "C:/Users/maritazi/Documents/Projects/fpsim/fpsim/locations/kenya/data/contra_coef_simple_pp1.csv", row.names = F)
# write.csv(simple.function(svydes.pp6), "C:/Users/maritazi/Documents/Projects/fpsim/fpsim/locations/kenya/data/contra_coef_simple_pp6.csv", row.names = F)


# Contraception mid function (only demographics, no empowerment or history)... no longitudinal or empowerment data needed, could be done with DHS
mid.function <- function(svychoice){
  model.mid <- svyglm(current_contra ~ ns(age, knots = c(25,40))*prior_user + edu.level + live_births + urban + wealthquintile, 
                      family = quasibinomial(), 
                      design = svychoice)
  contra_coef.mid <- as.data.frame(summary(model.mid)$coefficients) %>% 
    mutate(rhs = rownames(.)) %>%
    mutate(rhs = gsub("live_births", "parity",                                                   
                      gsub("yrs.edu","edu_attainment",
                           gsub("current_contra", "contraception", rhs))))
  return(contra_coef.mid)
}

# write.csv(mid.function(svydes), "C:/Users/maritazi/Documents/Projects/fpsim/fpsim/locations/senegal/data/contra_coef_mid.csv", row.names = F)
# write.csv(mid.function(svydes.pp1), "C:/Users/maritazi/Documents/Projects/fpsim/fpsim/locations/senegal/data/contra_coef_mid_pp1.csv", row.names = F)
# write.csv(mid.function(svydes), "C:/Users/maritazi/Documents/Projects/fpsim/fpsim/locations/senegal/data/contra_coef_mid_pp6.csv", row.names = F)

# write.csv(mid.function(svydes), "C:/Users/maritazi/Documents/Projects/fpsim/fpsim/locations/ethiopia/data/contra_coef_mid.csv", row.names = F)
# write.csv(mid.function(svydes.pp1), "C:/Users/maritazi/Documents/Projects/fpsim/fpsim/locations/ethiopia/data/contra_coef_mid_pp1.csv", row.names = F) # Model does not converge
# write.csv(mid.function(svydes.pp6), "C:/Users/maritazi/Documents/Projects/fpsim/fpsim/locations/ethiopia/data/contra_coef_mid_pp6.csv", row.names = F)

# write.csv(mid.function(svydes), "C:/Users/maritazi/Documents/Projects/fpsim/fpsim/locations/kenya/data/contra_coef_mid.csv", row.names = F)
# write.csv(mid.function(svydes.pp1), "C:/Users/maritazi/Documents/Projects/fpsim/fpsim/locations/kenya/data/contra_coef_mid_pp1.csv", row.names = F)
# write.csv(mid.function(svydes.pp6), "C:/Users/maritazi/Documents/Projects/fpsim/fpsim/locations/kenya/data/contra_coef_mid_pp6.csv", row.names = F)


# create csv for age splines
splines <- as.data.frame(ns(c(15:49), knots = c(25)))
names(splines) <- c("knot_1", "knot_2")
splines$age <- c(15:49)
write.csv(splines, "C:/Users/maritazi/Documents/Projects/fpsim/fpsim/locations/splines_25.csv", row.names = F)

splines <- as.data.frame(ns(c(15:49), knots = c(25,40)))
names(splines) <- c("knot_1", "knot_2","knot_3")
splines$age <- c(15:49)
write.csv(splines, "C:/Users/maritazi/Documents/Projects/fpsim/fpsim/locations/splines_25_40.csv", row.names = F)










