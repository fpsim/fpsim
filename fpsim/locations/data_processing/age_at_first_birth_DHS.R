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
afb.table <- dhs.data %>% select(wt, age = v012, afb)
write.csv(afb.table, "locations/kenya/afb.table.csv", row.names = F)
