#########################################
# -- Skyscraper data
# -- Marita Zimmermann
# -- January 2023
#########################################

rm(list=ls())

# -- Libraries -- #
library(tidyverse)
library(withr)
library(haven)
library(survey)

# -- Download data -- #
# Most recent DHS. This the stata version of the individual recode file.
dhs.raw <- with_dir(normalizePath(file.path(Sys.getenv("ONEDRIVE"), "DHS/KEIR8ADT"), "/"), {read_dta("KEIR8AFL.DTA")})

data <- dhs.raw %>% mutate(wt = v005/1000000, age = as.numeric(v012), parity=as.numeric(v220)) %>% 
  mutate(age = case_when(age > 14 & age <= 19 ~ "15-19",
                         age > 19 & age <= 24 ~ "20-24",
                         age > 24 & age <= 29 ~ "25-29",
                         age > 29 & age <= 34 ~ "30-34",
                         age > 34 & age <= 39 ~ "35-39",
                         age > 39 & age <= 44 ~ "40-44",
                         age > 44 & age <= 49 ~ "45-49"))

data.result <- as.data.frame(svytable(~ age + parity, svydesign(id=~v021, weights=~wt, strata=~v023, data = data))) %>% mutate(percentage = Freq/sum(Freq)*100)

write.table(data.result, file="ageparity.csv", sep=",", row.names = F)
