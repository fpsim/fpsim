#########################################
# -- Skyscraper data
# -- Ethiopia
# -- Marita Zimmermann
# -- January 2023
#########################################

rm(list=ls())

# -- Libraries -- #
library(tidyverse)
library(withr)
library(haven)
library(scales)

# -- Download data -- #
# Most recent PMA round. This is the stata data file version of the 2019 PMA from Ethiopia, I'm waiting or access to more recent.
pma.raw <- with_dir(normalizePath(file.path(Sys.getenv("ONEDRIVE"), "WRICH/Data/PMA/Ethiopia/PMAET_HQFQ_2019_CrossSection_v2.0_30Apr2021/PMAET_HQFQ_2019_CrossSection_v2.0_30Apr2021"), "/"), {read_dta("PMAET_HQFQ_2019_CrossSection_v2.0_30Apr2021.DTA")})
# Most recent DHS. Thi sthe stata version of the individual recode file.
dhs.raw <- with_dir(normalizePath(file.path(Sys.getenv("ONEDRIVE"), "DHS/IR_all/ETIR81DT"), "/"), {read_dta("ETIR81FL.DTA")})

data <- pma.raw  %>% filter(gender == 2 & !is.na(FQ_age) & !is.na(FQweight)) %>% select(birth_events_rw, age = FQ_age, wt = FQweight) %>%
  mutate(parity = case_when(birth_events_rw>6 ~ 6, is.na(birth_events_rw) | birth_events_rw == -99 ~ 0, T ~ birth_events_rw), dataset = "PMA 2019") %>% select(-birth_events_rw) %>%
  bind_rows(dhs.raw %>% mutate(wt = v005/1000000, age = as.numeric(v012), parity=as.numeric(v220), dataset = paste("DHS", v007)) %>% select(wt, age, parity, dataset))

data.result = data %>%
  mutate(age = case_when(age > 14 & age <= 19 ~ "15-19",
                         age > 19 & age <= 24 ~ "20-24",
                         age > 24 & age <= 29 ~ "25-29",
                         age > 29 & age <= 34 ~ "30-34",
                         age > 34 & age <= 39 ~ "35-39",
                         age > 39 & age <= 44 ~ "40-44",
                         age > 44 & age <= 49 ~ "45-49")) %>%
  group_by(dataset) %>% mutate(n = sum(wt, na.rm = T)) %>% ungroup %>%
  group_by(age, parity, dataset, n) %>% summarize(Freq = sum(wt, na.rm=T)) %>% mutate(percentage = Freq/n*100) %>% select(-n)

write.table(data.result, file="skyscraper_ethiopia_2021-01-27.csv", sep=",", row.names = F)
