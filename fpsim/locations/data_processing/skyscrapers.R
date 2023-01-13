#########################################
# -- Skyscraper data
# -- Marita Zimmermann
# -- December 2022
#########################################

rm(list=ls())

# -- Libraries -- #
library(tidyverse)
library(withr)
library(haven)
library(scales)

# -- Download data -- #
pma.raw <- with_dir(normalizePath(file.path(Sys.getenv("ONEDRIVE"), "WRICH/Data/PMA/Kenya"), "/"), {read_dta("PMA2022_KEP3_HQFQ_v2.0_17Aug2022/PMA2022_KEP3_HQFQ_v2.0_17Aug2022.DTA")})
dhs.raw <- with_dir(normalizePath(file.path(Sys.getenv("ONEDRIVE"), "DHS"), "/"), {read_dta("KE_2014_DHS_09162022_2324_122388/KEIR72DT/KEIR72FL.DTA")})

data <- pma.raw  %>% filter(gender == 2 & !is.na(FQ_age) & !is.na(FQweight)) %>% select(birth_events, age = FQ_age, wt = FQweight) %>%
  mutate(parity = case_when(birth_events>6 ~ 6, is.na(birth_events) ~ 0, T ~ birth_events), dataset = "PMA 2022") %>% select(-birth_events) %>%
  bind_rows(dhs.raw %>% mutate(wt = v005/1000000, age = as.numeric(v012), parity=as.numeric(v220), dataset = "DHS 2014") %>% select(wt, age, parity, dataset))

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

write.table(data.result, file="Results/skyscraper_kenya_2022-12-13.csv", sep=",", row.names = F)
