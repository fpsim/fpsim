#########################################
# -- Create switching matrix for ABM - ##
# -- DHS data
# -- Marita Zimmermann
# -- June 2022
#########################################

rm(list=ls())

# -- Libraries -- #
library(tidyverse)      
library(haven)   
library(withr)  
library(labelled)
library(lubridate)
library(expm)
library(data.table)

# -- Download data -- #
# Source https://dhsprogram.com/data/dataset_admin/index.cfm
# recode https://www.dhsprogram.com/publications/publication-dhsg4-dhs-questionnaires-and-manuals.cfm
data.path <- normalizePath(file.path(Sys.getenv("ONEDRIVE"), "DHS"), "/")

# Senegal
data.raw <- with_dir(data.path, {read_dta("SN_2018_ContinuousDHS_10072020_1835_122388/SNIR80DT/SNIR80FL.DTA")}) %>% 
  bind_rows(with_dir(data.path, {read_dta("SN_2017_ContinuousDHS_10142020_2310_122388/SNIR7ZDT/SNIR7ZFL.DTA")})) 

# Kenya
data.raw <- with_dir(data.path, {read_dta("KE_2014_DHS_09162022_2324_122388/KEIR72DT/KEIR72FL.DTA")}) 

# Ethiopia
data.raw <- with_dir(data.path, {read_dta("ETIR71DT/ETIR71FL.DTA")}) %>% 
  bind_rows(with_dir(data.path, {read_dta("ETIR81DT/ETIR81FL.DTA")}))

# -- Data manipulation -- #
# keep only calendar data, age, and parity
data <- data.raw %>%
  mutate(wt = v005/1000000,
         age = as.numeric(v012),                                                                            # convert age to numeric
         sex.active = v527,                                                                                 # Time since the last sexual relations as reported by the respondent.  The first digit gives the units in which the respondent gave her answer:  1 - Days ago, 2 - Weeks ago, 3 - Months ago, 4 - Years ago, with 9 meaning a special answer was given.  The last two digits give the time in the units given.  
         parity1=as.numeric(v220)) %>%                                                                      # convert parity to numeric, living children categorical 0-5, 6+
  select(wt, age, parity1, vcal_1, sex.active) %>%
  mutate(cal = str_squish(vcal_1),                                                                          # remove whitespace
         cal = gsub("8","W",cal),                                                                           # recode contraceptive methods, Periodic abstinence/rhythm to other traditional
         cal = gsub("S","M",cal), cal = gsub("4", "M", cal), cal = gsub("C", "M", cal),                     # recode contraceptive methods to other modern, std days, diaphragm, female condom
         cal = gsub("E", "M", cal), cal = gsub("F", "M", cal), cal = gsub("7", "M", cal),                   # recode contraceptive methods to other modern, emergency contraception, foam and jelly, male sterilization  
         cal_clean = sapply(strsplit(cal, split = ""),  function(str) {paste(rev(str), collapse = "")}),    # reverse string so oldest is first
         pattern_split = strsplit(cal_clean,""),                                                            # split into individual characters
         month.a = lapply(pattern_split, function(m) m[1:79]),                                              # codes for months 1-79
         month.b = lapply(pattern_split, function(m) m[2:80]),                                              # codes for months 2-80
         obs = 1:nrow(.)) %>%                                                                               # create obs variable to identify the individual
  unnest(c(month.a, month.b)) %>%                                                                           # create long format
  group_by(obs) %>% mutate(month = row_number(),                                                            # create variable for month from beginning of calendar
                           age = age - (79-month)/12,                                                       # adjust for age in calendar
                           age_grp = case_when(age <= 18 ~ "<18",                                           # define age groups
                                               age > 18 & age <= 20 ~ "18-20",
                                               age > 20 & age <= 25 ~ "20-25",
                                               age > 25 & age <= 35 ~ "25-35",
                                               age > 35 ~ ">35"),
                           month.6 = lead(month.a,6),                                                       # code for what happened 6 months after current month (for use in pp calculation)
                           age_grp.6 = lead(age_grp, 5),                                                    # code for age 6 months after current month (for use in pp calculation)
                           pp.6 = str_detect(substr(cal_clean, month-6, month), "B"),                       # postpartum in past 6 months
                           pp.12 = str_detect(substr(cal_clean, month-12, month), "B"),                     # postpartum in past 12 months
                           b.month = ifelse(month.a == "B", month, NA)) %>% fill(b.month) %>% mutate(       # month of most recent birth
                           month.pp = ifelse(pp.12, month-b.month, NA),                                     # number of months pp
                           termination = str_detect(substr(cal_clean, month-6, month), "T")) %>%            # termination within past 6 months  
  filter(!is.na(month.a) & !is.na(month.b) & month.a != "P" & month.b != "P" &  month.a != "T" & month.b != "T" &  month.a != "L" & month.b != "L") %>% # remove missing values, remove pregnancy, remove lac.am, remove termination
  mutate_at(c("month.a", "month.b", "month.6"), ~factor(.,
                                         levels = c("0",    "1",    "2",   "3",          "5",      "6",               "9",          "N",       "W",          "M",         "B",     "A"),
                                         labels = c("None", "Pill", "IUD", "Injectable", "Condom", "F.sterilization", "Withdrawal", "Implant", "Other.trad", "Other.mod", "Birth", "Abstinence")))

matrices <- data %>%
  mutate(postpartum = case_when(month.a == "Birth" ~ "1", pp.6 == F & termination == F ~ "No"), From = month.a, To = month.b) %>%
  filter(!is.na(postpartum)) %>%                                                                            # Not using months 1-5 postpartum
  bind_rows(data %>% filter(month.a == "Birth" & !is.na(month.6) & month.6 != "Lac.Am" & month.6 != "Termination" & month.6 != "Pregnant") %>%  # Duplicate pp rows so we can have 6 month pp timepoint
              mutate(postpartum = "6", age_grp = age_grp.6, From = month.b, To = month.6)) %>%              # For the 6 month pp matrices, replace age group with age group at the 6 month point
  filter(!(postpartum == "No" & sex.active > 311)) %>%                                                      # Filter out women in the non-postartum matrix who have not has sex in the past year+
  group_by(postpartum, From, age_grp) %>% mutate(n = sum(wt)) %>% ungroup %>%                               # sum total in 'from' each method
  group_by(postpartum, From, To, n, age_grp) %>% summarise(Freq = sum(wt)) %>% mutate(Freq = Freq/n) %>% ungroup %>%
  full_join(expand.grid(age_grp = unique(data$age_grp), From = unique(levels(data$month.a))[1:11], To = unique(levels(data$month.a))[1:11], postpartum = c("6", "No")) %>% # Add in any missing rows to make sure matrices will be square
              filter(From != 'Birth' & From != "None" & To != "Birth")) %>%                                 # Don't need from none or birth or to birth in the switching matrices so remove from new rows
  mutate(Freq = ifelse(is.na(Freq), ifelse(From == To, 1, 0), Freq), n = ifelse(is.na(n), 0, n)) %>%  ungroup %>% # for added rows, set p(x to x) to 1 and all others to 0, and set n to 0
  spread(To, Freq, fill = 0) %>% filter(rowSums(.[5:14], na.rm = T) > 0) %>%                                # spread to wide format by each method, for missing fill with 0, for some reason adding columns with 0 for everything so filtered those out
  filter(!(From == "F.sterilization" & F.sterilization == 0)) %>%                                           # one woman in <18 is going from sterilization to none and there's no other women in that category so filter her out)
  mutate(group = paste0("grp", postpartum, age_grp)) %>% as.data.table()                                    # create a group variable and make into a data table for matrix manipulation


npdata <- matrices[postpartum == "No"]
groups <- unique(npdata[, group])
for (g in groups){
  m <- as.matrix(npdata[group == g, 5:14])
  m <- m %^% 12                                                                                             # matrix multiplication for the non postpartum switching matrices to convert from 1 month to 1 yr probabilities
  npdata[group == g, 5:14] <- as.data.frame(m)
}
matrices.result <- as.data.frame(data.table::rbindlist(list(matrices[postpartum != "No"], npdata))) %>%
  mutate(age_grp = factor(age_grp, levels = c("<18", "18-20", "20-25", "25-35", ">35")),
         From = factor(From, levels = c("Birth", "None", "Withdrawal", "Other.trad", "Condom", "Pill", "Injectable", "Implant", "IUD", "F.sterilization", "Other.mod"))) %>%
  select(postpartum, age_grp, From, n, None, Withdrawal, Other.trad, Condom, Pill, Injectable, Implant, IUD, F.sterilization, Other.mod) %>% arrange( postpartum, age_grp, From) 

rowSums(matrices.result[, 5:14], na.rm = T) # check sum to 1

# save matrices
# write.table(matrices.result, file="senegal/matrices_senegal_dhs_2017_18.csv", sep=",", row.names = F) 
# write.table(matrices.result, file="kenya/matrices_kenya_dhs_2014.csv", sep=",", row.names = F) 
# write.table(matrices.result, file="ethiopia/matrices_ethiopia_dhs_2008_11.csv", sep=",", row.names = F) 