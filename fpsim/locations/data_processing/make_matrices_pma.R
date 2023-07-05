#########################################
# -- Create switching matrix for ABM - ##
# -- PMA data
# -- Marita Zimmermann
# -- August 2022
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
data.path <- normalizePath(file.path(Sys.getenv("ONEDRIVE"), "WRICH/Data/PMA/Kenya"), "/")
data1.raw <- with_dir(data.path, {read_dta("PMA2019_KEP1_HQFQ_v2.0_25Aug2021/PMA2019_KEP1_HQFQ_v2.0_25Aug2021.DTA")})
data2.raw <- with_dir(data.path, {read_dta("PMA2020_KEP2_HQFQ_v2.0_25Jan2022/PMA2020_KEP2_HQFQ_v2.0_25Jan2022.DTA")})
data3.raw <- with_dir(data.path, {read_dta("PMA2022_KEP3_HQFQ_v2.0_17Aug2022/PMA2022_KEP3_HQFQ_v2.0_17Aug2022.DTA")})
var_labels1 <- pivot_longer(as.data.frame(var_label(data1.raw)), cols = 1:611, names_to = "Variable", values_to = "Description") # data frame of variable descriptions
var_labels2 <- pivot_longer(as.data.frame(var_label(data2.raw)), cols = 1:567, names_to = "Variable", values_to = "Description") # data frame of variable descriptions
val_labels1 <- val_labels(data1.raw) # value labels
val_labels2 <- val_labels(data2.raw) # value labels

with(data1.raw, summary(parse_date_time(FQdoi_corrected, "b d Y H M S Op")))
with(data2.raw, summary(parse_date_time(FQdoi_corrected, "b d Y H M S Op")))
with(data3.raw, summary(parse_date_time(FQdoi_correctedSIF, "Y m d H M S")))

# keep only calendar data, age, and parity
data <- data2.raw %>%
  bind_rows(data1.raw) %>% mutate(county = as.character(county)) %>%
  bind_rows(data3.raw %>% mutate_at(c("doi_corrected", "county"), list(~as.character(.)))) %>%
  select(birth_events, ur, age = FQ_age, calendar_c1_full, phase, wt = FQweight, last_time_sex, last_time_sex_value) %>%
  mutate(parity1 = case_when(birth_events>6 ~ 6, is.na(birth_events) ~ 0, T ~ birth_events),
         sex.active = case_when(last_time_sex>0 & last_time_sex_value < 100 ~ as.numeric(paste0(last_time_sex, ifelse(last_time_sex_value<10,0,""), last_time_sex_value))),
         cal = str_squish(calendar_c1_full),
         cal = gsub("30","39",cal),                                                                         # recode contraceptive methods, Periodic abstinence/rhythm to other traditional
         cal = gsub("13","M",cal), cal = gsub("11", "M", cal), cal = gsub("10", "M", cal),                  # recode contraceptive methods to other modern, std days, diaphragm, female condom
         cal = gsub("8", "M", cal), cal = gsub("12", "M", cal), cal = gsub(",2", ",M", cal),                # recode contraceptive methods to other modern, emergency contraception, foam and jelly, male sterilization  
         cal = gsub("14", "L", cal), cal = gsub("31", "R", cal), cal= gsub("39", "W", cal),                 # change double digit codes to single
         cal = gsub(",", "", cal),                                                                          # remove commas
         cal_clean = sapply(strsplit(cal, split = ""),  function(str) {paste(rev(str), collapse = "")}),    # reverse string so oldest is first
         pattern_split = strsplit(cal_clean,""),                                                            # split into individual characters
         month.a = lapply(pattern_split, function(m) m[1:35]),                                              # codes for months 1-35
         month.b = lapply(pattern_split, function(m) m[2:36]),                                              # codes for months 2-36
         obs = 1:nrow(.)) %>%                                                                               # create obs variable to identify the individual
  unnest(c(month.a, month.b)) %>%                                                                           # create long format
  mutate(month.a = ifelse(month.a == "2", "M", month.a), month.b = ifelse(month.b == "2", "M", month.b)) %>%# correct an error where some 2's remained at first calendar position
  group_by(obs) %>% mutate(month = row_number(),                                                            # create variable for month from beginning of calendar
                           age = age - (36-month)/12,                                                       # adjust for age in calendar
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
  filter(month.b != "B") %>%                                                                                # had to remove a few (17) weird entries of none to birth or birth to birth
  filter(phase == 1 | (phase == 2 & month > 26) | (phase == 3 & month > 26)) %>%                            # remove Nov 2017 - Dec 2019 (26 months) from phase 2 cal because that will overlap with phase 1, similarly remove Nov 2018 - Dec 2020 from cal 3
  mutate_at(c("month.a", "month.b", "month.6"), ~factor(.,
                                         levels = c("0",    "7",    "4",   "5",          "9",      "1",               "R",          "L",      "3",       "W",          "M",         "P",        "T",           "B",     "A"),
                                         labels = c("None", "Pill", "IUD", "Injectable", "Condom", "F.sterilization", "Withdrawal", "Lac.Am", "Implant", "Other.trad", "Other.mod", "Pregnant", "Termination", "Birth", "Abstinence")))

matrices <- data %>%
  mutate(postpartum = case_when(month.a == "Birth" ~ "1", pp.6 == F & termination == F ~ "No"), From = month.a, To = month.b) %>%
  filter(!is.na(postpartum)) %>%                                                                            # Not using months 1-5 postpartum
  bind_rows(data %>% filter(month.a == "Birth" & !is.na(month.6) & month.6 != "Lac.Am" & month.6 != "Termination" & month.6 != "Pregnant") %>%  # Duplicate pp rows so we can have 6 month pp timepoint
              mutate(postpartum = "6", age_grp = age_grp.6, From = month.b, To = month.6)) %>%              # For the 6 month pp matrices, replace age group with age group at the 6 month point
  filter(!(postpartum == "No" & sex.active > 311)) %>%                                                      # Filter out women in the non-postartum matrix who have not has sex in the past year+
  group_by(postpartum, From, age_grp) %>% mutate(n = sum(wt, na.rm = T)) %>% ungroup %>%                                   # sum total in 'from' each method
  group_by(postpartum, From, To, n, age_grp) %>% summarise(Freq = sum(wt, na.rm = T)) %>% mutate(Freq = Freq/n) %>% ungroup %>%
  full_join(expand.grid(age_grp = unique(data$age_grp), From = unique(data$month.a), To = unique(data$month.a), postpartum = c("6", "No")) %>% # Add in any missing rows to make sure matrices will be square
              filter(From != 'Birth' & From != "None" & To != "Birth")) %>%                                 # Don't need from none or birth or to birth in the switching matrices so remove from new rows
  mutate(Freq = ifelse(is.na(Freq), ifelse(From == To, 1, 0), Freq), n = ifelse(is.na(n), 0, n)) %>%  ungroup %>% # for added rows, set p(x to x) to 1 and all others to 0, and set n to 0
  spread(To, Freq, fill = 0) %>% filter(rowSums(.[5:14], na.rm = T) > 0) %>%                                # spread to wide format by each method, for missing fill with 0, for some reason adding columns with 0 for everything so filtered those out
  filter(!(From == "F.sterilization" & F.sterilization == 0)) %>%                                           # one woman in <18 is going from sterilization to none and there's no other women in that category so filter her out
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
# write.table(matrices.result, file="kenya/matrices_kenya_pma_2019_20_21.csv", sep=",", row.names = F) 
