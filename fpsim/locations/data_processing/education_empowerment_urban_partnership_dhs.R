#########################################
# -- Education data from DHS         - ##
# -- DHS data
# -- Marita Zimmermann
# -- August 2023
#########################################

rm(list=ls())

# -- Libraries -- #
library(tidyverse)      
library(haven)   
library(survey)

# -- Data -- #

# Kenya 2022 household data
data.raw.h <- read_dta("C:/Users/maritazi/OneDrive - Bill & Melinda Gates Foundation/DHS/KEHR8ADT/KEHR8AFL.DTA")
# Kenya 2022 individual recode
data.raw <- read_dta("C:/Users/maritazi/OneDrive - Bill & Melinda Gates Foundation/DHS/KEIR8ADT/KEIR8AFL.DTA")

# -- Clean data -- #

data <- data.raw %>%
  mutate(age = v012, parity = v220, edu = v133,
         paidwork = case_when(v731 %in% c(1,2,3) & v741 %in% c(1,2,3) ~ 1, # done work in the past year and paid in cash or in kind
                              v731 == 0 | v741 == 0 ~ 0),
         decisionwages = case_when(v739 == 1 ~ 1, v739 %in% c(2,3) ~ 0.5, v739 %in% c(4,5) ~ 0), # 1 she decides, 0.5 she decides with someone else, 0 someone else decides
         refusesex = case_when(v850a == 1 ~ 1, v850a == 8 ~ 0.5, v850a == 0 ~ 0), # 1 she can refuse sex, 0.5 don't know/it depends, 0 no
         urban = ifelse(v025 == 1, 1, 0), # 1 if urban
         age_partner = v511) # age at first cohabitation
svydes1 = svydesign(id = data$v001, strata=data$v023, weights = data$v005/1000000, data=data)


#  -- Education -- # 

# table of education by age and parity
table.edu.ind <- as.data.frame(svytable(~age+edu+parity, svydes1))

# Create data frame for education for girls age 6-15 from household survey
data.h <- data.raw.h %>%
  select(starts_with("hv103"), starts_with("hv105"), starts_with("hv104"), starts_with("hv109"), 
         starts_with("hhid"), starts_with("hv001"), starts_with("hv023"), starts_with("hv005")) %>%
  gather(var, val, -c(hhid, hv001, hv023, hv005)) %>% mutate(num = substr(var,7,9), var = substr(var,1,5)) %>% spread(var, val) %>%
  filter(hv103 == 1 & hv105 %in% c(6:14) & hv104 == 2) %>% # de facto hh member, age 6-15, female
  rename(edu = hv109, age = hv105) %>%
  mutate(parity = 0) # assume girls under age 15 are parity 0
svydes2 = svydesign(id = data.h$hv001, strata=data.h$hv023, weights = data.h$hv005/1000000, data=data.h)

# table of education by age for youth
table.edu.hh <- as.data.frame(svytable(~age+edu+parity, svydes2)) 

# combine age group tables
table.edu <- table.edu.hh %>% 
  bind_rows(table.edu.ind) %>%
  mutate(edu = as.numeric(as.character(edu))) %>%
  group_by(age, parity) %>% arrange(-edu) %>%
  mutate(total = sum(Freq), percent.dist = Freq/total, # percentage with each year of education in the age/parity group
         sum = cumsum(Freq), cum.percent = sum/total) %>%
  select(-total, -sum, -Freq)
# write.csv(table.edu, "fpsim/locations/kenya/education.csv", row.names = F)
# final table: percent.dist is the percent distribution with each year of education for that age/parity group,
# cum.percent is the percentage in the age/parity group with that year or more of education

# -- Empowerment -- #

# Table of the three empowerment outcomes by age
table.emp <- as.data.frame(svyby(~paidwork, ~age, svydes1, svymean)) %>% rename(paidwork.se = se) %>%
  left_join(as.data.frame(svyby(~decisionwages, ~age, svydes1, svymean, na.rm = T)) %>% rename(decisionwages.se = se)) %>%
  left_join(as.data.frame(svyby(~refusesex, ~age, svydes1, svymean, na.rm = T)) %>% rename(refusesex.se = se))
# write.csv(table.emp, "fpsim/locations/kenya/empowerment.csv", row.names = F)

# -- urban/rural -- #

table.urban <- as.data.frame(svymean(~urban, svydes1)) %>% rename(urban.se = urban)
# write.csv(table.urban, "fpsim/locations/kenya/urban.csv", row.names = F)

# -- age at partnership -- #

table.partner <- as.data.frame(svytable(~age_partner, svydes1)) %>%
  mutate(percent = Freq/sum(Freq)) %>% select(-Freq)
# write.csv(table.partner, "fpsim/locations/kenya/age_partnership.csv", row.names = F)




