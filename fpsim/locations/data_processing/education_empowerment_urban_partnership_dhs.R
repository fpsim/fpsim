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

# table of ages at first birth
as.data.frame(svytable(~v212, svydes1)) %>% mutate(perc = Freq/sum(Freq), cumsum = cumsum(Freq)/sum(Freq))
# 5.7% of women had age at first birth before 15

#  -- Education -- # 

# table of education by age and parity for age 15+
table.edu.ind <- as.data.frame(svytable(~age+edu+parity, svydes1))

# For women who have a birth before age 15, create matrix for parity 1+ and age<15 using birth calendar
data.edu <- data %>%
  select(caseid, v001, v023, v005, v212, v011, age = v012, edu = v133, starts_with("b3"), starts_with("bord")) %>%
  filter(v212 < 15) %>% # age at first birth under 15
  gather(var, val, -caseid, -v212, -v011, -age, -edu, -v001, -v023, -v005) %>% separate(var, c("var", "order")) %>% spread(var, val) %>% # one row per pregnancy
  mutate(age_at_birth = floor((b3 - v011)/12)) %>%
  filter(age_at_birth < 15) %>% # look at birth under age 15
  mutate(edu_at_birth = pmin(age_at_birth - 6, edu)) # assume school starts at age 6, women continue education from then until age at birth or until their actual education level, whichever is less
svydes2 = svydesign(id = data.edu$v001, strata=data.edu$v023, weights = data.edu$v005/1000000, data=data.edu)
table.edu.young <- as.data.frame(svytable(~age_at_birth+edu_at_birth+bord, svydes2)) %>%
  rename(age = age_at_birth, edu = edu_at_birth, parity = bord)

# Option 1: For women with no birth before age 15, create matrix for each year 
data.edu.0 <- data %>%
  select(caseid, v001, v023, v005, v212, v011, age = v012, parity = v220, edu = v133) %>%
  filter(v212 < 15) %>% # include all women who didn't have a before age 15
  slice(rep(1:n(), 9)) %>% group_by(caseid) %>% mutate(age_edu = 1:n() + 5, # repeat rows for school years from age 6-14
                                                       edu_yr = pmin(age_edu - 5, edu))
svydes3 = svydesign(id = data.edu.0$v001, strata=data.edu.0$v023, weights = data.edu.0$v005/1000000, data=data.edu.0)
table.edu.young.0 <- as.data.frame(svytable(~age_edu+edu_yr, svydes3)) %>%
  rename(age = age_edu, edu = edu_yr) %>% mutate(parity = 0)


# Option 2: For 0 parity under age 15, create data frame for education for girls age 6-15 from household survey
data.h <- data.raw.h %>%
  select(starts_with("hv103"), starts_with("hv105"), starts_with("hv104"), starts_with("hv109"), 
         starts_with("hhid"), starts_with("hv001"), starts_with("hv023"), starts_with("hv005")) %>%
  gather(var, val, -c(hhid, hv001, hv023, hv005)) %>% mutate(num = substr(var,7,9), var = substr(var,1,5)) %>% spread(var, val) %>%
  filter(hv103 == 1 & hv105 %in% c(6:14) & hv104 == 2) %>% # de facto hh member, age 6-15, female
  rename(edu = hv109, age = hv105) %>%
  mutate(parity = 0) # assume girls under age 15 are parity 0
svydes4 = svydesign(id = data.h$hv001, strata=data.h$hv023, weights = data.h$hv005/1000000, data=data.h)
table.edu.hh <- as.data.frame(svytable(~age+edu+parity, svydes4)) 

# Compare options 1 and 2
compare <- table.edu.young.0 %>% 
  left_join(table.edu.hh %>% 
              rename(Freq.hh = Freq) %>% mutate( parity = as.numeric(as.character(parity)))) %>%
  mutate(edu = as.numeric(as.character(edu)),
         age = as.numeric(as.character(age))) %>%
  group_by(age, parity) %>% arrange(-edu) %>%
  mutate(cum.percent = cumsum(Freq)/sum(Freq), cum.percent.hh = cumsum(ifelse(is.na(Freq.hh), 0, Freq.hh))/sum(Freq.hh, na.rm = T), dif = cum.percent - cum.percent.hh) 
# There are some significant differences between these two approaches, for now use hh roster

# combine age group tables
table.edu <- table.edu.ind %>% 
  bind_rows(table.edu.young) %>%
  bind_rows(table.edu.hh) %>%
  mutate(edu = as.numeric(as.character(edu)),
         age = as.numeric(as.character(age))) %>%
  group_by(age, parity) %>% arrange(-edu) %>%
  mutate(total = sum(Freq), sum = cumsum(Freq), cum.percent = sum/total) %>% # percentage with each year of education in the age/parity group
  select(-total, -sum, -Freq)
# write.csv(table.edu, "fpsim/locations/kenya/education.csv", row.names = F)
# cum.percent is the percentage in the age/parity group with that year or more of education

# Visualize
table.edu %>%
  filter(!is.na(cum.percent)) %>%
  filter(edu >0) %>%
  ggplot() +
  #geom_point(aes(y = cum.percent, x = age, color = parity)) +
  geom_smooth(aes(y = cum.percent, x = age, color = parity)) +
  ylim(0,1) + ylab("prob of x+ yrs of edu") +
  theme_bw(base_size = 13) +
  theme(strip.background = element_rect(fill = "white")) +
  facet_wrap(~edu, labeller = "label_both")  

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




