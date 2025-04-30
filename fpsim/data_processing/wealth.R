#########################################
# -- Empowerment data from DHS       - ##
# -- DHS data
# -- Marita Zimmermann
# -- August 2023
#########################################
 #TODO: to refactor and tidy-up
rm(list=ls())

# -- Libraries -- #
library(tidyverse)
library(haven)
library(survey)
library(withr)

# -- DHS Data -- #

# individual recode
data.raw <- read_dta("C:/Users/maritazi/OneDrive - Bill & Melinda Gates Foundation/DHS/KEIR8ADT/KEIR8AFL.DTA") # Kenya

# Senegal
filepath <- file.path("C:/Users/maritazi/OneDrive - Bill & Melinda Gates Foundation/DHS/IR_all/SNIR8BDT/SNIR8BFL.DTA")
#Ethiopia
filepath <- file.path("C:/Users/maritazi/OneDrive - Bill & Melinda Gates Foundation/DHS/IR_all/ETIR71DT/ETIR71FL.DTA")
# -- Load all the data
data.raw <- read_dta(filepath)

# -- Clean data -- #
data <- data.raw %>%
  mutate(age = v012, parity = v220, edu = v133,
         paid_employment = case_when(v731 %in% c(1,2,3) & v741 %in% c(1,2,3) ~ 1, # done work in the past year and paid in cash or in kind
                              v731 == 0 | v741 == 0 ~ 0),
         decision.wages = case_when(v739 %in% c(1,2,3) ~ 1, v739 %in% c(4,5) ~ 0), # 1 she decides with or without someone else, 0 someone else decides
         decision_wages = case_when(paid_employment == 1 & decision.wages == 1 ~ 1, paid_employment == 0 | decision.wages == 0 ~ 0), # create a combined variable for paid work and decision making autonomy
         sexual_autonomy = case_when(v850a == 1 ~ 1, v850a == 8 ~ 0.5, v850a == 0 ~ 0), # 1 she can refuse sex, 0.5 don't know/it depends, 0 no
         decision_health = case_when(v743a %in% c(1,2,3) ~ 1, v743a %in% c(4,5) ~ 0), # health, 1 she decides with or without someone else, 0 someone else decides
         decision_purchase = case_when(v743b %in% c(1,2,3) ~ 1, v743b %in% c(4,5) ~ 0), # large purchases, 1 she decides with or without someone else, 0 someone else decides
         urban = ifelse(v025 == 1, 1, 0), # 1 if urban
         age_partner = v511) # age at first cohabitation
svydes1 = svydesign(id = data$v001, strata=data$v023, weights = data$v005/1000000, data=data)



# -- wealth quintile -- #
table.wealth <- as.data.frame(svytable(~v190, svydes1)) %>%
  mutate(percent = Freq/sum(Freq)) %>%
  rename(wealth.quint = v190) %>% select(-Freq)
# write.csv(table.wealth, "fpsim/locations/kenya/wealth.csv", row.names = F)
write.csv(table.partnership, "C:/Users/maritazi/Documents/Projects/fpsim/fpsim/locations/senegal/data/wealth.csv", row.names = F)
write.csv(table.partnership, "C:/Users/maritazi/Documents/Projects/fpsim/fpsim/locations/ethiopia/data/wealth.csv", row.names = F)
