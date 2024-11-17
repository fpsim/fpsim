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
data.raw <- read_dta("C:/Users/maritazi/OneDrive - Bill & Melinda Gates Foundation/DHS/SN_2018_ContinuousDHS_10072020_1835_122388/SNIR80DT/SNIR80FL.DTA") # Senegal

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

# -- PMA data -- #
# Run PMA_recode
svypma <- svydesign(id = ~EA_ID, strata = ~strata, weights =  ~FQweight, data = All_data , nest = T)


# -- Empowerment metrics -- #

# Table of the empowerment outcomes by age
# Define a function to simplify the svyby and rename steps
calculate_svyby <- function(variable, design, group_var) {
  as.data.frame(svyby(as.formula(paste0("~", variable)), as.formula(paste0("~", group_var)), design, svymean, na.rm = TRUE)) %>%
    rename_at(vars(se), ~paste0(variable, ".se"))}
results.dhs <- lapply(c("paid_employment", "decision_wages", "sexual_autonomy", "decision_purchase", "decision_health"), calculate_svyby, design = svydes1, group_var = "age")
results.pma <- lapply(c("buy_decision_major", "buy_decision_daily", "decide_spending_partner", "buy_decision_clothes", "has_savings", "has_fin_knowl", "has_fin_goals"), calculate_svyby, design = svypma, group_var = "age")
table.emp <- Reduce(function(x, y) left_join(x, y, by = "age"), c(results.dhs, results.pma))
# write.csv(table.emp, "fpsim/locations/kenya/empowerment.csv", row.names = F)


# Ability to refuse sex
table.emp.1 <- table.emp %>%
  gather(var, val, -age) %>% mutate(var2 = ifelse(grepl("\\.se", var), "se", "value"), var = gsub("\\.se", "", var)) %>% spread(var2, val) %>%
  filter(var == "sexual_autonomy")
pred1 <- lm(value~age + I((age-25)*(age>=25)), data = table.emp.1)
table.emp.1 %>%
  ggplot() +
  geom_point(aes(y = value, x = age)) +
  geom_line(aes(y=predict(pred1), x = age)) +
  ylab("Sexual autonomy: ability to refuse sex") +
  theme_bw(base_size = 13)
predict(pred1)[25-14]
summary(pred1)
summary(pred1)$coefficients[2,]
summary(pred1)$coefficients[2,] + summary(pred1)$coefficients[3.]

# Decision making autonomy over her wages
table.emp.2 <- table.emp %>%
  gather(var, val, -age) %>% mutate(var2 = ifelse(grepl("\\.se", var), "se", "value"), var = gsub("\\.se", "", var)) %>% spread(var2, val) %>%
  filter(var == "decision_wages")
pred2 <- lm(value~age + I((age-28)*(age>=28)), data = table.emp.2)
table.emp.2 %>%
  ggplot() +
  geom_point(aes(y = value, x = age)) +
  geom_line(aes(y=predict(pred2), x = age)) +
  ylab("Decision making: her wages") +
  theme_bw(base_size = 13)
predict(pred2)[28-14]
summary(pred2)
summary(pred2)$coefficients[2,]
summary(pred2)$coefficients[2,] + summary(pred2)$coefficients[3.]

# Paid work
table.emp.3 <- table.emp %>%
  gather(var, val, -age) %>% mutate(var2 = ifelse(grepl("\\.se", var), "se", "value"), var = gsub("\\.se", "", var)) %>% spread(var2, val) %>%
  filter(var == "paid_employment")
pred3 <- lm(value~age + I((age-25)*(age>=25)), data = table.emp.3)
table.emp.3 %>%
  ggplot() +
  geom_point(aes(y = value, x = age)) +
  geom_line(aes(y=predict(pred3), x = age)) +
  ylab("Paid work") +
  theme_bw(base_size = 13)
predict(pred3)[25-14]
summary(pred3)
summary(pred3)$coefficients[2,]
summary(pred3)$coefficients[2,] + summary(pred3)$coefficients[3.]

# Decision making power over health
table.emp.4 <- table.emp %>%
  gather(var, val, -age) %>% mutate(var2 = ifelse(grepl("\\.se", var), "se", "value"), var = gsub("\\.se", "", var)) %>% spread(var2, val) %>%
  filter(var == "decision_health")
table.emp.4 %>%
  ggplot() +
  geom_point(aes(y = value, x = age)) +
  ylab("Decision making autonomy over health") +
  theme_bw(base_size = 13)

# -- urban/rural -- #

table.urban <- as.data.frame(svymean(~urban, svydes1)) %>% rename(urban.se = urban)
# write.csv(table.urban, "fpsim/locations/kenya/urban.csv", row.names = F)
# write.csv(table.urban, "fpsim/locations/senegal/urban.csv", row.names = F)

# -- age at partnership -- #

table.partner <- as.data.frame(svytable(~age_partner, svydes1)) %>%
  mutate(percent = Freq/sum(Freq)) %>% select(-Freq)
# write.csv(table.partner, "fpsim/locations/kenya/age_partnership.csv", row.names = F)

# -- education -- #

# - for initialization
# Visualize education by age
table.edu.mean <- as.data.frame(svyby(~edu, ~age, svydes1, svymean))
# create projections for older and younger women by slope
table.edu.lag <- table.edu.mean %>%
  mutate(slope = (edu-lag(edu,1)) / (age-lag(age,1)),
         group = case_when(age<20~1, age>43~2)) %>%
  group_by(group) %>% mutate(avg.slope = ifelse(group == 1, mean(slope, na.rm = T), NA),
                             avg.edu = ifelse(group == 2, mean(edu), NA))
# visualize
table.edu.mean %>%
  ggplot() +
  geom_line(aes(y = edu, x = age)) +
  geom_ribbon(aes(ymin = edu-se, ymax = edu+se, x = age), alpha = 0.5) +
  geom_segment(aes(x = 49, y = min(table.edu.lag$avg.edu, na.rm = T), xend = 70, yend = min(table.edu.lag$avg.edu, na.rm = T))) +
  geom_segment(aes(y = 0, x = 15 - min(table.edu.lag$edu, na.rm = T)/min(table.edu.lag$avg.slope, na.rm = T), xend = 15, yend = min(table.edu.lag$edu, na.rm = T))) +
  ylab("Mean years of education") + xlab("Age") +
  theme_bw(base_size = 13)

# final edu table for initialization
table.edu.inital <- data.frame(age = c(1:14, 50:99)) %>%
  mutate(edu = ifelse(age<15, pmax(min(table.edu.lag$edu, na.rm = T) - min(table.edu.lag$avg.slope, na.rm = T)*(15-age), 0), min(table.edu.lag$avg.edu, na.rm = T))) %>%
  bind_rows(table.edu.mean)
# write.csv(table.edu.inital, "fpsim/locations/kenya/edu_initialization.csv", row.names = F)

# - for education objective
# Distribution of years of education for all women over age 20 (assumed that they have finished edu)
data.20 <- data %>% filter(age>20)
svydes2 = svydesign(id = data.20$v001, strata=data.20$v023, weights = data.20$v005/1000000, data=data.20)
table.edu.20 <- as.data.frame(svytable(~edu+urban, svydes2)) %>% group_by(urban) %>% mutate(percent = Freq/sum(Freq)) %>% select(-Freq)
table.edu.20 %>%
  ggplot() +
  geom_line(aes(y = percent, x = edu, group = urban, color = urban)) +
  ylab("Percent of women") +
  theme_bw(base_size = 13)
# write.csv(table.edu.20, "fpsim/locations/kenya/edu_objective.csv", row.names = F)

# - For interruption following pregnancy
# Need to use PMA data because it has age at stopping education (DHS doesn't)
pma.path <- normalizePath(file.path(Sys.getenv("ONEDRIVE"), "WRICH/Data/PMA"), "/")

data.raw.pma <-  with_dir(pma.path, {read_dta("Kenya/PMA2019_KEP1_HQFQ_v3.0_21Oct2021/PMA2019_KEP1_HQFQ_v3.0_21Oct2022.DTA")}) %>% mutate(wave = 1) %>% mutate_at(c("RE_ID", "county"), list(~as.character(.))) %>%
  bind_rows(with_dir(pma.path, {read_dta("Kenya/PMA2020_KEP2_HQFQ_v3.0_21Oct2022/PMA2021_KEP2_HQFQ_v3.0_21Oct2022.DTA")}) %>% mutate(wave = 2) %>% mutate_at(c("RE_ID", "county"), list(~as.character(.)))) %>%
  bind_rows(with_dir(pma.path, {read_dta("Kenya/PMA2022_KEP3_HQFQ_v3.0_21Oct2022/PMA2022_KEP3_HQFQ_v3.0_21Oct2022.DTA")}) %>% mutate(wave = 3) %>% mutate_at(c("doi_corrected", "county"), list(~as.character(.))))

# recode data for school and birth timing
data.pma <- data.raw.pma %>% filter(!is.na(FQweight)) %>%
  mutate(birthage1 = floor(as.numeric(difftime(parse_date_time(first_birth, "Y-m-d"), parse_date_time(birthdateSIF, "Y-m-d"), units = "weeks")/52)), # age at first birth
         school_birth1 = ifelse(between(school_left_age, birthage1-1, birthage1+1), 1, 0), # stop school within one year of first birth
         school_birth1 = ifelse(is.na(school_birth1), 0, school_birth1), # replace NA with no
         birthage2 = floor(as.numeric(difftime(parse_date_time(recent_birthSIF, "Y-m-d"), parse_date_time(birthdateSIF, "Y-m-d"), units = "weeks")/52)), # age at most recent (but not first) birth
         school_birth2 = ifelse(between(school_left_age, birthage2-1, birthage2+1), 1, 0), # stopped school within one year of recent birth
         school_birth2 = ifelse(is.na(school_birth2), 0, school_birth2), # replace NA with no
         recentbyoung = ifelse(recent_birthSIF != first_birthSIF & birthage2<23, 1, 0)) %>% # was recent birth at age 22 or younger
  filter(birth_events>0) # only use data for women who have had a birth
svydes3 <- svydesign(id = ~EA_ID, strata = ~strata, weights =  ~FQweight, data = data.pma, nest = T)

# table of probability of stopping school by age and parity
stop.school <- as.data.frame(prop.table(svytable(~school_birth1+birthage1, svydes3), margin = 2)) %>% filter(school_birth1 == 1) %>% rename(`1` = Freq, age = birthage1) %>% select(-school_birth1) %>%
  left_join(stop.school2 <- as.data.frame(prop.table(svytable(~school_birth2+birthage2, svydes3), margin = 2)) %>% filter(school_birth2 == 1) %>% rename(`2+` = Freq, age = birthage2) %>% select(-school_birth2)) %>%
  gather(parity, percent, -age)
stop.school %>%
  ggplot() +
  geom_point(aes(y = percent, x = age, color = parity))
# write.csv(stop.school, "fpsim/locations/kenya/edu_stop.csv", row.names = F)


# -- wealth quintile -- #
table.wealth <- as.data.frame(svytable(~v190, svydes1)) %>%
  mutate(percent = Freq/sum(Freq)) %>%
  rename(wealth.quint = v190) %>% select(-Freq)
# write.csv(table.wealth, "fpsim/locations/kenya/wealth.csv", row.names = F)
# write.csv(table.wealth, "fpsim/locations/senegal/wealth.csv", row.names = F)
