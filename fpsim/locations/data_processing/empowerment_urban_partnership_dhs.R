#########################################
# -- EMpowerment data from DHS       - ##
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

# Kenya 2022 individual recode
data.raw <- read_dta("C:/Users/maritazi/OneDrive - Bill & Melinda Gates Foundation/DHS/KEIR8ADT/KEIR8AFL.DTA")

# -- Clean data -- #

data <- data.raw %>%
  mutate(age = v012, parity = v220, edu = v133,
         paidwork = case_when(v731 %in% c(1,2,3) & v741 %in% c(1,2,3) ~ 1, # done work in the past year and paid in cash or in kind
                              v731 == 0 | v741 == 0 ~ 0),
         decisionwages = case_when(v739 %in% c(1,2,3) ~ 1, v739 %in% c(4,5) ~ 0), # 1 she decides with or without someone else, 0 someone else decides
         refusesex = case_when(v850a == 1 ~ 1, v850a == 8 ~ 0.5, v850a == 0 ~ 0), # 1 she can refuse sex, 0.5 don't know/it depends, 0 no
         urban = ifelse(v025 == 1, 1, 0), # 1 if urban
         age_partner = v511) # age at first cohabitation
svydes1 = svydesign(id = data$v001, strata=data$v023, weights = data$v005/1000000, data=data)


# -- Empowerment metrics -- #

# Table of the three empowerment outcomes by age
table.emp <- as.data.frame(svyby(~paidwork, ~age, svydes1, svymean)) %>% rename(paidwork.se = se) %>%
  left_join(as.data.frame(svyby(~decisionwages, ~age, svydes1, svymean, na.rm = T)) %>% rename(decisionwages.se = se)) %>%
  left_join(as.data.frame(svyby(~refusesex, ~age, svydes1, svymean, na.rm = T)) %>% rename(refusesex.se = se))
# write.csv(table.emp, "fpsim/locations/kenya/empowerment.csv", row.names = F)

# Ability to refuse sex
table.emp.1 <- table.emp %>%
  gather(var, val, -age) %>% mutate(var2 = ifelse(grepl("\\.se", var), "se", "value"), var = gsub("\\.se", "", var)) %>% spread(var2, val) %>%
  filter(var == "refusesex") 
pred1 <- lm(value~age + I((age-25)*(age>=25)), data = table.emp.1)
table.emp.1 %>%
  ggplot() +
  geom_point(aes(y = value, x = age)) +
  geom_line(aes(y=predict(pred1), x = age)) +
  ylab("Ability to refuse sex") +
  theme_bw(base_size = 13) 
predict(pred1)[25-14]
summary(pred1)
summary(pred1)$coefficients[2,]
summary(pred1)$coefficients[2,] + summary(pred1)$coefficients[3.]

# Decision making autonomy over her wages
table.emp.2 <- table.emp %>%
  gather(var, val, -age) %>% mutate(var2 = ifelse(grepl("\\.se", var), "se", "value"), var = gsub("\\.se", "", var)) %>% spread(var2, val) %>%
  filter(var == "decisionwages") 
pred2 <- lm(value~age + I((age-20)*(age>=20)), data = table.emp.2)
table.emp.2 %>%
  ggplot() +
  geom_point(aes(y = value, x = age)) +
  geom_line(aes(y=predict(pred2), x = age)) +
  ylab("Decision making: her wages") +
  theme_bw(base_size = 13) 
predict(pred2)[20-14]
summary(pred2)
summary(pred2)$coefficients[2,]
summary(pred2)$coefficients[2,] + summary(pred2)$coefficients[3.]

# Paid work
table.emp.3 <- table.emp %>%
  gather(var, val, -age) %>% mutate(var2 = ifelse(grepl("\\.se", var), "se", "value"), var = gsub("\\.se", "", var)) %>% spread(var2, val) %>%
  filter(var == "paidwork") 
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

# -- urban/rural -- #

table.urban <- as.data.frame(svymean(~urban, svydes1)) %>% rename(urban.se = urban)
# write.csv(table.urban, "fpsim/locations/kenya/urban.csv", row.names = F)

# -- age at partnership -- #

table.partner <- as.data.frame(svytable(~age_partner, svydes1)) %>%
  mutate(percent = Freq/sum(Freq)) %>% select(-Freq)
# write.csv(table.partner, "fpsim/locations/kenya/age_partnership.csv", row.names = F)

# -- education -- #

# for initialization
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

# for education objective
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





