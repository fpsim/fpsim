#############################################
# -- Longitudinal analysis on PMA data
# -- For empowerment implementation in FPsim
# -- Marita Zimmermann
# -- October 2023
#############################################

rm(list=ls())

library(lavaan)
library(tidySEM)
library(ggpubr)
library(extrafont)
library(ggalluvial)
library(survey)
library(lavaan.survey)
library(tidyverse)      
library(haven)  
library(withr)  

pma.path <- normalizePath(file.path(Sys.getenv("ONEDRIVE"), "WRICH/Data/PMA"), "/")
All_data_wide_clpm <- with_dir(pma.path, {read.csv("Kenya/All_data_wide_clpm.csv")})
All_data_long <- with_dir(pma.path, {read.csv("Kenya/All_data_long.csv")})
options(survey.lonely.psu="adjust")

# duplicate dataset to increase sample size... so we have a 1-2 time and 2-3 time
data.edit.2 <- All_data_long %>%
  filter(wave != 1) %>% mutate(wave = case_when(wave == 2 ~ 1, wave == 3 ~ 2), time = "b")
filter_data <- All_data_long %>%
  filter(wave != 3) %>% mutate(time = "a") %>% bind_rows(data.edit.2) %>% # a for timepoint 1-2
  # to wide 
  gather(var, val, -wave, -female_ID, -time) %>% unite("var", c(var, wave)) %>% spread(var, val, convert = T, drop = F, fill = NA) %>%
  # take out no EA ID
  mutate(FQweight = ifelse(is.na(FQweight_2), FQweight_1, FQweight_2),
         strata = ifelse(is.na(strata_2), strata_1, strata_2),
         EA_ID = ifelse(is.na(EA_ID_2), EA_ID_1, EA_ID_2)) %>%
  filter(!is.na(EA_ID)) 
  # refuse sex only waves 1 and 3, and no variation, so taking that one out

filter.data.notpp <- filter_data %>% filter((pp.time_2>6 | is.na(pp.time_2)) & pregnant_2 == 0)
filter.data.pp1 <- filter_data %>% filter(pp.time_2<2)
filter.data.pp6 <- filter_data %>% filter(pp.time_2<6)

svydes <- svydesign(id = ~EA_ID, strata = ~strata, weights =  ~FQweight, data = filter.data.notpp , nest = T)
svydes.pp1 <- svydesign(id = ~EA_ID, strata = ~strata, weights =  ~FQweight, data = filter.data.pp1 , nest = T)
svydes.pp6 <- svydesign(id = ~EA_ID, strata = ~strata, weights =  ~FQweight, data = filter.data.pp6 , nest = T)
other.outcomes <- c("paidw_12m", "decide_spending_mine", "buy_decision_health")
      
# Empowerment
empower_results <- list()
modellist <- list()
for (i in other.outcomes) {
  print(i)
  model <- svyglm(as.formula(paste0(i,"_2 ~ current_contra_1 + ",i,"_1  + age_2 + school_2 + live_births_2 + urban_2 + wealthquintile_2")), 
                  family = quasibinomial(), 
                  #design = svydes)
                  #design = svydes.pp1)
                  design = svydes.pp6)
modellist[[i]] <- model
  empower_results[[i]] <- as.data.frame(summary(model)$coefficients) %>% 
    mutate(lhs = i, rhs = rownames(.)) }
empower_coef <- bind_rows(empower_results)  %>%
  # rename variables to match model
  mutate(across(c(lhs, rhs), ~gsub("_3", "", gsub("_2", "_0", 
                                                  gsub("school","edu_attainment",
                                                       gsub("paidw_12m","paid_employment",
                                                            gsub("decide_spending_mine","decision_wages",
                                                                 gsub("buy_decision_health","decision_health",
                                                                      gsub("wge_sex_eff_tell_no","sexual_autonomy",
                                                                           gsub("married","partnered",
                                                                                gsub("live_births", "parity", 
                                                                                     gsub("current_contra", "contraception", .))))))))))))

# write.csv(empower_coef, "fpsim/locations/kenya/empower_coef.csv", row.names = F)
# write.csv(empower_coef, "fpsim/locations/kenya/empower_coef_pp1.csv", row.names = F)
# write.csv(empower_coef, "fpsim/locations/kenya/empower_coef_pp6.csv", row.names = F)


# Contraception
model <- svyglm(current_contra_2 ~ current_contra_1 + paidw_12m_1 + decide_spending_mine_1 + buy_decision_health_1 + age_2 + school_2 + live_births_2 + urban_2 + wealthquintile_2, 
                family = quasibinomial(), 
                #design = svydes)
                design = svydes.pp1)
                #design = svydes.pp6)
contra_coef <- as.data.frame(summary(model)$coefficients) %>% 
  mutate(rhs = rownames(.)) %>%
  mutate(rhs = gsub("_3", "", gsub("_2", "_0", 
                                   gsub("school","edu_attainment",
                                        gsub("paidw_12m","paid_employment",
                                             gsub("decide_spending_mine","decision_wages",
                                                  gsub("buy_decision_health","decision_health",
                                                            gsub("married","partnered",
                                                                 gsub("live_births", "parity", 
                                                                      gsub("current_contra", "contraception", rhs))))))))))

# write.csv(contra_coef, "fpsim/locations/kenya/contra_coef.csv", row.names = F)
# write.csv(contra_coef, "fpsim/locations/kenya/contra_coef_pp1.csv", row.names = F)
# write.csv(contra_coef, "fpsim/locations/kenya/contra_coef_pp6.csv", row.names = F)


# Method choice arrays
data.path <- normalizePath(file.path(Sys.getenv("ONEDRIVE"), "DHS"), "/")
data.raw <- with_dir(data.path, {read_dta("KE_2014_DHS_09162022_2324_122388/KEIR72DT/KEIR72FL.DTA")}) 

data <- data.raw %>%
  mutate(wt = v005/1000000,
         age = as.numeric(v012), 
         parity=as.numeric(v220),
         pp.1 = str_detect(substr(str_squish(vcal_1), 1, 2), "B"), # postpartum in past 6 months
         pp.6 = str_detect(substr(str_squish(vcal_1), 1, 7), "B"), # postpartum in past 6 months
         age_grp = case_when(age <= 18 ~ "<18",                                          
                             age > 18 & age <= 20 ~ "18-20",
                             age > 20 & age <= 25 ~ "20-25",
                             age > 25 & age <= 30 ~ "25-30",
                             age > 30 & age <= 35 ~ "30-35",
                             age > 35 ~ ">35"),
         curr_use = ifelse(v312 == 0, 0, ifelse(is.na(v312), NA, 1)),
         method = factor(case_when(v312 == 7 ~ 17, # male sterilization to other modern
                            v312 == 13 ~ NA, # LAM to missing
                            v312 == 14 ~ 17, # female condom to other modern
                            v312 == 8 ~ 10, # abstinence to other traditional
                            T ~ v312),
                         levels = c(1,2,3,5,6,9,10,11,17),
                         labels = c("Pill", "IUD", "Injectable", "Condom", "F.sterilization", "Withdrawal", "Other.trad", "Implant", "Other.mod"))) 
data.notpp <- data %>% filter(pp.1 == F & pp.6 == F)
data.pp1 <- data %>% filter(pp.1)
data.pp6 <- data %>% filter(pp.6)

svydes1 <- svydesign(id = ~v001, strata= ~v023, weights = ~wt, data=data.notpp, nest = T)
svydes1.pp1 <- svydesign(id = ~v001, strata= ~v023, weights = ~wt, data=data.pp1, nest = T)
svydes1.pp6 <- svydesign(id = ~v001, strata= ~v023, weights = ~wt, data=data.pp6, nest = T)
methods.notpp <- as.data.frame(svytable(~method+age_grp+parity, svydes1)) %>%
  group_by(age_grp, parity) %>% mutate(percent = Freq/sum(Freq)) %>% select(-Freq)
# write.csv(methods, "fpsim/locations/kenya/method_mix.csv", row.names = F)
# not enough pp women so have to combne age parity groups
methods.pp1 <- as.data.frame(svytable(~method, svydes1.pp1)) %>% mutate(percent = Freq/sum(Freq)) %>% select(-Freq)
# write.csv(methods.pp1, "fpsim/locations/kenya/method_mix_pp1.csv", row.names = F)
methods.pp6 <- as.data.frame(svytable(~method, svydes1.pp6)) %>% mutate(percent = Freq/sum(Freq)) %>% select(-Freq)
# write.csv(methods.pp1, "fpsim/locations/kenya/method_mix_pp1.csv", row.names = F)

methods %>%
ggplot()+
  geom_line(aes(y = percent, x = method, group = parity, color = parity)) +
  facet_wrap(~age_grp)

methods.pp1 %>%
  ggplot()+
  geom_point(aes(y = percent, x = method))


