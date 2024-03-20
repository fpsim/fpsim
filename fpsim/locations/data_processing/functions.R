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
options(survey.lonely.psu="adjust")

data.edit.1 <- All_data_wide_clpm %>%
  # create long verson of data
  select(-EA_ID, -strata, -FQweight) %>%
  mutate(id = row_number()) %>%
  gather(var, val, -id) %>% mutate(wave = as.numeric(str_sub(var, -1)), var = str_sub(var, 1, -3)) %>% spread(var, val, convert = T, drop = F, fill = NA)
# duplicate dataset to increase sample size... so we have a 1-2 time and 2-3 time
data.edit.2 <- data.edit.1 %>%
  filter(wave != 1) %>% mutate(wave = case_when(wave == 2 ~ 1, wave == 3 ~ 2), time = "b") # b for timepoint 2-3
filter_data <- data.edit.1 %>%
  filter(wave != 3) %>% mutate(time = "a") %>% bind_rows(data.edit.2) %>% # a for timepoint 1-2
  # back to wide 
  gather(var, val, -wave, -id, -time) %>% unite("var", c(var, wave)) %>% spread(var, val, convert = T, drop = F, fill = NA) %>%
  # take out no EA ID
  mutate(FQweight = ifelse(is.na(FQweight_2), FQweight_1, FQweight_2),
         strata = ifelse(is.na(strata_2), strata_1, strata_2),
         EA_ID = ifelse(is.na(EA_ID_2), EA_ID_1, EA_ID_2)) %>%
  filter(!is.na(EA_ID)) 
  # # refuse sex only waves 1 and 3, and no variation, so taking that one out

svydes <- svydesign(id = ~EA_ID, strata = ~strata, weights =  ~FQweight, data = filter_data, nest = T)
other.outcomes <- c("paidw_12m", "decide_spending_mine", "buy_decision_health")
      
# Empowerment
empower_results <- list()
modellist <- list()
for (i in other.outcomes) {
  print(i)
  model <- svyglm(as.formula(paste0(i,"_2 ~ current_contra_1 + ",i,"_1  + age_2 + school_2 + live_births_2 + urban_2 + wealthquintile_2")), 
                  family = quasibinomial(), design = svydes)
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


# Contraception
model <- svyglm(current_contra_2 ~ current_contra_1 + paidw_12m_1 + decide_spending_mine_1 + buy_decision_health_1 + age_2 + school_2 + live_births_2 + urban_2 + wealthquintile_2, 
                family = quasibinomial(), design = svydes)
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


# Method choice arrays
data.path <- normalizePath(file.path(Sys.getenv("ONEDRIVE"), "DHS"), "/")
data.raw <- with_dir(data.path, {read_dta("KE_2014_DHS_09162022_2324_122388/KEIR72DT/KEIR72FL.DTA")}) 

data <- data.raw %>%
  mutate(wt = v005/1000000,
         age = as.numeric(v012), 
         parity=as.numeric(v220),
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

svydes1 <- svydesign(id = ~v001, strata= ~v023, weights = ~wt, data=data, nest = T)
methods <- as.data.frame(svytable(~method+age_grp+parity, svydes1)) %>%
  group_by(age_grp, parity) %>% mutate(percent = Freq/sum(Freq)) %>% select(-Freq)
# write.csv(methods, "fpsim/locations/kenya/method_mix.csv", row.names = F)

methods %>%
ggplot()+
  geom_line(aes(y = percent, x = method, group = parity, color = parity)) +
  facet_wrap(~age_grp)




