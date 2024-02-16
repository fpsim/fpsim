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

filter_data <- All_data_wide_clpm %>%
  filter(!is.na(EA_ID)) %>%
  # refuse sex only waves 1 and 3, so set wave 2 = wave 1 but remember that coefficient is for 2 years prior
  mutate(wge_sex_eff_tell_no_2 = wge_sex_eff_tell_no_3)

svydes <- svydesign(id = ~EA_ID, strata = ~strata, weights =  ~FQweight, data = filter_data, nest = T)
other.outcomes <- c("paidw_12m", "decide_spending_mine", "buy_decision_health", "wge_sex_eff_tell_no")
      
# Empowerment
empower_results <- list()
modellist <- list()
for (i in other.outcomes) {
  print(i)
  model <- svyglm(as.formula(paste0(i,"_3 ~ current_contra_2 + ",i,"_2  + age_3 + school_3 + live_births_3 + urban_3 + wealthquintile_3")), 
                  family = quasibinomial(), design = svydes)
  modellist[[i]] <- model
  empower_results[[i]] <- as.data.frame(summary(model)$coefficients) %>% 
    mutate(lhs = i, rhs = rownames(.)) }
empower_coef <- bind_rows(empower_results)  %>%
  # couldn't converge for sexual autonomy so replace
  filter(lhs != "wge_sex_eff_tell_no") %>%
  bind_rows(data.frame(lhs = "wge_sex_eff_tell_no", rhs = unique(empower_results$wge_sex_eff_tell_no$rhs)) %>% mutate(Estimate = ifelse(rhs == "wge_sex_eff_tell_no_2", 1, 0))) %>%
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

write.csv(empower_coef, "fpsim/locations/kenya/empower_coef.csv", row.names = F)


# Contraception
model <- svyglm(current_contra_3 ~ current_contra_2 + paidw_12m_2 + decide_spending_mine_2 + buy_decision_health_2 + wge_sex_eff_tell_no_2 + age_3 + school_3 + live_births_3 + urban_3 + wealthquintile_3, 
                family = quasibinomial(), design = svydes)
contra_coef <- as.data.frame(summary(model)$coefficients) %>% 
  mutate(rhs = rownames(.)) %>%
  mutate(rhs = gsub("_3", "", gsub("_2", "_0", 
                                   gsub("school","edu_attainment",
                                        gsub("paidw_12m","paid_employment",
                                             gsub("decide_spending_mine","decision_wages",
                                                  gsub("buy_decision_health","decision_health",
                                                       gsub("wge_sex_eff_tell_no","sexual_autonomy",
                                                            gsub("married","partnered",
                                                                 gsub("live_births", "parity", 
                                                                      gsub("current_contra", "contraception", rhs)))))))))))

write.csv(contra_coef, "fpsim/locations/kenya/contra_coef.csv", row.names = F)


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
                            T ~ v312),
                         levels = c(1,2,3,5,6,8,9,10,11,17),
                         labels = c("Pill", "IUD", "Injectable", "Condom", "F.sterilization", "Abstinence", "Withdrawal", "Other.trad", "Implant", "Other.mod"))) 

svydes1 <- svydesign(id = ~v001, strata= ~v023, weights = ~wt, data=data, nest = T)
methods <- as.data.frame(svytable(~method+age_grp+parity, svydes1)) %>%
  group_by(age_grp, parity) %>% mutate(percent = Freq/sum(Freq)) %>% select(-Freq)
write.csv(methods, "fpsim/locations/kenya/method_mix.csv", row.names = F)

methods %>%
ggplot()+
  geom_line(aes(y = percent, x = method, group = parity, color = parity)) +
  facet_wrap(~age_grp)




