#############################################
# -- Analysis on PMA data
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
library(splines)
library(glmnet)
library(boot)

# -- To get formatted PMA dataset, run PMA_recode in data processing folder

# Coefficients for two economic empowerment scales, household decision making and financial autonomy. From Cardona et al. 2023 PMA analysis
CFA.load <- (c(buy_decision_major = 0.720, buy_decision_daily= 0.776, buy_decision_health = 0.799, buy_decision_clothes = 0.763, 
               savings = 0.747, financial_info = 0.730, financial_goals = 0.750))
# write.csv(CFA.load, "fpsim/locations/kenya/CFA.loadings.csv", row.names = T, col.names = F)

# add these coefficients to dataframe and calculate scores
All_data <- All_data %>%
  mutate(decision.making = buy_decision_major*CFA.load["buy_decision_major"] + buy_decision_daily*CFA.load["buy_decision_daily"] + buy_decision_health*CFA.load["buy_decision_health"] + buy_decision_clothes*CFA.load["buy_decision_clothes"],
         financial.autonomy = savings*CFA.load["savings"] + financial_info*CFA.load["financial_info"] + financial_goals*CFA.load["financial_goals"])

  




# duplicate dataset to increase sample size... so we have a 1-2 time and 2-3 time
data.edit <- All_data %>% # filtered dataset from just wave 2-3 and rename to 1-2 and label time b
  filter(wave != 1) %>% mutate(wave = case_when(wave == 2 ~ 1, wave == 3 ~ 2), time = "b")
filter_data <- All_data %>%
  filter(wave != 3) %>% mutate(time = "a") %>% # filter to timepoint 102 and label a
  bind_rows(data.edit) %>% # add back in other timepoint
  # to wide 
  gather(var, val, -wave, -female_ID, -time) %>% unite("var", c(var, wave)) %>% spread(var, val, convert = T, drop = F, fill = NA) %>%
  mutate(FQweight = ifelse(is.na(FQweight_2), FQweight_1, FQweight_2),
         strata = ifelse(is.na(strata_2), strata_1, strata_2),
         EA_ID = ifelse(is.na(EA_ID_2), EA_ID_1, EA_ID_2)) %>%
  filter(!is.na(EA_ID)) 
  # refuse sex only waves 1 and 3, and no variation, so taking that one out

# create dataset for postpartum and not
filter.data.notpp <- filter_data %>% filter((pp.time_2>6 | is.na(pp.time_2)) & pregnant_2 == 0)
filter.data.pp1 <- filter_data %>% filter(pp.time_2<2)
filter.data.pp6 <- filter_data %>% filter(pp.time_2<6)

# create survey objects
options(survey.lonely.psu="adjust")
svydes.full <- svydesign(id = ~EA_ID, strata = ~strata, weights =  ~FQweight, data = filter_data , nest = T)
svydes <- svydesign(id = ~EA_ID, strata = ~strata, weights =  ~FQweight, data = filter.data.notpp , nest = T)
svydes.pp1 <- svydesign(id = ~EA_ID, strata = ~strata, weights =  ~FQweight, data = filter.data.pp1 , nest = T)
svydes.pp6 <- svydesign(id = ~EA_ID, strata = ~strata, weights =  ~FQweight, data = filter.data.pp6 , nest = T)


      







# -- Empowerment function -- #
emp.outcomes <- c("paidw_12m", 
                  "decide_spending_mine", "buy_decision_health", 
                  "buy_decision_major", "buy_decision_daily", "decide_spending_partner", "buy_decision_clothes",
                  "savings", "financial_info", "financial_goals")
empower_results <- list()
modellist <- list()
for (i in emp.outcomes) {
  print(i)
  model <- svyglm(as.formula(paste0(i,"_2 ~ current_contra_1 + ",i,"_1  + ns(age_2,knots = c(25)) + yrs.edu_2 + live_births_2 + urban_2 + wealthquintile_2")),
                  family = quasibinomial(), 
                  design = svydes.full)
modellist[[i]] <- model
  empower_results[[i]] <- as.data.frame(summary(model)$coefficients) %>% 
    mutate(lhs = i, rhs = rownames(.)) }

for (i in c("decision.making", "financial.autonomy")) {
  print(i)
  model <- svyglm(as.formula(paste0(i,"_2 ~ current_contra_1 + ",i,"_1  + ns(age_2,knots = c(25)) + yrs.edu_2 + live_births_2 + urban_2 + wealthquintile_2")),
                  family = gaussian(), 
                  design = svydes.full)
  modellist[[i]] <- model
  empower_results[[i]] <- as.data.frame(summary(model)$coefficients) %>% 
    mutate(lhs = i, rhs = rownames(.)) }

empower_coef <- bind_rows(empower_results)  %>%
  # rename variables to match model
  mutate(across(c(lhs, rhs), ~gsub("_2", "", gsub("_1", "_0", 
                                                  gsub("yrs.edu","edu_attainment",
                                                       gsub("paidw_12m","paid_employment",
                                                            gsub("decide_spending_mine","decision_wages",
                                                                 gsub("buy_decision_health","decision_health",
                                                                                gsub("live_births", "parity", 
                                                                                     gsub("current_contra", "contraception", .))))))))))

# write.csv(empower_coef, "fpsim/locations/kenya/empower_coef.csv", row.names = F)



# Look at each over age to assess spline#
i = "paidw_12m" # set i here then then run model line above
new.frame = expand.grid(current_contra_1 = c(0,1), paidw_12m_1 = c(0,1), decide_spending_mine_1 = c(0,1), buy_decision_health_1 = c(0,1), age_2 = 15:49, yrs.edu_2 = 0:5, live_births_2 = 0:15, urban_2 = c(0,1), wealthquintile_2 = 1:5)
predicted <- new.frame %>% cbind(as.data.frame(predict(model, new.frame, type = "response"))) # create df of predicted work with predictor values
predicted.age <- predicted %>% group_by(age_2) %>% summarise(fit = mean(response, na.rm = T)) # summarize predicted work by age
data.age <- filter_data %>% group_by(age_2) %>% 
  summarise(work = mean(paidw_12m_2), dec.hers = mean(decide_spending_mine_2, na.rm = T), dec.hlth = mean(buy_decision_health_2, na.rm = T), contra = mean(current_contra_2, na.rm = T)) # summarize data by age
predicted.age %>% left_join(data.age) %>%
  ggplot() +
  geom_point(aes(y = work, x = age_2)) + 
  #geom_point(aes(y = dec.hers, x = age_2)) + 
  #geom_point(aes(y = dec.hlth, x = age_2)) + 
  geom_line(aes(x= age_2, y = fit))






# contraception functions (3 levels of complexity) -- #

# Visualize contraception by age
filter_data %>% group_by(age_2) %>% summarise(contra = mean(current_contra_2, na.rm = T)) %>% # summarize data by age
  ggplot() + geom_point(aes(y = contra, x = age_2)) 

# Contraception simple function
model.simple <- svyglm(current_contra_2 ~ age_grp_2, 
                family = quasibinomial(), 
                design = svydes)
                #design = svydes.pp1)
                #design = svydes.pp6)
contra_coef.simple <- as.data.frame(summary(model.simple)$coefficients) %>% 
  mutate(rhs = gsub("_2", "", rownames(.)))

# write.csv(contra_coef.simple, "fpsim/locations/kenya/contra_coef_simple.csv", row.names = F)
# write.csv(contra_coef.simple, "fpsim/locations/kenya/contra_coef_simple_pp1.csv", row.names = F)
# write.csv(contra_coef.simple, "fpsim/locations/kenya/contra_coef_simple_pp6.csv", row.names = F)

# look at predicted probabilites of contrceptive use for model validation purposes
predicted.p <- predict(model.simple, newdata = data.frame(age_grp_2 = unique(All_data_long$age_grp)), type = "response")




# Contraception mid function (only demographics, no empowerment or history)... no longitudinal or empowerment data needed, could be done with DHS
model.mid <- svyglm(current_contra_2 ~ ns(age_2, knots = c(25,40)) + yrs.edu_2 + live_births_2 + urban_2 + wealthquintile_2, 
                     family = quasibinomial(), 
                     #design = svydes)
                     #design = svydes.pp1)
                     design = svydes.pp6)
contra_coef.mid <- as.data.frame(summary(model.mid)$coefficients) %>% 
  mutate(rhs = rownames(.)) %>%
  mutate(rhs = gsub("_2", "", gsub("live_births", "parity",                                                   
                                   gsub("yrs.edu","edu_attainment",
                                        gsub("current_contra", "contraception", rhs)))))

# write.csv(contra_coef.mid, "fpsim/locations/kenya/contra_coef_mid.csv", row.names = F)
# write.csv(contra_coef.mid, "fpsim/locations/kenya/contra_coef_mid_pp1.csv", row.names = F)
# write.csv(contra_coef.mid, "fpsim/locations/kenya/contra_coef_mid_pp6.csv", row.names = F)





# Contraception full function
# Lasso for full contraception function
# list all predictors and take out missing values
lasso.vars <- c("intent_cat_1", "paidw_12m_1", 
                "decide_spending_mine_1", "buy_decision_health_1", "buy_decision_major_1", "buy_decision_daily_1", "decide_spending_partner_1", "buy_decision_clothes_1",
                "savings_1", "financial_info_1", "financial_goals_1",
                "decision.making_1", "financial.autonomy_1",
                "age_2", "yrs.edu_2", "live_births_2", "urban_2", "wealthquintile_2")
lasso.data <- filter.data.notpp %>%
  select(EA_ID, strata, FQweight, current_contra_2, all_of(lasso.vars)) %>%
  na.omit
svydes.lasso <- svydesign(id = ~EA_ID, strata = ~strata, weights =  ~FQweight, data = lasso.data , nest = T)

# Model with all predictors and all pairwise comparison interactions
design_matrix <- model.matrix(~ (intent_cat_1 + paidw_12m_1 + 
                                   decide_spending_mine_1 + buy_decision_health_1 + buy_decision_major_1 + buy_decision_daily_1 + decide_spending_partner_1 + buy_decision_clothes_1 +
                                   savings_1 + financial_info_1 + financial_goals_1 +
                                   decision.making_1 + financial.autonomy_1 +
                                   ns(age_2, knots = c(25,40)) + yrs.edu_2 + live_births_2 + urban_2 + wealthquintile_2)^2, 
                             data = svydes.lasso$variables)
response <- svydes.lasso$variables$current_contra_2
weights <- weights(svydes.lasso)

# Fit Lasso model
lasso_model <- glmnet(design_matrix, response, family = "binomial", weights = weights, alpha = 1)

# Cross-validate to find the optimal lambda
cv_lasso_model <- cv.glmnet(design_matrix, response, family = "binomial", weights = weights, alpha = 1)
best_lambda <- cv_lasso_model$lambda.min

# Extract the coefficients for the best lambda
lasso_coefs <- coef(lasso_model, s = best_lambda)
print(lasso_coefs)

# Bootstrap for standard errors of coefficients
# Function to fit Lasso model and extract coefficients for a bootstrap sample
lasso_boot <- function(data, indices) {
  boot_data <- data[indices, ]
  boot_design_matrix <- model.matrix(~ (intent_cat_1 + paidw_12m_1 + 
                                          decide_spending_mine_1 + buy_decision_health_1 + buy_decision_major_1 + buy_decision_daily_1 + decide_spending_partner_1 + buy_decision_clothes_1 +
                                          savings_1 + financial_info_1 + financial_goals_1 +
                                          decision.making_1 + financial.autonomy_1 +
                                          ns(age_2, knots = c(25,40)) + yrs.edu_2 + live_births_2 + urban_2 + wealthquintile_2)^2, 
                                     data = boot_data)
  boot_response <- boot_data$current_contra_2
  boot_weights <- weights[indices]
  fit <- glmnet(boot_design_matrix, boot_response, family = "binomial", weights = boot_weights, alpha = 1)
  as.vector(predict(fit, s = best_lambda, type = "coefficients"))
}

# Perform bootstrapping
n_boot <- 1000 # Number of bootstrap samples
set.seed(1008)
boot_results <- boot(lasso.data, lasso_boot, R = n_boot)
boot_coefs <- boot_results$t # Extract coefficients from bootstrap results
coef_se <- apply(boot_coefs, 2, sd) # Calculate standard errors
coef_inclusion_freq <- colMeans(boot_coefs != 0) # Calculate frequency of inclusion
print(coef_se)

# Format and save coefficients
contra_coef <- data.frame(
  rhs = rownames(lasso_coefs)[as.vector(lasso_coefs) != 0],
  Estimate = lasso_coefs[as.vector(lasso_coefs) != 0],
  `Std. Error` = coef_se[as.vector(lasso_coefs) != 0]) %>%
  mutate(rhs = gsub("_2", "", gsub("_1", "_0", 
                                   gsub("yrs.edu","edu_attainment",
                                        gsub("paidw_12m","paid_employment",
                                             gsub("decide_spending_mine","decision_wages",
                                                  gsub("buy_decision_health","decision_health",
                                                       gsub("live_births", "parity", 
                                                            gsub("current_contra", "contraception", rhs)))))))))

# write.csv(contra_coef, "fpsim/locations/kenya/contra_coef.csv", row.names = F)
# write.csv(contra_coef, "fpsim/locations/kenya/contra_coef_pp1.csv", row.names = F)
# write.csv(contra_coef, "fpsim/locations/kenya/contra_coef_pp6.csv", row.names = F)















# -- intent to use contraception function -- #

# visualize intent by age
# Visualize contraception by age
filter_data %>% group_by(age_2) %>% summarise(intent = mean(intent_contra_2, na.rm = T)) %>% # summarize data by age
  ggplot() + geom_point(aes(y = intent, x = age_2)) 

# Model
model.intent <- svyglm(intent_contra_2 ~ fertility_intent_2 + ns(age_2, knots = c(25,40)) + yrs.edu_2 + live_births_2 + urban_2 + wealthquintile_2, 
                     family = quasibinomial(), 
                     design = svydes.full)
intent_coef <- as.data.frame(summary(model.intent)$coefficients) %>% 
  mutate(rhs = rownames(.)) %>%
  mutate(rhs = gsub("_2", "", gsub("yrs.edu","edu_attainment",
                                   gsub("live_births", "parity", rhs))))

# write.csv(intent_coef, "fpsim/locations/kenya/intent_coef.csv", row.names = F)












