library(survival)
library(data.table)
library(tidyverse)
library(survey)
library(ggplot2)
library(ciTools)
library(tidyverse)
library(survminer)
library(flexsurv)
library(withr)  
library(haven)  

data.path <- normalizePath(file.path(Sys.getenv("ONEDRIVE"), "DHS/IR_all"), "/")

# List of variables we want to use
varlist <- c("v000", #country-survey code
             "v001", #cluster
             "v005", #ind weights
             "v008", #date of interview (CMC)
             "v011", #respondent's age (CMC)
             "v012", #respondent age
             "v021", #PSU
             "v022", #strata
             "v101", #region
             "v130",
             "v133", #education in years
             "v212", #age at first birth
             "v220", #parity
             "vcal_1") #contraceptive calendar
data.raw <- with_dir(data.path, {read_dta(paste0("KEIR72DT/KEIR72FL.DTA"), col_select = any_of(varlist))})

# -- Survival analysis version -- #
data.surv <- data.raw %>%
  mutate(cc = str_sub(v000,1,2),
         wt = v005/1000000,
         age = as.numeric(v012)) %>%                                                                        # convert age to numeric
  filter(age < 50 & !is.na(vcal_1) & vcal_1 != "") %>%                                                      # Keep only women of reproductive age with calendar data
  select(wt, age, vcal_1, v000, v021, v022) %>%
  mutate(cal = str_squish(vcal_1),                                                                          # remove whitespace
         cal = gsub("8","W",cal),                                                                           # recode contraceptive methods, Periodic abstinence/rhythm to other traditional
         cal = gsub("S","M",cal), cal = gsub("4", "M", cal), cal = gsub("D", "M", cal), cal = gsub("C", "M", cal),  # recode contraceptive methods to other modern, std days, diaphragm x2, female condom
         cal = gsub("E", "M", cal), cal = gsub("F", "M", cal), cal = gsub("7", "M", cal),                   # recode NUR-ISTERATE from SA to injectable, per https://userforum.dhsprogram.com/index.php?t=msg&goto=25256&&srch=vcal#msg_25256  
         cal = gsub("I","3",cal),                                                                           # recode contraceptive methods, Periodic abstinence/rhythm to other traditionalhttp://127.0.0.1:42617/graphics/plot_zoom_png?width=2560&height=1378
         cal_clean = sapply(strsplit(cal, split = ""),  function(str) {paste(rev(str), collapse = "")}),    # reverse string so oldest is first
         pattern_split = strsplit(cal_clean,""),                                                            # split into individual characters
         method = lapply(pattern_split, function(m) m[1:80]),                                               # codes for months
         obs = 1:nrow(.)) %>%                                                                               # create obs variable to identify the individual
  unnest(c(method)) %>%                                                                                     # create long format
  group_by(obs) %>% mutate(month = row_number(),                                                            # create variable for month from beginning of calendar
                           age = age - (79-month)/12,                                                       # adjust for age in calendar
                           age_grp = cut(age, c(0,18,20,25,35,50)),                                         # define age groups
                           new.method = ifelse(method != lag(method,1) | month == 1,1,0),                   # Indicator for a month where a new method is started (the first month of a switch/discontinue), with methods including pregnant, none, etc
                           method.month = rowid(method, cumsum(new.method == 1))) %>%                       # Months on individual method
  filter(lead(method.month, 1) == 1) %>%                                                                    # Keep only last row of each method with number of months on that method
  mutate(discontinued = ifelse(month == max(month), 0, 1)) %>%                                                    # Indicator for observed switch (1) or censored (0)
  filter(month != min(month)) %>%                                                                           # Take out first method because we don't know when started
  filter(method != "P" & method != "B" & method != "T" & method != "L" & method != "L") %>%                 # Don't need to look at these methods
  mutate(method = factor(method, levels = c("0",    "1",    "2",   "3",          "5",      "9",          "N",       "W",          "M"),
                         labels = c("None", "Pill", "IUD", "Injectable", "Condom", "Withdrawal", "Implant", "Other.trad", "Other.mod"))) %>%
  select(obs, v000, v021, v022, wt, age, age_grp, method, month, method.month, discontinued) 




# -- accelerated failure time models -- #
dat <- data.surv
dat$age_grp_fact <- factor(dat$age_grp,levels=c('(20,25]','(0,18]','(18,20]','(25,35]','(35,50]'))


########## Injectables ################
inj <- dat[dat$method %in% 'Injectable',] #subset to method group
length(inj$obs); length(unique(inj$obs))   ## Obs vs unique women
# removing multiple events for now since we have all sorts of correlation going on already
inj <- inj[order(inj$obs),]
inj.first.switch <- inj[!duplicated(inj$obs), ]
inj.dup.switch <- inj[duplicated(inj$obs) |duplicated(inj$obs,fromLast = T), ]
# AFT model, gamma had best fit
inj.aft.gamma <- flexsurvreg(Surv(method.month,discontinued)~age_grp_fact,data=inj.first.switch,dist='gamma')     
inj.aft.coef <- inj.aft.gamma$coefficients
pred_df <- function(x,name){
  new_dat <- data.frame('age_grp_fact'=levels(inj.first.switch$age_grp_fact))
  tmp <- predict(x,newdata=new_dat,times = seq(1,60,1),type='survival',conf.int = TRUE)
  pred_dat <- data.frame(rbind(
    tmp$.pred[[1]] %>% mutate(.age=levels(inj.first.switch$age_grp_fact)[1]),
    tmp$.pred[[2]] %>% mutate(.age=levels(inj.first.switch$age_grp_fact)[2]),
    tmp$.pred[[3]] %>% mutate(.age=levels(inj.first.switch$age_grp_fact)[3]),
    tmp$.pred[[4]] %>% mutate(.age=levels(inj.first.switch$age_grp_fact)[4]),
    tmp$.pred[[5]] %>% mutate(.age=levels(inj.first.switch$age_grp_fact)[5])))
  
  pred_dat <- pred_dat %>% select(.age,.time,.pred_survival,.pred_lower,.pred_upper)
  #names(pred_dat)[3:5] <- paste0(name, names(pred_dat)[3:5])
  return(pred_dat)
}
inj_predict <- pred_df(x=inj.aft.gamma,name='gamma') 


########## None ################
none <- dat[dat$method %in% 'None',] 
none <- none[order(none$obs),]
none.first.switch <- none[!duplicated(none$obs), ]
none.dup.switch <- none[duplicated(none$obs) |duplicated(none$obs,fromLast = T), ]
none.aft.logN <- flexsurvreg(Surv(method.month,discontinued)~age_grp_fact,data=none.first.switch,dist='lnorm')
none.aft.coef <- none.aft.logN$coefficients
none_predict <- pred_df(x=none.aft.logN,name='logN') 


########## Pill ################
pill <- dat[dat$method %in% 'Pill',] 
pill <- pill[order(pill$obs),]
pill.first.switch <- pill[!duplicated(pill$obs), ]
pill.aft.logN <- flexsurvreg(Surv(method.month,discontinued)~age_grp_fact,data=pill.first.switch,dist='lnorm')     
pill.aft.coef <- pill.aft.logN$coefficients
pill_predict <- pred_df(x=pill.aft.logN,name='logN') 

########## IUD ################
IUD <- dat[dat$method %in% 'IUD',] 
IUD <- IUD[order(IUD$obs),]
IUD.first.switch <- IUD[!duplicated(IUD$obs), ]
IUD.aft.ext <- flexsurvreg(Surv(method.month,discontinued)~age_grp_fact,data=IUD.first.switch,dist='gompertz')
IUD.aft.coef <- IUD.aft.ext$coefficients
IUD_predict <- pred_df(x=IUD.aft.ext,name='gompertz') 

########## Condom ################
Condom <- dat[dat$method %in% 'Condom',] 
Condom <- Condom[order(Condom$obs),]
Condom.first.switch <- Condom[!duplicated(Condom$obs), ]
Condom.aft.logN <- flexsurvreg(Surv(method.month,discontinued)~age_grp_fact,data=Condom.first.switch,dist='lnorm') 
Condom.aft.coef <- Condom.aft.logN$coefficients
Condom_predict <- pred_df(x=Condom.aft.logN,name='logN') 

########## Withdrawal ################
Withdrawal <- dat[dat$method %in% 'Withdrawal',] 
Withdrawal <- Withdrawal[order(Withdrawal$obs),]
Withdrawal.first.switch <- Withdrawal[!duplicated(Withdrawal$obs), ]
Withdrawal.aft.loglog <- flexsurvreg(Surv(method.month,discontinued)~age_grp_fact,data=Withdrawal.first.switch,dist='llogis')
Withdrawal.aft.coef <- Withdrawal.aft.loglog$coefficients
Withdrawal_predict <- pred_df(x=Withdrawal.aft.loglog,name='loglog') 

########## Implant ################
Implant <- dat[dat$method %in% 'Implant',] 
Implant <- Implant[order(Implant$obs),]
Implant.first.switch <- Implant[!duplicated(Implant$obs), ]
Implant.aft.ext <- flexsurvreg(Surv(method.month,discontinued)~age_grp_fact,data=Implant.first.switch,dist='gompertz')
Implant.aft.coef <- Implant.aft.ext$coefficients
Implant_predict <- pred_df(x=Implant.aft.ext,name='gompertz') 

########## Other.trad ################
Other.trad <- dat[dat$method %in% 'Other.trad',] 
Other.trad <- Other.trad[order(Other.trad$obs),]
Other.trad.first.switch <- Other.trad[!duplicated(Other.trad$obs), ]
Other.trad.aft.gamma <- flexsurvreg(Surv(method.month,discontinued)~age_grp_fact,data=Other.trad.first.switch,dist='gamma')  
Other.trad.aft.coef <- Other.trad.aft.gamma$coefficients
Other.trad_predict <- pred_df(x=Other.trad.aft.gamma,name='gamma') 

########## Other.mod ################
Other.mod <- dat[dat$method %in% 'Other.mod',] 
Other.mod <- Other.mod[order(Other.mod$obs),]
Other.mod.first.switch <- Other.mod[!duplicated(Other.mod$obs), ]
Other.mod.aft.logN <- flexsurvreg(Surv(method.month,discontinued)~age_grp_fact,data=Other.mod.first.switch,dist='lnorm') 
Other.mod.aft.coef <- Other.mod.aft.logN$coefficients
Other.mod_predict <- pred_df(x=Other.mod.aft.logN,name='logN') 



# Combine all coefficients
method_time_coefficients <- as.data.frame(inj.aft.coef) %>%
  cbind(as.data.frame(none.aft.coef)) %>%
  cbind(as.data.frame(Condom.aft.coef)) %>%
  cbind(as.data.frame(Implant.aft.coef)) %>%
  cbind(as.data.frame(IUD.aft.coef)) %>%
  cbind(as.data.frame(pill.aft.coef)) %>%
  cbind(as.data.frame(Withdrawal.aft.coef)) %>%
  cbind(as.data.frame(Other.trad.aft.coef)) %>%
  cbind(as.data.frame(Other.mod.aft.coef)) 
write.csv(method_time_coefficients, "C:/Users/maritazi/Documents/Projects/fpsim/fpsim/locations/Kenya/method_time_coefficients.csv", row.names = T)

# combine all predictions
method_time_predictions <- inj_predict %>% mutate(method = "Injectable") %>%
  rbind(none_predict %>% mutate(method = "None")) %>%
  rbind(pill_predict %>% mutate(method = "Pill")) %>%
  rbind(IUD_predict %>% mutate(method = "IUD")) %>%
  rbind(Condom_predict %>% mutate(method = "Condom")) %>%
  rbind(Withdrawal_predict %>% mutate(method = "Withdrawal")) %>%
  rbind(Implant_predict %>% mutate(method = "Implant")) %>%
  rbind(Other.trad_predict %>% mutate(method = "Other.trad")) %>%
  rbind(Other.mod_predict %>% mutate(method = "Other.mod")) 
write.csv(method_time_predictions, "C:/Users/maritazi/Documents/Projects/fpsim/fpsim/locations/Kenya/method_time_predictions.csv", row.names = F)
