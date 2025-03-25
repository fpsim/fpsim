##############################################################################
# -- Analysis of time on contraceptive method using DHS calendar data

library(survival)
library(survey)
library(ggplot2)
library(ciTools)
library(tidyverse)
library(survminer)
library(flexsurv)
library(withr)  
library(haven)  
library(tidyverse)
library(data.table)
library(mvtnorm)
library(extrafont)
library(data.table)
rm(list=ls())

###############################################################################
## - Data cleaning

# Create var to the data path
data.path <- normalizePath(file.path(Sys.getenv("ONEDRIVE"), "DHS/IR_all"), "/")


# Get list of all the county data files to be included
temp = with_dir(data.path, {list.files()}) #list all files in the working directory with extension
temp = temp[temp != "country-region.csv"]  
temp = temp[temp != "IAIR7DDT"]            # Remove India (at least for now) because it's huge
# temp = temp[1:3] #### OPTIONAL use this to make a smaller dataset while getting the code to work
# temp = "SNIR8BDT" # Senegal only

# List of country code and names and regions
countries <- with_dir(data.path, {read.csv("country-region.csv")})


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
             "v218", #living children
             "v190", #wealth index
             "vcal_1", #contraceptive calendar
             "v025") #urban/rural

# Get all country datasets into one data frame
datalist = list()
for (i in 1:length(temp)){
  dat <- with_dir(data.path, {read_dta(paste0(temp[i], "/", substr(temp[i], 1, 6), "FL.DTA"), col_select = any_of(varlist))})
  datalist[[i]] <- dat
}
data.raw <- bind_rows(datalist)
rm(dat, datalist)

# -- Survival analysis data organization -- #
data.surv <- data.raw %>%
  mutate(cc = str_sub(v000,1,2),
         wt = v005/1000000,
         urban = ifelse(v025 == 1, 1, 0),
         live_births = v218,
         wealth_index = v190,
         edu_years = ifelse(v133 == 99, NA, v133),
         age = as.numeric(v012)) %>%                                                                        # convert age to numeric
  left_join(countries) %>%                                                                                  # Add in country name and region
  filter(age < 50 & !is.na(vcal_1) & vcal_1 != "") %>%                                                      # Keep only women of reproductive age with calendar data
  select(wt, age, live_births, edu_years, wealth_index, urban, vcal_1, v000, v021, v022, Country) %>%
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
                           age_grp = cut(age, c(0,18,20,25,35,50)),                                            # define age groups
                           age_grp_fact = factor(age_grp,levels=c('(20,25]','(0,18]','(18,20]','(25,35]','(35,50]')), # factor age group
                           new.method = ifelse(method != lag(method,1) | month == 1,1,0),                   # Indicator for a month where a new method is started (the first month of a switch/discontinue), with methods including pregnant, none, etc
                           method.month = rowid(method, cumsum(new.method == 1)),                           # Months on individual method    
                           on_method = ifelse(method %in% c("1", "2", "3", "5", "9", "N", "W", "M"), 1, 0), # Identify relevant methods
                           cumulative_method_months = cumsum(on_method),                                    # Cumulative months for relevant methods
                           methods_so_far = purrr::accumulate(method, ~ unique(c(.x, .y))),                 # List of methods up to current month
                           methods_so_far = purrr::map(methods_so_far, ~ .x[.x %in% c("1", "2", "3", "5", "9", "N", "W", "M")]),
                           unique_methods_count = sapply(methods_so_far, length),                           # count unique methods
                           births = cumsum(method == "B"),                                                  # count births
                           parity = ifelse(live_births - births<0, 0, live_births - births),
                           expected_edu = round(age,0) - 5,
                           edu = ifelse(edu_years<expected_edu, edu_years, expected_edu)) %>%
  filter(lead(method.month, 1) == 1) %>%                                                                    # Keep only last row of each method with number of months on that method
  mutate(discontinued = ifelse(month == max(month), 0, 1)) %>%                                              # Indicator for observed switch (1) or censored (0)
  filter(month != min(month)) %>%                                                                           # Take out first method because we don't know when started
  filter(method != "P" & method != "B" & method != "T" & method != "L" & method != "L") %>%                 # Don't need to look at these methods
  mutate(method = factor(method, levels = c("0",    "1",    "2",   "3",          "5",      "9",          "N",       "W",          "M"),
                         labels = c("None", "Pill", "IUD", "Injectable", "Condom", "Withdrawal", "Implant", "Other.trad", "Other.mod"))) %>%
  select(Country, obs, v000, v021, v022, wt, age, age_grp, age_grp_fact, parity, edu, wealth_index, urban, cumulative_method_months, unique_methods_count, method, month, method.month, discontinued) 
# write.csv(data.surv, "C:/Users/maritazi/OneDrive - Bill & Melinda Gates Foundation/WRICH/Contraceptive choices/timeonmethod_fpsim.csv", row.names = F)
# data.surv <- read.csv("C:/Users/maritazi/OneDrive - Bill & Melinda Gates Foundation/WRICH/Contraceptive choices/timeonmethod_fpsim.csv")

# data set to use
dat <- data.surv
age_grp_labels <- as.character(unique(dat$age_grp_fact)) 




###############################################################################
# -- Accelerated failure time model for each country and method

results <- list()
# loop through each country
for (c in unique(dat$Country)) {
  df_country <- dat[dat$Country == c, ]
  results[[c]] <- list()
  
  # loop through each method
  for (m in c("None", "Pill", "IUD", "Injectable", "Condom", "Withdrawal", "Implant", "Other.trad", "Other.mod")) {
    df_method <- df_country[!is.na(df_country$method) & df_country$method == m, ]
    if (nrow(df_method) == 0) next
    
    lowest_aic <- Inf
    best_model <- NULL
    best_dist <- NULL
    
    # loop through each distribution
    for (d in c("exponential", "weibull", "llogis", "lnorm", "gamma")) {
      df_method <- df_method[order(df_method$obs), ]
      df.first.switch <- df_method[!duplicated(df_method$obs), ] # use only first switch
      aft <- tryCatch(
        #flexsurvreg(Surv(method.month, discontinued) ~ age_grp_fact + parity + edu + wealth_index + urban, data = df.first.switch, dist = d),
        flexsurvreg(Surv(method.month, discontinued) ~ age_grp_fact, data = df.first.switch, dist = d),
        error = function(e) NULL ) 
      
      # check if best fit
      if (!is.null(aft) && aft$AIC < lowest_aic) {
        lowest_aic <- aft$AIC
        best_model <- aft
        best_dist <- d}
    }
    
    # save best fit model
    if (!is.null(best_model)) {
      results[[c]][[m]] <- list(
        best_distribution = best_dist,
        lowest_aic = lowest_aic,
        model = best_model )}
  }
}

saveRDS(results, file = "C:/Users/maritazi/Documents/Projects/fp_data_analysis/Results/results_list_fpsim.rds") # model with only age group as a predictor
# results <- readRDS("C:/Users/maritazi/Documents/Projects/fp_data_analysis/Results/results_list_fpsim.rds") # model with only age group as a predictor


# --------------- Take out estimates and save
coef_res <- list()
# Iterate through the results list
for (c in names(results)) {
  for (m in names(results[[c]])) {
    model_info <- results[[c]][[m]]
    
    if (!is.null(model_info$model)) {
      model_summary <- model_info$model
      
      # Extract estimates, SEs, CIs
      res <- as.data.frame(model_summary$res.t)
      output <- res %>%
        mutate(coef = rownames(res),
               country = c,
               method = m,
               functionform = model_info$best_distribution)
      
      coef_res <- rbind(coef_res, output)}}}


coef_fpsim <- coef_res %>%
  mutate(ifelse(functionform == "lnorm", "lognormal", functionform)) %>%
  select(estimate = est, coef, se, method, functionform, country) 

write.csv(filter(coef_fpsim, country == "Kenya"), "C:/Users/maritazi/Documents/Projects/fpsim/fpsim/locations/kenya/data/method_time_coefficients.csv", row.names = F)
write.csv(filter(coef_fpsim, country == "Senegal"), "C:/Users/maritazi/Documents/Projects/fpsim/fpsim/locations/senegal/data/method_time_coefficients.csv", row.names = F)
write.csv(filter(coef_fpsim, country == "Ethiopia"), "C:/Users/maritazi/Documents/Projects/fpsim/fpsim/locations/ethiopia/data/method_time_coefficients.csv", row.names = F)


# Check PDFs
time_range <- seq(1, 100, by = 1)
values <- NULL
for (c in setdiff(names(results), "Comoros")) {
  for (m in names(results[[c]])) {
    for (age_group in c("age_grp_fact(0,18]","age_grp_fact(18,20]","age_grp_fact(20,25]", "age_grp_fact(25,35]","age_grp_fact(35,50]")) {
      coefs <-  results[[c]][[m]]$model$coefficients
      dist_name <- results[[c]][[m]]$best_distribution
      if (dist_name == "exponential") {
        val <- data.frame(pdf = dexp(time_range, rate = exp(coefs[1]+ coalesce(coefs[age_group], 0))))
      } else if (dist_name == "weibull") {
        val <- data.frame(pdf = dweibull(time_range, shape = exp(coefs[1]), scale = exp(coefs[2]+ coalesce(coefs[age_group], 0))))
      } else if (dist_name == "llogis") {
        val <- data.frame(pdf = dllogis(time_range, shape = exp(coefs[1]), scale = exp(coefs[2]+ coalesce(coefs[age_group], 0))))
      } else if (dist_name == "lnorm") {
        val <- data.frame(pdf = dlnorm(time_range, meanlog = (coefs[1]+ coalesce(coefs[age_group], 0)), sdlog = exp(coefs[2]))) 
      } else if (dist_name == "gompertz") {
        val <- data.frame(pdf = dgompertz(time_range, shape = coefs[1], rate = exp(coefs[2]+ coalesce(coefs[age_group], 0))))
      } else if (dist_name == "genf") {
        val <- data.frame(pdf = dgenf(time_range, mu = coefs[1] + coalesce(coefs[age_group], 0), sigma = (coefs[2] ), Q = coefs[3], P = exp(coefs[4])))
      } else if (dist_name == "gamma") {
        val <- data.frame(pdf = dgamma(time_range, shape = exp(coefs[1]), scale = 1/exp(coefs[2]+ coalesce(coefs[age_group], 0))))
      }
      val <- val %>% mutate(country = c, method = m, age = age_group, time = time_range, dist = dist_name)
      values <- rbind(values, val)    }
  }}


# Plot PDFs
values %>%
  filter(country %in% c("Ethiopia", "Kenya", "Senegal")) %>%
  ggplot() +
  geom_line(aes(x = time, y = pdf, color = age)) +
  facet_grid(method ~ country, scales = "free_y") +
  labs(title = "Probability Density Functions (PDFs) by Distribution and Age Group",
       x = "Time (months)",
       y = "Density",
       color = "Age Group") +
  theme_minimal()


###### Get PDFs using the package directly so that we can compare what we are doing and make sure it's right
library(mvtnorm)
source("summary.flexsurvreg.R")
source("flexsurvreg.R")
values2 <- NULL
for (c in c("Kenya","Ethiopia","Senegal")) {
  for (m in names(results[[c]])) {
    x = results[[c]][[m]]$model
    newdata = data.frame(age_grp_fact = x$covdata$xlev)
    dist_name <- results[[c]][[m]]$best_distribution
    X <- newdata_to_X(x = x, newdata = newdata)
    args <- xt_to_fnargs(x = x, X = X, t = time_range, quantiles, start = 0, type = "survival", cross = T)
    if (dist_name == "exponential") {
      val <- x$dfns$d(args$t, rate = args$rate) 
    } else if (dist_name %in% c("weibull", "llogis")) {
      val <- x$dfns$d(args$t, scale = args$scale, shape = args$shape) 
    } else if (dist_name == "lnorm") {
      val <- x$dfns$d(args$t, meanlog = args$meanlog, sdlog = args$sdlog)
    } else if (dist_name == "genf") {
      val <- x$dfns$d(args$t, mu = args$mu, sigma = args$sigma, Q = args$Q, P = args$P) 
    } else if (dist_name %in% c("gamma", "gompertz")) {
      val <- x$dfns$d(args$t, rate = args$rate, shape = args$shape) 
    }
    val <- data.frame(pdf2 = val, time = rep(time_range, nrow(newdata)), age = rep(unique(newdata$age_grp_fact) , each = 100), dist = dist_name, country = c, method = m)
    values2 <- rbind(values2, val)
  }}


# plot manual and package together to verify that manual is right
values %>%
  left_join(values2 %>% mutate(age = paste0("age_grp_fact",age))) %>%
  filter(country %in% c("Ethiopia")) %>%
  ggplot() +
  geom_line(aes(x = time, y = pdf, color = age)) +
  geom_line(aes(x = time, y = pdf2, color = age), linetype = "dashed") +
  facet_wrap(~method, scale = "free") +
  #facet_grid(method ~ country, scales = "free_y") +
  labs(title = "Probability Density Functions (PDFs) by Distribution and Age Group",
       x = "Time (months)",
       y = "Density",
       color = "Age Group") +
  theme_minimal()


# compare to FPsim
process_fp_sim <- function(file_path, country_name) {
  fp.sim <- read.csv(file_path, header = FALSE, stringsAsFactors = FALSE)
  
  methods <- fp.sim[1, ]
  ages <- fp.sim[2, ]
  fp.sim <- fp.sim[-c(1,2), ]
  colnames(fp.sim) <- paste0(methods, "_", ages)
  
  fp.sim_long <- fp.sim %>%
    pivot_longer(cols = -Method_Age, names_to = "M_A", values_to = "pdf") %>%
    separate(M_A, into = c("method", "age_grp"), sep = "_", extra = "merge") %>%
    mutate(pdf = as.numeric(pdf),
           country = country_name,
           time = as.numeric(Method_Age),
           method = recode(method,
                           "cond" = "Condom",
                           "impl" = "Implant",
                           "inj" = "Injectable",
                           "iud" = "IUD",
                           "none" = "None",
                           "othmod" = "Other.mod",
                           "othtrad" = "Other.trad",
                           "pill" = "Pill",
                           "wdraw" = "Withdrawal"),
           age = recode(age_grp,
                        "18" = "age_grp_fact(0,18]",
                        "20" = "age_grp_fact(18,20]",
                        "25" = "age_grp_fact(20,25]",
                        "35" = "age_grp_fact(25,35]",
                        "50" = "age_grp_fact(35,50]")) %>%
  select(-Method_Age,-age_grp)

return(fp.sim_long)
}

# File paths
senegal_path <- "C:/Users/maritazi/downloads/duration_dists_senegal.csv"
kenya_path <- "C:/Users/maritazi/downloads/duration_dists_kenya.csv"
ethiopia_path <- "C:/Users/maritazi/downloads/duration_dists_ethiopia.csv"

# Process each dataset
senegal_data <- process_fp_sim(senegal_path, "Senegal")
kenya_data <- process_fp_sim(kenya_path, "Kenya")
ethiopia_data <- process_fp_sim(ethiopia_path, "Ethiopia")

# Combine all datasets
all_countries_data <- bind_rows(senegal_data, kenya_data, ethiopia_data)

# plot manual and package together to verify that FPsim is right
values %>% mutate(source = "R") %>%
  bind_rows(all_countries_data %>% mutate(source = "Python")) %>%
  filter(country %in% c("Ethiopia", "Kenya", "Senegal")) %>%
  filter(time>0 & time<100) %>%
  ggplot() +
  geom_line(aes(x = time, y = pdf, color = age, linetype = source)) +
  facet_grid(method ~ country, scales = "free_y") +
  labs(title = "Probability Density Functions (PDFs) by Distribution and Age Group",
       x = "Time (months)",
       y = "Density",
       color = "Age Group") +
  theme_minimal()
