################################################################
# Calculate sexual activity
# DHS data
# Percent of women who have been active in past 4 weeks among those ever active
################################################################

rm(list=ls())

# -- Libraries -- #
library(tidyverse)      
library(withr)  
library(haven)   
library(survey)

# -- DHS data -- #
dhs.data <- with_dir(normalizePath(file.path(Sys.getenv("ONEDRIVE"), "DHS/IR_all"), "/"),                        # path to DHS data
                      {read_dta("KEIR72DT/KEIR72FL.DTA",                                                         # data file
                                col_select = c("v005", "v021", "v023", "v012", "v536"))}) %>%                    # variables to include
  mutate(active = ifelse(is.na(v536) | v536 == 0, NA, ifelse(v536 == 1, 1, 0)),                                  # sexually active within last 4 weeks, only for ever active
         age = cut(v012, breaks = seq(0, 50, by = 5), right = F, labels = seq(0, 45, by = 5)),                   # 5 year age bins
         wt = v005/1000000)                                                                                      # calculate individual weight 

# Calculate weighted data table 
active <- as.data.frame(svytable(~ active + age, svydesign(id=~v021, weights=~wt, strata=~v023, data =dhs.data))) %>%
  group_by(age) %>%
           mutate(probs = Freq / sum(Freq) * 100, probs = replace_na(probs, 0)) %>%
  filter(active == 1) %>%
  select(age,probs)

# add a row for age 50
active <- bind_rows(active, mutate(tail(active, 1), age = "50"))

# save csv
write.csv(spacing, "fpsim/locations/kenya/sexually_active.csv")
