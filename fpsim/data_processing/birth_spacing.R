################################################################
# Calculate birth spacing intervals for calibration
# Using most recent DHS data
# Kenya
# January 2023
################################################################

rm(list=ls())

# -- Libraries -- #
library(tidyverse)      
library(withr)  
library(haven)   
library(survey)

# -- DHS data -- #
# Source https://dhsprogram.com/data/dataset_admin/index.cfm
# recode https://www.dhsprogram.com/publications/publication-dhsg4-dhs-questionnaires-and-manuals.cfm
dhs.data <- with_dir(normalizePath(file.path(Sys.getenv("ONEDRIVE"), "DHS/IR_all"), "/"),                        # read individual recode data
                      {read_dta("KEIR72DT/KEIR72FL.DTA", col_select = c("v005", "v021", "v023", "v102", "caseid",         # weight, individual id, urban/rural
                                                                        starts_with("b0"),                       # multiples
                                                                        starts_with("b11")))}) %>%               # preceding birth interval
  gather(var, val, -v005, -v021, -v023, -v102, -caseid) %>% separate(var, c("var", "num"), sep = "_") %>% spread(var, val) %>% # create long format with a row for each birth
  filter(b0<2) %>%                                                                                               # remove multips
  mutate(space_mo = b11, wt = v005/1000000,                                                                      # name variable for birth spacing interval (in months), calculate individual weight
         urban_rural = factor(v102, levels = c(1,2), labels = c("urban", "rural")))                              # calculate individual weight 


# Calculate weighted data table of number of births (Freq) in each individual birth spacing (by month) category
spacing <- as.data.frame(svytable(~ space_mo + urban_rural, svydesign(id=~v021, weights=~wt, strata=~v023, data =dhs.data)))
write.csv(spacing, "locations/kenya/birth_spacing_dhs.csv", row.names = F)
