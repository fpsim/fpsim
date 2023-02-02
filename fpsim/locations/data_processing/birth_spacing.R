################################################################
# Calculate birth spacing intervals for Kenya calibration
# January 2023
################################################################

rm(list=ls())

# -- Libraries -- #
library(tidyverse)      
library(withr)  
library(haven)   

# -- DHS data -- #
# Source https://dhsprogram.com/data/dataset_admin/index.cfm
# recode https://www.dhsprogram.com/publications/publication-dhsg4-dhs-questionnaires-and-manuals.cfm
dhs.data <- with_dir(normalizePath(file.path(Sys.getenv("ONEDRIVE"), "DHS/IR_all"), "/"), 
                      {read_dta("KEIR72DT/KEIR72FL.DTA", col_select = c("v005", "v007", "caseid", "v102", "v011",                  # read individual recode data
                                                                        starts_with("bord"), starts_with("b3"), starts_with("b11")))}) %>% # birth order, mother cmc at birth, receding birth interval
  gather(var, val, -v005, -caseid, -v102, -v007, -v011) %>% separate(var, c("var", "num"), sep = "_") %>% spread(var, val) %>%     # create long format with a row for each birth
  filter(!is.na(b3)) %>%                                                                                                           # keep only birth with a subsequent birth and hence birth interval
  mutate(birth_spacing = b3-v011, space_cat = cut(b11, c(0, 12, 24, 48, 300)), birth_order = bord-1,                               # calculate age of mother (in months) at each birth, change birth order to start at 0 instead of 1
         wt = v005/1000000, urban_rural = factor(v102, levels = c(1,2), labels = c("urban", "rural")), source = "DHS") %>%         # calculate individual weight and recode urban rural
  select(wt, source, year = v007, caseid, birth_spacing, space_cat, birth_order, urban_rural)                                      # keep only relevant variables

dhs.results <- dhs.data %>%
  filter(!is.na(space_cat)) %>%
  mutate(wt_tot = sum(wt)) %>%
  group_by(source, year, wt_tot, space_cat) %>% summarize(wt_n = sum(wt)) %>% mutate(percent = wt_n/wt_tot) %>% select(-wt_n, -wt_tot)
write.csv(dhs.results, "locations/kenya/birth_spacing_dhs.csv")




