#######################################################################################
# -- Extract the proportion of women aged X years when they first entered a partnership
# from a DHS individual recode (IR) dataset
#######################################################################################

# -- Import libraries -- #
library(tidyverse)
library(haven)
library(survey)
library(withr)

# Kenya 2022 individual recode
home_dir <- path.expand("~")   # replace with your own path to the DTA file
dhs_dir <- "DHS"
survey_dir <-"KEIR8BDT"
filename <- "KEIR8BFL.DTA"
filepath <- file.path(home_dir, dhs_dir, survey_dir, filename)

# -- Load the data
data.raw <- read_dta(filepath)

# -- Clean data -- #
data <- data.raw %>%
  mutate(age_partner = v511)

# -- Create survey object so means and SEM are properly adjusted for the survey design --#
svydesign_obj = svydesign(id = data$v001, strata=data$v023, weights = data$v005/1000000, data=data)


table.partnership <- as.data.frame(svytable(~age_partner, svydesign_obj)) %>%
  mutate(percent = Freq/sum(Freq)) %>% select(-Freq)


home_dir <- path.expand("~")   # replace with your own path to the DTA file
fpsim_dir <- "fpsim"           # path to root directory of fpsim
locations_dir <- "fpsim/locations"
country_dir <- "kenya"
data_dir <- "data"
filename <- "age_partnership.csv"
country_data_path <- file.path(home_dir, fpsim_dir, locations, country_dir, data_dir, filename)

# -- Save file --# Note that this will replace the default *.csv file for the specified country
write.csv(table.partnerership, country_data_path, row.names = F)