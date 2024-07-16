#################################################################
# -- Extract urban data from a DHS individual recode (IR) dataset
#################################################################

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
  mutate(urban = ifelse(v025 == 1, 1, 0)) # 1 if urban

# -- Create survey object  so means and SEM are properly adjusted for the survey design --#
svydesign_obj = svydesign(id = data$v001, strata=data$v023, weights = data$v005/1000000, data=data)

# -- Compute proportion of women living in an urban setting -- #
table.urban <- as.data.frame(svymean(~urban, svydesign_obj)) %>% rename(urban.se = urban)

home_dir <- path.expand("~")   #
fpsim_dir <- "fpsim"           # replace with your path to root directory of fpsim
locations_dir <- "fpsim/locations"
country_dir <- "kenya"
data_dir <- "data"
filename <- "urban_.csv"
filepath <- file.path(home_dir, fpsim_dir, locations_dir,  country_dir, data_dir, filename)

# -- Save file --# Note that this will replace the default urban.csv for the specified country
write.csv(table.urban, filepath, row.names = F)
