################################################################################
# -- Find distribution of fertility intent by age from recoded PMA datasets
# -- To get "recoded.datasets" table, first run recode_pma_datasets.R
# Original analysis by Marita Zimmermann, May 2024
###############################################################################


library(tidyverse)
library(survey)

#--------------Find distribution of fertility intent---------------------------#
options(survey.lonely.psu = "adjust")
svydes <- svydesign(id = ~EA_ID, strata = ~strata, weights =  ~FQweight,
                    data = recoded.datasets, nest = TRUE)

fertility.intent <- as.data.frame(prop.table(svytable(~fertility_intent + age,
                                                      svydes), 2))

# Rename column, we tend to use small  inside fpsim
names(fertility.intent)[names(fertility.intent) == 'Freq'] <- 'freq'

# Transform age to numeric values
fertility.intent$age <- as.numeric(as.character(fertility.intent$age))

#-----------------------------------------------------------------------------#

home_dir <- path.expand("~")   # replace with your own path to the PMA file
fpsim_dir <- "fpsim"           # REPLACE WITH PATH to root directory of fpsim
locations_dir <- "fpsim/locations"
country_dir <- "kenya"
data_dir <- "data"
filename <- "fertility_intent.csv"
country_data_path <- file.path(home_dir, fpsim_dir, locations_dir, country_dir,
                               data_dir, filename)

# -- Save file --#
# Note that this will replace the default *.csv file for the specified country
write.csv(fertility.intent, country_data_path, row.names = FALSE)
