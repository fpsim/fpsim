###################################################
# -- Analysis on PMA data
# -- Find distribution of fertility intent by age
# -- Marita Zimmermann
# -- May 2024
###################################################

# -- To get formatted PMA dataset, run PMA_recode in data processing folder

library(tidyverse)      
library(survey)


options(survey.lonely.psu="adjust")
svydes <- svydesign(id = ~EA_ID, strata = ~strata, weights =  ~FQweight, data = All_data , nest = T)

fertility_intent <- as.data.frame(prop.table(svytable(~fertility_intent+age, svydes), 2)) 

write.csv(fertility_intent, "fpsim/locations/data_processing/fertility_intent.csv", row.names = F)
