#####################################################
#### Family planning analysis 2018
#### Rwanda and Nigeria DHS
#### Adapted from https://github.com/mzimmermann-IDM/Family-Planning/blob/master/Youth%20and%20subnational

rm(list=ls())

# -- Libraries -- #
library(tidyverse)      #contains ggplot, tibble, stingr, dplyr, plry, tidyselect, tidyr
library(haven)          #Import and Export 'SPSS', 'Stata' and 'SAS' Files
library(sf)             #spatial mapping
library(survey)

# ----------------------------------------------------------------------------- #
# -- Preparing the dataset -- #

options(survey.lonely.psu="adjust")

# -- access the boundaries at:
# https://spatialdata.dhsprogram.com/boundaries/#view=table&countryId=AF

# -- set working directory -- #
#setwd("/u/cliffk/idm/fp/fp_analyses") # CK
setwd("/home/dklein/GIT/fp_analyses") # DK

# -- read shapefiles -- #

# Nigeria # CK
#pts.N <- st_read("/u/cliffk/idm/Dropbox/fp-dhs-data/nigeria/NGGE6AFL", layer = "NGGE6AFL") # read in cluster shape # CK
shape2.N <- st_read("/home/idm_user/idm/Dropbox/fp-dhs-data/nigeria/NGGE6AFL/NGGE6AFL.shp") # read in state shape # CK
shape.N <- shape2.N %>% st_join(pts.N) #%>% mutate(v001 = DHSCLUST) # join all shape files
key.N <- as.data.frame(shape.N) #%>% select(v001, DHSREGEN)

# Senegal

# -- read individual recode data -- #

#dhs.raw.R<-read_dta("Data/RwandaDHS/2014/RWIR70DT/RWIR70FL.DTA")
#dhs.raw.N<-read_dta("/home/idm_user/idm/Dropbox/fp-dhs-data/nigeria/NGIR61DT/NGIR61FL.DTA") # CK
dhs.raw.SN<-read_dta("/home/dklein/sdb2/Dropbox (IDM)/FP Dynamic Modeling/DHS/Country data/Senegal/2017/SNIR7ZDT/SNIR7ZFL.DTA")

# This works (returns columns)
v <- dhs.raw.SN %>% select(starts_with("v"))
#vcal <- dhs.raw.SN %>% select(starts_with("vcal_"))

DHSdesign <- svydesign(id = dhs.raw.SN$v021, strata=dhs.raw.SN$v022, weights = dhs.raw.SN$v005/1000000, data=dhs.raw.SN)
svymean(~v213, DHSdesign)
cv(svymean(~v213, DHSdesign))
confint(svymean(~v213, DHSdesign))
svymean(~factor(v025), DHSdesign)

svyby(~v213, ~v025, DHSdesign, svymean)
