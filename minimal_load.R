#####################################################
#### Family planning analysis 2018
#### Rwanda and Nigeria DHS
#### Adapted from https://github.com/mzimmermann-IDM/Family-Planning/blob/master/Youth%20and%20subnational

rm(list=ls())

# -- Libraries -- #
library(tidyverse)      #contains ggplot, tibble, stingr, dplyr, plry, tidyselect, tidyr
library(haven)          #Import and Export 'SPSS', 'Stata' and 'SAS' Files
library(sf)             #spatial mapping

# ----------------------------------------------------------------------------- #
# -- Preparing the dataset -- #

options(survey.lonely.psu="adjust")

# -- access the boundaries at:
# https://spatialdata.dhsprogram.com/boundaries/#view=table&countryId=AF

# -- set working directory -- #
setwd("/u/cliffk/idm/fp/fp_analyses") # CK

# -- read shapefiles -- #

# Nigeria # CK
pts.N <- st_read("/u/cliffk/idm/Dropbox/fp-dhs-data/nigeria/NGGE6AFL", layer = "NGGE6AFL") # read in cluster shape # CK
shape2.N <- st_read("/home/idm_user/idm/Dropbox/fp-dhs-data/nigeria/NGGE6AFL/NGGE6AFL.shp") # read in state shape # CK
shape.N <- shape2.N %>% st_join(pts.N) #%>% mutate(v001 = DHSCLUST) # join all shape files
key.N <- as.data.frame(shape.N) #%>% select(v001, DHSREGEN)

# -- read individual recode data -- #

#dhs.raw.R<-read_dta("Data/RwandaDHS/2014/RWIR70DT/RWIR70FL.DTA")
#dhs.raw.N<-read_dta("/home/idm_user/idm/Dropbox/fp-dhs-data/nigeria/NGIR61DT/NGIR61FL.DTA") # CK
dhs.raw.N<-read_dta("/u/cliffk/idm/fp/data/test_load/NGIR6ADT/NGIR6AFL.DTA")

# This works (returns columns)
worked <- dhs.raw.N %>% select(starts_with("v"))

# But this doesn't...
failed <- dhs.raw.N %>% select(starts_with("vcal_"))