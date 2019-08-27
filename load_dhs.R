#####################################################
#### Family planning analysis 2018
#### Rwanda and Nigeria DHS
#### Adapted from https://github.com/mzimmermann-IDM/Family-Planning/blob/master/Youth%20and%20subnational

rm(list=ls())

# -- Libraries -- #
library(tidyverse)      #contains ggplot, tibble, stingr, dplyr, plry, tidyselect, tidyr
library(haven)          #Import and Export 'SPSS', 'Stata' and 'SAS' Files
library(survey)         #complex survey data
library(DHS.rates)      #Calculate fertility rates from DHS
library(jtools)         #visualize regression results
library(latticeExtra)   #3d bar charts
library(ggridges)       #formatting continuous ggplot
library(viridis)        #shading figures
library(sf)             #spatial mapping
library(gridExtra)      #combine multiple plots
library(scales)         # formatting graph axes
library(factoextra)     #plotting clusters

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
dhs.raw.N<-read_dta("/home/idm_user/idm/Dropbox/fp-dhs-data/nigeria/NGIR61DT/NGIR61FL.DTA") # CK

# -- Create working dataset -- #

data.input <- function(dhs.raw, key){
  #keep only necessary data points
  dhs.min <- dhs.raw %>%
    dplyr::select("v000", "awfactt", "v001", "v005", "v008", "v010", "v011", "v012",  "v013", "v017", "v024", "v025",
                  "v106", "v133", "v190", "v191",
                  "v201", "v211", "v212", "v213", "v220", "v221", "v222", "v225",
                  "v312", "v313",
                  "v502", "v509","v511", "v525", "v527", "v528", "v529", "v536",
                  "v605", "v623", "v625a", "v626a",
                  "v714", "v721", "v731",
                  starts_with("bidx_"), starts_with("b2_"), starts_with("b3_"), starts_with("b5_"), starts_with("b7_"), starts_with("b11_"),
                  starts_with("m10_"), starts_with("vcal_"), starts_with("v3a08")) %>%
    left_join(key, by = "v001")
}
dhs.min.N <- data.input(dhs.raw = dhs.raw.N, key = key.N) # CK

# ------------------------------------------------------------------------ #
# -- start analyzing survey -- #

# first checking to see if "awfctt" variable is in the dataset
# This is the all women factor, if it is not there we add it
# it will only be in surveys that ONLY survey married women
data.svyvars <- function(dhs){
  if(!("awfctt"%in%names(dhs))){dhs$awfactt<-100} # CK
  
  dhs<-dhs%>%mutate(wt=v005/1000000, # have to scale the weight
                    year_cmc=floor((v008-1)/12)+1900, # finding the year based on Century Month Codes
                    month_cmc=v008-(year_cmc-1900)*12, # finding the month
                    year=year_cmc+month_cmc/12, # grabbing year
                    country_code=substr(v000,start=1,stop=2), # finding the country code
                    recode=substr(v000,3,3), # which recode manual to look at
                    aw=awfactt/100*wt) # the all woman factor
}

dhs.svy.N <- data.svyvars(dhs = dhs.min.N) # CK

# ------------------------------------------------------------------------ #
# -- recode the variables of interest -- #

variable.recode <- function(dhs, agestart_edu, UNpop){
  dhs<-dhs%>%mutate(
    # age
    age_cat = ordered(as.numeric(v013), levels = c(1,2,3,4,5,6,7), labels = c("15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49")), # current age in 5 year buckets 1 15-19, 2 20-24, 3 25-29, 4 30-34, 5 35-39, 6 40-44, 7 45-49
    age = as.numeric(v012),
    agefirstbirth = as.numeric(v212), # age at first birth 10-49
    agefirstbirthcat = ifelse(v201==0,"no children",as.character(cut(v212, c(0,14,19,24,29,49), labels = c("0-14","15-19","20-24","25-29","30+")))), # create 5 year age buckets for age at first birth, with additional category for no children
    birthcohort = cut( ifelse(v010<100, v010+1900, v010), breaks = seq(1940, 2000, by=5)), #reformat birth year to 4 digits then find 5 year birth cohort
    # current contraceptive method
    curr_method=v312, # 0 Not using  1 Pill  2 IUD  3 Injections  4 Diaphragm  5 Condom  6 Female sterilization  7 Male sterilization  8 Periodic abstinence  9 Withdrawal  10 Other  11 Implants/Norplant  12 Abstinence  13 Lactational amenorrhea (LAM)  14 Female condom  15 Foam or jelly  17 Oher modern method  18 Specific method 1  19 Specific method 2  20 Specific method 3
    pill=as.numeric(ifelse(!is.na(v312), as.character(v312)%in%c("pill","1",1)),NA), 
    iud=as.numeric(ifelse(!is.na(v312),as.character(v312)%in%c("iud","2",2)),NA),
    injections=as.numeric(ifelse(!is.na(v312),as.character(v312)%in%c("injections","3",3)),NA),
    diaphragm=as.numeric(ifelse(!is.na(v312),as.character(v312)%in%c("diaphragm","4","diaphragm/foam/jelly",4)),NA),
    condom=as.numeric(ifelse(!is.na(v312),as.character(v312)%in%c("condom","5","male condom",5)),NA),
    fem_ster=as.numeric(ifelse(!is.na(v312),as.character(v312)%in%c("female sterilization","6",6)),NA),
    implant=as.numeric(ifelse(!is.na(v312),as.character(v312)%in%c("norplant","11","implants/norplant","implants",11)),NA),
    curr_methodtype = v313, # 0 No method 1 Folkloric method  2 Traditional method  3 Modern method
    contracept_modern = ifelse(v313==9, NA, ifelse(v313==3,1,0)), #using modern contracptive
    contracept_any = ifelse(v313==9, NA, ifelse(v313==0,0,1)), #using any contraceptive
    contracept_none_eligible = ifelse(is.na(v213), NA, ifelse(v312==0 & v213 != 1 & (v605==5 | v605==2),1,0)), # not using any contraception, not pregnant, not want to become pregnant
    # fertility
    parity_cat=as.factor(ifelse(v220 == 6, "6+", v220)), # categorical 0-5, 6+
    parity = ifelse(is.na(bidx_01),0,pmax(bidx_01,bidx_02,bidx_03,bidx_04,bidx_05,bidx_06,bidx_07,bidx_08,bidx_09,bidx_10,bidx_11,bidx_12,bidx_13,bidx_14,bidx_15,bidx_16,bidx_17,bidx_18,bidx_19,bidx_20, na.rm=T)), # numerical
    fertility_status = v623, # 0 fecund, 1 pregnant, 2 postpartum amenorrheic, 3 fecund menopausal
    birthinterval = rowMeans(subset(dhs, select = c(b11_01, b11_02, b11_03, b11_04, b11_05, b11_06, b11_07, b11_08, b11_09, b11_10, b11_11, 
                                                    b11_12, b11_13, b11_14, b11_15, b11_16, b11_17, b11_18, b11_19, b11_20)), na.rm=TRUE), # mean of birth interval for all births
    birthintervalcat = ifelse(is.na(birthinterval),0,cut(birthinterval, c(0, 24, 36, 48, 300), labels = c("(9,24]",  "(24,36]",  "(36,48]", "(48,271]"))), # categories of birth intervals 0-2 years, 2-3, 3-4, 4+
    wanted = ifelse(parity==1,m10_1, ifelse(parity==2,m10_2, ifelse(parity==3,m10_3, ifelse(parity==4,m10_4, ifelse(parity==5,m10_5, ifelse(parity==6,m10_6,NA)))))), # was first birth wanted
      wanted = ifelse(wanted==2 | wanted==3, 0, wanted),
    pregnancy_surveyyr = ifelse(v222<=12, 1, 0), # if women had a birth in year of survey
    vcal_start = v017, #century month for start of 80 month fertility cal
    vcal_end = v017 + nchar(vcal_1)-1, #century month for end of 80 month fertility cal
    # need for contraception
    unmetneed = v626a, # 0 Never had sex, 1 Unmet need for spacing, 2 Unmet need for limiting, 3 Using for spacing, 4 Using for limiting, 7 No unmet need, 8 Not married and no sex in last 30 days, 9 Infecund, menopausal, 99 Missing
    unmetneed_cat = ifelse(is.na(v626a), NA, ifelse(v626a==3 | v626a==4, 0, ifelse(v626a==1 | v626a==2, 1, 0))), # 0 met need, 1 unmet neet, 2 no need
    unmet_need = ifelse(unmetneed_cat == 1, 1, 0),
    met_need = ifelse(unmetneed_cat == 0, 1, 0), 
    no_need = ifelse(unmetneed_cat == 2, 1, 0),
    # marriage status and timing
    evermarr=ifelse(!is.na(v502), v502%in%c(1,2), NA),
    age_marr = v511, # age at marriage
    marr_birth = v211 - v509, #CMC of first birth minus marriage
    marr_birth_close = ifelse(is.na(marr_birth),NA,ifelse(marr_birth>=-9 & marr_birth<=9,1,0)), # birth within 9 months of marriage
    # education status and timing
    education  = as.character(v106), # 0 No education  1 Primary  2 Secondary  3 Higher
    eduyrs = ifelse(v133==99,NA,v133), # total years of education
    finish_edu = eduyrs + agestart_edu, # age finish education, starting at age 7 in Rwanda, 6 in Nigeria (World Bank)
    edu_birth = v212 - finish_edu, # finish education to first birth interval
    stop_edu_birth = ifelse(is.na(edu_birth), NA, ifelse(edu_birth >= -1 & edu_birth <= 1, 1, 0)), # yes or no did woman stop school within 1 year of birth
    stop_edu_birth.2 = ifelse(is.na(edu_birth), NA, ifelse(edu_birth >= -2 & edu_birth <= 2, 1, 0)), # yes or no did woman stop school within 1 year of birth
    edu_marr = v511 - finish_edu, # Finish education to marriage interval (years)
    edu_birth_cat = ifelse(is.na(edu_birth), NA, ifelse(edu_birth > 1, 2, ifelse(edu_birth >= -1 & edu_birth <= 1, 1, 0))), # categorical variable for 0 - continued school after first birth, 1 stopped school within 1 year of 1st birth, 2 - had completed school at least 1 year prior to first birth
    in_school = ifelse(finish_edu<age , 0, 1), # if woman is still in school at time of survey
    # employment
    emp.yr = ifelse(v731>0 & v731<9, 1, ifelse(v731==0,0,NA)), # 1 if employed in past year, currently, or on leave in last 7 days
    employed = ifelse(v714==1,1,ifelse(v714==0,0,NA)), # currenlty working
    # demographics
    wealthindex = (v191 / 100000) +10, # wealth index v191, make smaller and all positive for model fitting
    wealth 	  = as.numeric(as.character(v190)), # wealth index quintile ( 1 poorest, 5 richest)
    poorest = ifelse(wealth==1,1,0), # binary variable for poorest wealth quintile
    richest = ifelse(wealth==5,1,0), # binary varibale for richest wealth quintile
    urbanrural= as.character(v025))  # 1 urban, 2 rural

  # calculate age of the mother at each birth (rounded to the year), 1 is most recent
  dhs[paste0("agebirth_", 1:20)] <- floor((data.matrix(dhs[paste0("b3_", str_pad(1:20, 2, pad = "0"))])-matrix(dhs$v011,ncol=20,nrow=nrow(dhs)))/12)
  
  # calendar position of first 6 births... this doesn't work, needs the ifelse part
  # dhs[paste0("birthpos", 1:6)] <- matrix(dhs$vcal_end,ncol=6,nrow=nrow(dhs)) - data.matrix(dhs[paste0("b3_", str_pad(1:6, 2, pad="0"))]) +1
  dhs <- dhs %>% mutate(
    birthpos1 = ifelse(b3_01>=vcal_start & b3_01<=vcal_end, vcal_end - b3_01 +1, NA), # calendar position birth 1
    birthpos2 = ifelse(b3_02>=vcal_start & b3_02<=vcal_end, vcal_end - b3_02 +1, NA), # calendar position birth 2
    birthpos3 = ifelse(b3_03>=vcal_start & b3_03<=vcal_end, vcal_end - b3_03 +1, NA), # calendar position birth 3
    birthpos4 = ifelse(b3_04>=vcal_start & b3_04<=vcal_end, vcal_end - b3_04 +1, NA), # calendar position birth 4
    birthpos5 = ifelse(b3_05>=vcal_start & b3_05<=vcal_end, vcal_end - b3_05 +1, NA), # calendar position birth 5
    birthpos6 = ifelse(b3_06>=vcal_start & b3_06<=vcal_end, vcal_end - b3_06 +1, NA) ) # calendar position birth 6
  
  # create matrix with columns for ages 10-49, with a 1 if there was a birth in that year and 0 otherwise
  dhs[paste0("birth_", 10:49)] <- t(apply(data.matrix(dhs[paste0("agebirth_", 1:20)]),1,function(y)as.numeric(c(10:49)%in%y))) 
  # create matrix with columns for ages 10-49, with a 1 if the woman is under that age so included in sample size (year of survey not included because not a full year)
  dhs[paste0("insampleage_", 10:49)] <- t(apply(data.matrix(dhs$age),1,function(y)as.numeric(c(10:49)<=y))) 
  # create variables for if live childern at each age for the mother
  dhs[paste0("birthalive_", 10:49)] <- t(apply(data.matrix(
    dhs[paste0("agebirth_", 1:20)] * dhs[paste0("b5_", str_pad(1:20, 2, pad = "0"))]), # multiply birth ages by if the child is alove to get matrix of only live children
    1, function(y)as.numeric(c(10:49)%in%y))) 
  # create matrix with columns for ages 10-49, with a 1 if the woman is still in school
  dhs[paste0("eduage_", 10:49)] <- t(apply(data.matrix(dhs$finish_edu),1,function(y)as.numeric(c(10:49)>=y))) 
  # create binary variable for stopping education within 1 year of each age (for use in regression below)
  dhs[paste0("stopedu_", 12:45)] <- t(apply(data.matrix(dhs$finish_edu),1,function(y)as.numeric(c(13:46)>=y & c(11:44)<=y)))
  # create matrix with columns for ages 10-49, with a 1 if the woman is married
  dhs[paste0("marriedage_", 10:49)] <- t(apply(data.matrix(dhs$age_marr),1,function(y)as.numeric(c(10:49)>=y)))
  # calculate parity at each age as the total number of children alive in previous year
  dhs <- dhs %>% mutate( 
    parityage_10 = 0, 
    parityage_11 = birthalive_10,
    parityage_12 = rowSums(dhs[paste0("birthalive_", 10:11)]),
    parityage_13 = rowSums(dhs[paste0("birthalive_", 10:12)]),
    parityage_14 = rowSums(dhs[paste0("birthalive_", 10:13)]),
    parityage_15 = rowSums(dhs[paste0("birthalive_", 10:14)]),
    parityage_16 = rowSums(dhs[paste0("birthalive_", 10:15)]),
    parityage_17 = rowSums(dhs[paste0("birthalive_", 10:16)]),
    parityage_18 = rowSums(dhs[paste0("birthalive_", 10:17)]),
    parityage_19 = rowSums(dhs[paste0("birthalive_", 10:18)]),
    parityage_20 = rowSums(dhs[paste0("birthalive_", 10:19)]),
    parityage_21 = rowSums(dhs[paste0("birthalive_", 10:20)]),
    parityage_22 = rowSums(dhs[paste0("birthalive_", 10:21)]),
    parityage_23 = rowSums(dhs[paste0("birthalive_", 10:22)]),
    parityage_24 = rowSums(dhs[paste0("birthalive_", 10:23)]),
    parityage_25 = rowSums(dhs[paste0("birthalive_", 10:24)]),
    parityage_26 = rowSums(dhs[paste0("birthalive_", 10:25)]),
    parityage_27 = rowSums(dhs[paste0("birthalive_", 10:26)]),
    parityage_28 = rowSums(dhs[paste0("birthalive_", 10:27)]),
    parityage_29 = rowSums(dhs[paste0("birthalive_", 10:28)]),
    parityage_30 = rowSums(dhs[paste0("birthalive_", 10:29)]),
    parityage_31 = rowSums(dhs[paste0("birthalive_", 10:30)]),
    parityage_32 = rowSums(dhs[paste0("birthalive_", 10:31)]),
    parityage_33 = rowSums(dhs[paste0("birthalive_", 10:32)]),
    parityage_34 = rowSums(dhs[paste0("birthalive_", 10:33)]),
    parityage_35 = rowSums(dhs[paste0("birthalive_", 10:34)]),
    parityage_36 = rowSums(dhs[paste0("birthalive_", 10:35)]),
    parityage_37 = rowSums(dhs[paste0("birthalive_", 10:36)]),
    parityage_38 = rowSums(dhs[paste0("birthalive_", 10:37)]),
    parityage_39 = rowSums(dhs[paste0("birthalive_", 10:38)]),
    parityage_40 = rowSums(dhs[paste0("birthalive_", 10:39)]),
    parityage_41 = rowSums(dhs[paste0("birthalive_", 10:40)]),
    parityage_42 = rowSums(dhs[paste0("birthalive_", 10:41)]),
    parityage_43 = rowSums(dhs[paste0("birthalive_", 10:42)]),
    parityage_44 = rowSums(dhs[paste0("birthalive_", 10:43)]),
    parityage_45 = rowSums(dhs[paste0("birthalive_", 10:44)]),
    parityage_46 = rowSums(dhs[paste0("birthalive_", 10:45)]),
    parityage_47 = rowSums(dhs[paste0("birthalive_", 10:46)]),
    parityage_48 = rowSums(dhs[paste0("birthalive_", 10:47)]),
    parityage_49 = rowSums(dhs[paste0("birthalive_", 10:48)]))
  
  dhs<-dhs%>%mutate(
    bcpos1 = ifelse(substr(vcal_1, birthpos1+7, birthpos1+7) != "P",  birthpos1+7, 
                    ifelse(substr(vcal_1, birthpos1+8, birthpos1+8) != "P",  birthpos1+8,
                           ifelse(substr(vcal_1, birthpos1+9, birthpos1+9) != "P",  birthpos1+9,
                                  ifelse(substr(vcal_1, birthpos1+10, birthpos1+10) != "P",  birthpos1+10, birthpos1+11)))),
    bcpos2 = ifelse(substr(vcal_1, birthpos2+7, birthpos2+7) != "P",  birthpos2+7, 
                    ifelse(substr(vcal_1, birthpos2+8, birthpos2+8) != "P",  birthpos2+8,
                           ifelse(substr(vcal_1, birthpos2+9, birthpos2+9) != "P",  birthpos2+9,
                                  ifelse(substr(vcal_1, birthpos2+10, birthpos2+10) != "P",  birthpos2+10, birthpos2+11)))),
    bcpos3 = ifelse(substr(vcal_1, birthpos3+7, birthpos3+7) != "P",  birthpos3+7, 
                    ifelse(substr(vcal_1, birthpos3+8, birthpos3+8) != "P",  birthpos3+8,
                           ifelse(substr(vcal_1, birthpos3+9, birthpos3+9) != "P",  birthpos3+9,
                                  ifelse(substr(vcal_1, birthpos3+10, birthpos3+10) != "P",  birthpos3+10, birthpos3+11)))),
    bcpos4 = ifelse(substr(vcal_1, birthpos4+7, birthpos4+7) != "P",  birthpos4+7, 
                    ifelse(substr(vcal_1, birthpos4+8, birthpos4+8) != "P",  birthpos4+8,
                           ifelse(substr(vcal_1, birthpos4+9, birthpos4+9) != "P",  birthpos4+9,
                                  ifelse(substr(vcal_1, birthpos4+10, birthpos4+10) != "P",  birthpos4+10, birthpos4+11)))),
    bcpos5 = ifelse(substr(vcal_1, birthpos5+7, birthpos5+7) != "P",  birthpos5+7, 
                    ifelse(substr(vcal_1, birthpos5+8, birthpos5+8) != "P",  birthpos5+8,
                           ifelse(substr(vcal_1, birthpos5+9, birthpos5+9) != "P",  birthpos5+9,
                                  ifelse(substr(vcal_1, birthpos5+10, birthpos5+10) != "P",  birthpos5+10, birthpos5+11)))),
    bcpos6 = ifelse(substr(vcal_1, birthpos6+7, birthpos6+7) != "P",  birthpos6+7, 
                    ifelse(substr(vcal_1, birthpos6+8, birthpos6+8) != "P",  birthpos6+8,
                           ifelse(substr(vcal_1, birthpos6+9, birthpos6+9) != "P",  birthpos6+9,
                                  ifelse(substr(vcal_1, birthpos6+10, birthpos6+10) != "P",  birthpos6+10, birthpos6+11)))))
  dhs<-dhs%>%mutate(
    bc_1 = ifelse(bcpos1<=80, substr(vcal_1, bcpos1, bcpos1), NA), #bc immediately prior to pregnancy
    bc_2 = ifelse(bcpos2<=80, substr(vcal_1, bcpos2, bcpos2), NA),
    bc_3 = ifelse(bcpos3<=80, substr(vcal_1, bcpos3, bcpos3), NA),
    bc_4 = ifelse(bcpos4<=80, substr(vcal_1, bcpos4, bcpos4), NA),
    bc_5 = ifelse(bcpos5<=80, substr(vcal_1, bcpos5, bcpos5), NA),
    bc_6 = ifelse(bcpos6<=80, substr(vcal_1, bcpos6, bcpos6), NA) )
  
  # Unmet need algorithm with 4 definitions of sexual activity
  dhs <- dhs %>% mutate(
    notsexactive1 = ifelse(v528>30 | is.na(v528), 1, 0), # define sexual activity
    notsexactive2 = ifelse(v529>12 | is.na(v529), 1, 0), # if assume anyone who ever had sex is active
    notsexactive3 = ifelse(v525==0, 1, 0), # if assume anyone who ever had sex is active
    notsexactive4 = 0) # if assume everyone can be
  # 0 Never had sex
  # 1 Unmet need for spacing
  # 2 Unmet need for limiting
  # 3 Using for spacing
  # 4 Using for limiting
  # 7 No unmet need
  # 8 Not married and no sex
  # 9 Infecund, menopausal
  for (i in 1:4) {
    dhs <- dhs %>% mutate(
      # CONTRACEPTIVE USERS - GROUP 1
      unmet = ifelse( v312!=0 & v605>=5 & v605<=7, 4, NA), #  using to limit if wants no more, sterilized, or declared infecund
      unmet = ifelse(is.na(unmet) & v312!=0, 3, unmet), # using to space - all other contraceptive users
      # PREGNANT or POSTPARTUM AMENORRHEIC (PPA) WOMEN - GROUP 2
      wantedlast = ifelse((is.na(v225) | v225==9) & v213!=1, m10_1, v225),  # Classify based on wantedness of current pregnancy or last birth
      unmet = ifelse(is.na(unmet) & (v625a==1 | v625a==2) & wantedlast==1, 7, unmet), # no unmet need if wanted current pregnancy/last birth then/at that time
      unmet = ifelse(is.na(unmet) & (v625a==1 | v625a==2) & wantedlast==2, 1, unmet), # unmet need for spacing if wanted current pregnancy/last birth later
      unmet = ifelse(is.na(unmet) & (v625a==1 | v625a==2) & wantedlast==3, 2, unmet), # unmet need for limiting if wanted current pregnancy/last birth not at all
      unmet = ifelse(is.na(unmet) & (v625a==1 | v625a==2) & (is.na(wantedlast) | wantedlast==9), 99, unmet), # missing=missing
      # INFECUNDITY - GROUP 3
      unmet = ifelse( v625a==3 & is.na(unmet), 9, unmet),
      # NO NEED FOR UNMARRIED WOMEN WHO ARE NOT SEXUALLY ACTIVE
      unmet = ifelse(is.na(unmet) & v502==0 & v525==0 & i!=4, 0, unmet)) # never married and never had sex ##########################
    dhs$unmet <- with(dhs, ifelse(is.na(unmet) & (v502==0 | v502==2)  & dhs[,paste0("notsexactive", i)]==1, 8, unmet)) # if unmarried and not sexually active in last 30 days, assume no need
    # FECUND WOMEN - GROUP 4
    dhs$unmet <- with(dhs, ifelse( v605==1 & is.na(unmet), 7, unmet)) # wants within 2 years
    dhs$unmet <- with(dhs, ifelse( v605>=2 & v605<=4 & is.na(unmet), 1, unmet)) # wants in 2+ years, wants undecided timing, or unsure if wants
    dhs$unmet <- with(dhs, ifelse( v605==5 & is.na(unmet), 2, unmet)) # wants no more
    dhs$unmet <- with(dhs, ifelse( unmet==99, NA, unmet))
    dhs[,paste0("unmet",i)] <- dhs$unmet
    dhs$unmet <- NULL
  }
  
  dhs$discrepancy = with(dhs, ifelse(unmet1==v626a,0,1)) # check if algorithm matches DHS provided variable for unmet need
  
  dhs <- dhs %>% mutate( # 1 if unmet need by each definition, 0 otherwise
    unmet_1 = ifelse(is.na(unmet1), NA, ifelse(unmet1==1 | unmet1==2, 1, 0)),
    unmet_2 = ifelse(is.na(unmet2), NA, ifelse(unmet2==1 | unmet2==2, 1, 0)),
    unmet_3 = ifelse(is.na(unmet3), NA, ifelse(unmet3==1 | unmet3==2, 1, 0)),
    unmet_4 = ifelse(is.na(unmet4), NA, ifelse(unmet4==1 | unmet4==2, 1, 0)),
    # If had a pregnancy that year and has unmet need by each definition
    pregnancy.surveyyr.unmet_1 = ifelse(pregnancy_surveyyr==1 & unmet_1==1, 1, 0),
    pregnancy.surveyyr.unmet_2 = ifelse(pregnancy_surveyyr==1 & unmet_2==1, 1, 0),
    pregnancy.surveyyr.unmet_3 = ifelse(pregnancy_surveyyr==1 & unmet_3==1, 1, 0),
    pregnancy.surveyyr.unmet_4 = ifelse(pregnancy_surveyyr==1 & unmet_4==1, 1, 0)) %>%
    
  # create strata variable
  left_join(unique(dhs[,c("v024","v025")]) %>% mutate(strata = 1:n()), by = c("v024", "v025"))
  
  # categorize reasons for non-use
  dhs <- dhs %>% mutate(id = 1:n()) %>% 
    gather("var", "reason", grep("v3a08", names(.))) %>% filter(reason==1 | var == "v3a08a") %>% # gather all reasons variables then add a category variable
    group_by(id) %>%
    mutate(reason.cat = ifelse(var == "v3a08m" | var == "v3a08n", "knowledge", # knows no method, knows no source
                               ifelse(var == "v3a08r", "cost",
                                      ifelse(var == "v3a08d" | var == "v3a08e", "fecundity", # menopausal/hysterectomy, sub/infecund
                                             ifelse(var == "v3a08q" | var == "v3a08u" | var == "v3a08v" | var == "v3a08s", "access",  ##lack of access/too far, pref method not avail, no method avail, inconvenient to use
                                                    ifelse(var == "v3a08f" | var == "v3a08g", "recent.birth", # postpartum, breastfeeding
                                                           ifelse(var == "v3a08h" | var == "v3a08i" | var == "v3a08j" | var == "v3a08k" | var == "v3a08l", "opposition", # fatalistic, opposed, partner opp, others opposed, religious
                                                                  ifelse((var == "v3a08a" & reason == 1) | var == "v3a08b" | var == "v3a08c", "sexual.activity", # not married, no sex, infreq sex
                                                                         ifelse(var == "v3a08o" | var == "v3a08p" | var == "v3a08t", "health.risks", # health concerns, side effects, interefere with body
                                                                                ifelse(var == "v3a08x", "other", NA))))))))), reason.cat= paste0("unmet.", reason.cat),
           reason = ifelse(is.na(reason.cat), NA, 1), 
           #reason = ifelse(unmet_1 != 1, NA, reason), # INCLUDE REASONS ONLY FOR THOSE WITH UNMET NEED (DHS DEFINITION)
           var = NULL) %>% distinct() %>% spread(reason.cat, reason) %>% 
           rowwise() %>% mutate(num_reasons = sum(unmet.knowledge, unmet.cost, unmet.access, unmet.health.risks, unmet.fecundity, unmet.recent.birth, unmet.opposition, 
                                                  unmet.sexual.activity, unmet.other, na.rm = T)) %>% ungroup %>%
    # is reason for non-use adressable with interventions?
           mutate(reason.addressable = ifelse(unmet.knowledge == 1 | unmet.cost == 1 | unmet.access == 1 | unmet.health.risks == 1, 1, NA),
                  reason.notaddressable = ifelse(unmet.fecundity == 1 | unmet.recent.birth == 1 | unmet.opposition == 1 | unmet.sexual.activity == 1 | unmet.other == 1, 1, NA),
                  reason.addressable = ifelse(is.na(reason.addressable) & reason.notaddressable == 1, 0, reason.addressable),
                  reason.notaddressable = ifelse(is.na(reason.notaddressable) & reason.addressable == 1, 0, reason.notaddressable),
                  unmet.NA = NULL)
  
# Dataset with population size
  popweight <- cbind(as.matrix(table(dhs$age_cat)), UNpop) # number of women (in 1000s) from UN pop 2015 
  dhs <- dhs %>% 
    mutate(num_women = ifelse(age<20, popweight[1,2]*1000/popweight[1,1],
                              ifelse(age<25, popweight[2,2]*1000/popweight[2,1],
                                     ifelse(age<30, popweight[3,2]*1000/popweight[3,1],
                                            ifelse(age<35, popweight[4,2]*1000/popweight[4,1],
                                                   ifelse(age<40, popweight[5,2]*1000/popweight[5,1],
                                                          ifelse(age<45, popweight[6,2]*1000/popweight[6,1], 
                                                                 popweight[7,2]*1000/popweight[7,1] )))))), #add a variable for the number of women each woman represents
           svy_num_women = 1,
   # multiply unmet need categories and pregnancies by population size
   unmet.pop_1 = unmet_1 * num_women, unmet.pop_2 = unmet_2 * num_women, unmet.pop_3 = unmet_3 * num_women, unmet.pop_4 = unmet_4 * num_women,
   pregnancy.surveyyr.unmet.pop_1 = pregnancy.surveyyr.unmet_1 * num_women, 
   pregnancy.surveyyr.unmet.pop_2 = pregnancy.surveyyr.unmet_2 * num_women, 
   pregnancy.surveyyr.unmet.pop_3 = pregnancy.surveyyr.unmet_3 * num_women, 
   pregnancy.surveyyr.unmet.pop_4 = pregnancy.surveyyr.unmet_4 * num_women,
   unmet.knowledge.pop = unmet.knowledge * num_women, unmet.cost.pop = unmet.cost * num_women, unmet.access.pop = unmet.access * num_women, 
   unmet.health.risks.pop = unmet.health.risks * num_women, unmet.fecundity.pop = unmet.fecundity * num_women, 
   unmet.recent.birth.pop = unmet.recent.birth * num_women, unmet.opposition.pop = unmet.opposition * num_women, 
   unmet.sexual.activity.pop = unmet.sexual.activity * num_women, unmet.other.pop = unmet.other * num_women,  
   reason.addressable.pop = reason.addressable * num_women, reason.notaddressable.pop = reason.notaddressable * num_women)
}

dhs.N <- variable.recode(dhs = dhs.svy.N, agestart_edu = 6, UNpop = c(9174, 7725, 6683, 5805, 4872, 3942, 3161))

View(dplyr::select(filter(dhs.N,discrepancy==1), unmet1, v626a, v312, v605, v225, v213, v625a, v502, v525, v528, wantedlast))

# ----------------------------------------------------------------------------- #
# -- Set up the survey design object -- #

# for design variables look at: 
# ~\Dropbox (IDM)\SmallAreaEstimationForFP\LiteratureReview\U5MR Supplement 2017-11-12.pdf
# Table 2 provides variables

svy.setup <- function(dhs) {
    my.svydesign <- svydesign(id= ~v001,
                              strata=~strata,nest=T,
                              weights= ~wt, data=dhs)}

my.svydesign.N <- svy.setup(dhs = dhs.N)

# ----------------------------------------------------------------------------- #
# -- Timing of education, marriage, and birth analysis -- #

# expected percentages for women who have not yet had a child by that age
dhs.N.edu <- dhs.N %>%
  gather("agevar", "stopedu", starts_with("stopedu")) %>%
  separate(agevar, c('agevar', 'ageval'), sep = "_") %>% select(-agevar) %>% mutate(ageval = as.integer(ageval))

edu.models.null <- function(dhs.edu){
  model <- svyglm(stopedu~urbanrural, design=svy.setup(dhs = dhs.edu), family=quasibinomial)
  pred <- predict(model, newdata = data.frame("urbanrural" = c("1","2")), type = "response")
  pred.DF = as.data.frame(pred) %>% mutate(SE = NULL, urbanrural = rownames(.), age = mean(dhs.edu$ageval, na.rm = T))
  return(pred.DF)
}

datalist = list()
for (i in 1:33){ 
  a <- edu.models.null(dhs.edu = filter(dhs.N.edu, agefirstbirth > (i+11) & ageval == (i+11)))
  datalist[[i]] <- a
}
predictions.N2 = do.call(rbind, datalist)
rm(a, i, datalist)

# expected percentages for women who had first child at that age
datalist = list()
for (i in 1:33){ 
  a <- edu.models.null(dhs.edu = filter(dhs.N.edu, agefirstbirth == (i+11) & ageval == (i+11)))
  datalist[[i]] <- a
}
predictions.N1 = do.call(rbind, datalist)
rm(a, i, datalist)


# combine both of above
predictions.tot <- predictions.N1 %>% rename(edu.birth = response) %>% 
  left_join(predictions.N2, by = c("age", "urbanrural")) %>% rename(edu.nobirth = response) %>%
  mutate(difference = edu.birth - edu.nobirth, urbanrural = ifelse(urbanrural == "1", "Urban", "Rural")) %>%
  gather("var", "val", -age, -urbanrural)
# predictions.tot <- predictions.N %>% select(age = agefirstbirth, urbanrural, edu.birth = model5) %>% 
#   mutate(edu.birth = as.numeric(edu.birth), urbanrural = as.character(urbanrural)) %>%
#   left_join(predictions.N2, by = c("age", "urbanrural")) %>% rename(edu.nobirth = response) %>%
#   mutate(difference = edu.birth - edu.nobirth, urbanrural = ifelse(urbanrural == "1", "Urban", "Rural")) %>%
#   gather("var", "val", -age, -urbanrural)
ggplot(predictions.tot, aes(x = age, y = val, color = var)) + geom_smooth(method = "loess", se = F) +
  theme_classic() + scale_color_brewer(palette = "Dark2", labels = c("Difference", "First birth", "No birth")) +
  ylab("Percent of women stopping school with 1 year of age") + xlab("Age") + theme(legend.title=element_blank()) +
  facet_wrap(~ predictions.tot$urbanrural, ncol = 1) +
  scale_x_continuous(limits=c(10, 30), breaks = 10:30)

# ----------------------------------------------------------------------------- #
# -- Subnational by age and parity analysis with reasons for unmet need -- #

svy.subnational <- function(dhs.address){
  dhs.addressable <- dhs.address %>% mutate(reason.addressable = ifelse(unmet_1 != 1, NA, reason.addressable), # only want to look at reasons for those with unmet need
                                            reason.notaddressable = ifelse(unmet_1 != 1, NA, reason.notaddressable),
                                            reason.addressable.pop = ifelse(unmet_1 != 1, NA, reason.addressable.pop),
                                            reason.notaddressable.pop = ifelse(unmet_1 != 1, NA, reason.notaddressable.pop))
  my.svydesign <- svy.setup(dhs = dhs.addressable)
  results.subnational <- svyby(~unmet.pop_1, by = ~DHSREGEN, design = my.svydesign, FUN = svytotal, na.rm = T, na.rm.all = T) %>% rename(se.unmet.pop_1 = se) %>%
    left_join(svyby(~unmet_1, by = ~DHSREGEN, design = my.svydesign, FUN = svymean, na.rm = T, na.rm.all = T), by = "DHSREGEN") %>% rename(se.unmet_1 = se) %>% 
    left_join(svyby(~reason.addressable, by = ~DHSREGEN, design = my.svydesign, FUN = svymean, na.rm = T, na.rm.all = T), by = "DHSREGEN") %>% rename(se.reason.addressable = se) %>% 
    left_join(svyby(~reason.notaddressable, by = ~DHSREGEN, design = my.svydesign, FUN = svymean, na.rm = T, na.rm.all = T), by = "DHSREGEN") %>% rename(se.reason.notaddressable = se) %>% 
    left_join(svyby(~reason.addressable.pop, by = ~DHSREGEN, design = my.svydesign, FUN = svytotal, na.rm = T, na.rm.all = T), by = "DHSREGEN") %>% rename(se.reason.addressable.pop = se) %>% 
    left_join(svyby(~reason.notaddressable.pop, by = ~DHSREGEN, design = my.svydesign, FUN = svytotal, na.rm = T, na.rm.all = T), by = "DHSREGEN") %>% rename(se.reason.notaddressable.pop = se) 
}
results.subnational.N <- svy.subnational(dhs.address = dhs.N)
results.spatial.N <- shape2.N %>% left_join(results.subnational.N, by = "DHSREGEN")

# plot of unmet need map
results.spatial.N %>% ggplot() +
  geom_sf(aes(fill = unmet_1)) +
  scale_fill_distiller("Percent unmet need", palette = "OrRd", direction = 1) +
  theme_bw() 
results.spatial.N %>% ggplot() +
  geom_sf(aes(fill = unmet.pop_1)) +
  scale_fill_distiller("Population (1,000s) with unmet need", palette = "PuRd", direction = 1) +
  theme_bw()

# plot of addressable reasons map
results.spatial.N %>% ggplot() +
  geom_sf(aes(fill = reason.addressable)) +
  scale_fill_distiller("Percent reasons addressable", palette = "YlGn", direction = 1) +
  theme_bw()
results.spatial.N %>% ggplot() +
  geom_sf(aes(fill = reason.addressable.pop)) +
  scale_fill_distiller("Total with addressable reasons", palette = "Oranges", direction = 1) +
  theme_bw()

# plot of non-addressable reasons map
results.spatial.N %>% ggplot() +
  geom_sf(aes(fill = reason.notaddressable)) +
  scale_fill_distiller("Percent reasons not addressable", palette = "YlGn", direction = -1) +
  theme_bw()
results.spatial.N %>% ggplot() +
  geom_sf(aes(fill = reason.notaddressable.pop)) +
  scale_fill_distiller("Total with non-addressable", palette = "Oranges", direction = 1) +
  theme_bw()

# -- comparison of Lagos and Kaduna -- #

# unmet need by age, parity, and state
results.subnational.demo.address <-   dhs.N %>% mutate(unmet.knowledge.pop = ifelse(unmet_1 != 1, NA, unmet.knowledge.pop), # only want to look at reasons for those with unmet need
                                                       unmet.cost.pop = ifelse(unmet_1 != 1, NA, unmet.cost.pop),
                                                       unmet.access.pop = ifelse(unmet_1 != 1, NA, unmet.access.pop),
                                                       unmet.health.risks.pop = ifelse(unmet_1 != 1, NA, unmet.health.risks.pop),
                                                       unmet.fecundity.pop = ifelse(unmet_1 != 1, NA, unmet.fecundity.pop),
                                                       unmet.recent.birth.pop = ifelse(unmet_1 != 1, NA, unmet.recent.birth.pop),
                                                       unmet.opposition.pop = ifelse(unmet_1 != 1, NA, unmet.opposition.pop),
                                                       unmet.sexual.activity.pop = ifelse(unmet_1 != 1, NA, unmet.sexual.activity.pop),
                                                       unmet.other.pop = ifelse(unmet_1 != 1, NA, unmet.other.pop))
my.svydesign.subnat <- svy.setup(dhs = results.subnational.demo.address)

results.subnational.demo <- svyby(~unmet.pop_1, by = ~(DHSREGEN+age_cat+parity_cat), design = my.svydesign.subnat, FUN = svytotal, na.rm = T, na.rm.all = T) %>% rename(se.unmet.pop_1 = se) %>%
  left_join(svyby(~unmet_1, by = ~(DHSREGEN+age_cat+parity_cat), design = my.svydesign.subnat, FUN = svymean, na.rm = T, na.rm.all = T), by = c("DHSREGEN","age_cat", "parity_cat")) %>% rename(se.unmet_1 = se) %>%
  left_join(svyby(~unmet.knowledge.pop, by = ~(DHSREGEN+age_cat+parity_cat), design = my.svydesign.subnat, FUN = svytotal, na.rm = T, na.rm.all = T), by = c("DHSREGEN","age_cat", "parity_cat")) %>% rename(se.unmet.knowledge.pop = se) %>%
  left_join(svyby(~unmet.cost.pop, by = ~(DHSREGEN+age_cat+parity_cat), design = my.svydesign.subnat, FUN = svytotal, na.rm = T, na.rm.all = T), by = c("DHSREGEN","age_cat", "parity_cat")) %>% rename(se.unmet.cost.pop = se) %>%
  left_join(svyby(~unmet.access.pop, by = ~(DHSREGEN+age_cat+parity_cat), design = my.svydesign.subnat, FUN = svytotal, na.rm = T, na.rm.all = T), by = c("DHSREGEN","age_cat", "parity_cat")) %>% rename(se.unmet.access.pop = se) %>%
  left_join(svyby(~unmet.health.risks.pop, by = ~(DHSREGEN+age_cat+parity_cat), design = my.svydesign.subnat, FUN = svytotal, na.rm = T, na.rm.all = T), by = c("DHSREGEN","age_cat", "parity_cat")) %>% rename(se.unmet.health.risks.pop = se) %>%
  left_join(svyby(~unmet.fecundity.pop, by = ~(DHSREGEN+age_cat+parity_cat), design = my.svydesign.subnat, FUN = svytotal, na.rm = T, na.rm.all = T), by = c("DHSREGEN","age_cat", "parity_cat")) %>% rename(se.unmet.fecundity.pop = se) %>%
  left_join(svyby(~unmet.recent.birth.pop, by = ~(DHSREGEN+age_cat+parity_cat), design = my.svydesign.subnat, FUN = svytotal, na.rm = T, na.rm.all = T), by = c("DHSREGEN","age_cat", "parity_cat")) %>% rename(se.unmet.recent.birth.pop = se) %>%
  left_join(svyby(~unmet.opposition.pop, by = ~(DHSREGEN+age_cat+parity_cat), design = my.svydesign.subnat, FUN = svytotal, na.rm = T, na.rm.all = T), by = c("DHSREGEN","age_cat", "parity_cat")) %>% rename(se.unmet.opposition.pop = se) %>%
  left_join(svyby(~unmet.sexual.activity.pop, by = ~(DHSREGEN+age_cat+parity_cat), design = my.svydesign.subnat, FUN = svytotal, na.rm = T, na.rm.all = T), by = c("DHSREGEN","age_cat", "parity_cat")) %>% rename(se.unmet.sexual.activity.pop = se) %>%
  left_join(svyby(~unmet.other.pop, by = ~(DHSREGEN+age_cat+parity_cat), design = my.svydesign.subnat, FUN = svytotal, na.rm = T, na.rm.all = T), by = c("DHSREGEN","age_cat", "parity_cat")) %>% rename(se.unmet.other.pop = se) 

# figures: unmet need percent and total  
results.spatial.N %>% filter(DHSREGEN == "Lagos" | DHSREGEN == "Kaduna") %>% ggplot() +
  geom_sf(aes(fill = unmet_1), show.legend = F) +
  scale_fill_distiller("Percent unmet need", palette = "OrRd", direction = 1,
                       limits = c(min(results.spatial.N$unmet_1), max(results.spatial.N$unmet_1))) + 
  ggtitle('Percent with unmet need') + theme_bw()
results.spatial.N %>% filter(DHSREGEN == "Lagos" | DHSREGEN == "Kaduna") %>% ggplot() +
  geom_sf(aes(fill = unmet.pop_1)) +
  scale_fill_distiller("Percent unmet need", palette = "PuRd", direction = 1,
                       limits = c(min(results.spatial.N$unmet.pop_1), max(results.spatial.N$unmet.pop_1))) + 
  ggtitle('Number with unmet need') + theme_bw()
          
# figures: unmet need by age and parity percent and total
ggplot(filter(results.subnational.demo, DHSREGEN == "Lagos" | DHSREGEN == "Kaduna"), 
       aes(y = parity_cat, x = age_cat, height = unmet_1, group = interaction(DHSREGEN, parity_cat), fill = DHSREGEN)) + 
  geom_density_ridges(stat = "identity", scale = 1, alpha = 0.5) +
  theme_ridges(center = TRUE)+
  ylab("Parity") + xlab("Age") + 
  ggtitle('Percent with unmet need') +
  scale_fill_brewer('State', palette = "Dark2")
ggplot(filter(results.subnational.demo, DHSREGEN == "Lagos" | DHSREGEN == "Kaduna"), 
       aes(y = parity_cat, x = age_cat, height = unmet.pop_1, group = interaction(DHSREGEN, parity_cat), fill = DHSREGEN)) + 
  geom_density_ridges(stat = "identity", scale = 1, alpha = 0.5) +
  theme_ridges(center = TRUE)+
  ylab("Parity") + xlab("Age") + 
  ggtitle('Women with unmet need') +
  scale_fill_brewer('State', palette = "Dark2")

# figures: reasons for non-use
results.subnational.reasons <- results.subnational.demo %>%
  gather(variable, value, -DHSREGEN, -age_cat, -parity_cat, -unmet.pop_1, -se.unmet.pop_1, -unmet_1, -se.unmet_1) %>% 
  separate(variable, into = c("variable", "reason"), sep = "unmet.") %>%
  mutate(variable = ifelse(variable == "", "total", "se")) %>% spread(variable, value) %>% 
  mutate(reason = gsub(".pop", "", reason))

grid.arrange(
  ggplot(filter(results.subnational.reasons, DHSREGEN == "Lagos"), aes(x = age_cat, y = total, fill=reason)) + 
    geom_bar(stat="identity") + theme_bw() + ggtitle('Lagos') +
    ylab("Number of women") + xlab("Age") +
    theme(legend.title=element_blank()) + 
    scale_fill_brewer(palette = "Set2"),
    
  ggplot(filter(results.subnational.reasons, DHSREGEN == "Kaduna" & reason != "other"), aes(x = age_cat, y = total, fill=reason)) + 
    geom_bar(stat="identity") + theme_bw() + ggtitle('Kaduna') +
    ylab("Number of women") + xlab("Age") +
    theme(legend.title=element_blank()) + 
    scale_fill_brewer(palette = "Set2"),
  
  ggplot(filter(results.subnational.reasons, DHSREGEN == "Lagos"), aes(x = parity_cat, y = total, fill=reason)) + 
    geom_bar(stat="identity") + theme_bw() + ggtitle('Lagos') +
    ylab("Number of women") + xlab("Parity") +
    theme(legend.title=element_blank()) + 
    scale_fill_brewer(palette = "Set2"),
  
  ggplot(filter(results.subnational.reasons, DHSREGEN == "Kaduna" & reason != "other"), aes(x = parity_cat, y = total, fill=reason)) + 
    geom_bar(stat="identity") + theme_bw() + ggtitle('Kaduna') +
    ylab("Number of women") + xlab("Parity") +
    theme(legend.title=element_blank()) + 
    scale_fill_brewer(palette = "Set2"),
  
  ncol = 2)

# -- addressable reasons for non use -- #

reasons.address <- dhs.N %>% mutate(reason.addressable = ifelse(unmet_1 != 1, NA, reason.addressable), # only want to look at reasons for those with unmet need
  reason.notaddressable = ifelse(unmet_1 != 1, NA, reason.notaddressable),
  reason.addressable.pop = ifelse(unmet_1 != 1, NA, reason.addressable.pop),
  reason.notaddressable.pop = ifelse(unmet_1 != 1, NA, reason.notaddressable.pop))
my.svydesign.address <- svy.setup(dhs = reasons.address)

results.address <- 
  svyby(~reason.addressable.pop, by = ~(DHSREGEN+age_cat+parity_cat), design = my.svydesign.address, FUN = svytotal, na.rm = T, na.rm.all = T) %>% rename(se.reason.addressable.pop = se) %>%
  left_join(svyby(~reason.notaddressable.pop, by = ~(DHSREGEN+age_cat+parity_cat), design = my.svydesign.address, FUN = svytotal, na.rm = T, na.rm.all = T), by = c("DHSREGEN","age_cat", "parity_cat")) %>% rename(se.reason.notaddressable.pop = se) %>% 
  gather(variable, value, -DHSREGEN, -age_cat, -parity_cat) %>% 
  separate(variable, into = c("variable", "reason"), sep = "reason.") %>%
  mutate(variable = ifelse(variable == "", "total", "se")) %>% spread(variable, value) %>% 
  mutate(reason = gsub(".pop", "", reason))

# figures: adrressable reasons
grid.arrange(
  ggplot(filter(results.address, DHSREGEN == "Lagos"), aes(x = age_cat, y = total, fill=reason)) + 
    geom_bar(stat="identity") + theme_bw() + ggtitle('Lagos') +
    ylab("Number of women") + xlab("Age") +
    theme(legend.title=element_blank()) + 
    scale_fill_brewer(palette = "Set2"),
  
  ggplot(filter(results.address, DHSREGEN == "Kaduna"), aes(x = age_cat, y = total, fill=reason)) + 
    geom_bar(stat="identity") + theme_bw() + ggtitle('Kaduna') +
    ylab("Number of women") + xlab("Age") +
    theme(legend.title=element_blank()) + 
    scale_fill_brewer(palette = "Set2"),
  
  ggplot(filter(results.address, DHSREGEN == "Lagos"), aes(x = parity_cat, y = total, fill=reason)) + 
    geom_bar(stat="identity") + theme_bw() + ggtitle('Lagos') +
    ylab("Number of women") + xlab("Parity") +
    theme(legend.title=element_blank()) + 
    scale_fill_brewer(palette = "Set2"),
  
  ggplot(filter(results.address, DHSREGEN == "Kaduna"), aes(x = parity_cat, y = total, fill=reason)) + 
    geom_bar(stat="identity") + theme_bw() + ggtitle('Kaduna') +
    ylab("Number of women") + xlab("Parity") +
    theme(legend.title=element_blank()) + 
    scale_fill_brewer(palette = "Set2"),
  
  ncol = 2)

# figures: map addressable reasons
results.spatial.N %>% filter(DHSREGEN == "Lagos" | DHSREGEN == "Kaduna") %>%
  ggplot() +
  geom_sf(aes(fill = reason.addressable.pop)) +
  scale_fill_distiller("Total with addressable", palette = "Oranges", direction = 1,
                       limits = c(min(results.spatial.N$reason.addressable.pop), max(results.spatial.N$reason.addressable.pop))) +
  theme_bw()

# -- Kmeans on reasons for non-use and addressability-- #
 
# prep data
kmeans.svy.data <- dhs.N %>% mutate(unmet.knowledge = ifelse(unmet_1 != 1, NA, unmet.knowledge), # only want to look at reasons for those with unmet need
                                    unmet.cost = ifelse(unmet_1 != 1, NA, unmet.cost),
                                    unmet.access = ifelse(unmet_1 != 1, NA, unmet.access),
                                    unmet.health.risks = ifelse(unmet_1 != 1, NA, unmet.health.risks),
                                    unmet.fecundity = ifelse(unmet_1 != 1, NA, unmet.fecundity),
                                    unmet.recent.birth = ifelse(unmet_1 != 1, NA, unmet.recent.birth),
                                    unmet.opposition = ifelse(unmet_1 != 1, NA, unmet.opposition),
                                    unmet.sexual.activity = ifelse(unmet_1 != 1, NA, unmet.sexual.activity),
                                    unmet.other = ifelse(unmet_1 != 1, NA, unmet.other),
                                    reason.addressable = ifelse(unmet_1 != 1, NA, reason.addressable),
                                    reason.notaddressable = ifelse(unmet_1 != 1, NA, reason.notaddressable))
my.svydesign.kmeans <- svy.setup(dhs = kmeans.svy.data)

kmeans.svy.data.table <- svyby(~unmet_1, by = ~(DHSREGEN), design = my.svydesign.kmeans, FUN = svymean, na.rm = T, na.rm.all = T) %>% rename(se.unmet_1_p = se, unmet_1_p = unmet_1) %>%
  left_join(svyby(~unmet_1, by = ~(DHSREGEN), design = my.svydesign.kmeans, FUN = svytotal, na.rm = T, na.rm.all = T), by = "DHSREGEN") %>% rename(se.unmet_1 = se) %>%
  left_join(svyby(~unmet.knowledge, by = ~(DHSREGEN), design = my.svydesign.kmeans, FUN = svytotal, na.rm = T, na.rm.all = T), by = "DHSREGEN") %>% rename(se.unmet.knowledge = se) %>%
  left_join(svyby(~unmet.cost, by = ~(DHSREGEN), design = my.svydesign.kmeans, FUN = svytotal, na.rm = T, na.rm.all = T), by = "DHSREGEN") %>% rename(se.unmet.cost = se) %>%
  left_join(svyby(~unmet.access, by = ~(DHSREGEN), design = my.svydesign.kmeans, FUN = svytotal, na.rm = T, na.rm.all = T), by = "DHSREGEN") %>% rename(se.unmet.access = se) %>%
  left_join(svyby(~unmet.health.risks, by = ~(DHSREGEN), design = my.svydesign.kmeans, FUN = svytotal, na.rm = T, na.rm.all = T), by = "DHSREGEN") %>% rename(se.unmet.health.risks = se) %>%
  left_join(svyby(~unmet.fecundity, by = ~(DHSREGEN), design = my.svydesign.kmeans, FUN = svytotal, na.rm = T, na.rm.all = T), by = "DHSREGEN") %>% rename(se.unmet.fecundity = se) %>%
  left_join(svyby(~unmet.recent.birth, by = ~(DHSREGEN), design = my.svydesign.kmeans, FUN = svytotal, na.rm = T, na.rm.all = T), by = "DHSREGEN") %>% rename(se.unmet.recent.birth = se) %>%
  left_join(svyby(~unmet.opposition, by = ~(DHSREGEN), design = my.svydesign.kmeans, FUN = svytotal, na.rm = T, na.rm.all = T), by = "DHSREGEN") %>% rename(se.unmet.opposition = se) %>%
  left_join(svyby(~unmet.sexual.activity, by = ~(DHSREGEN), design = my.svydesign.kmeans, FUN = svytotal, na.rm = T, na.rm.all = T), by = "DHSREGEN") %>% rename(se.unmet.sexual.activity = se) %>%
  left_join(svyby(~unmet.other, by = ~(DHSREGEN), design = my.svydesign.kmeans, FUN = svytotal, na.rm = T, na.rm.all = T), by = "DHSREGEN") %>% rename(se.unmet.other = se) %>%
  left_join(svyby(~reason.addressable, by = ~(DHSREGEN), design = my.svydesign.kmeans, FUN = svytotal, na.rm = T, na.rm.all = T), by = "DHSREGEN") %>% rename(se.reason.addressable = se) %>%
  left_join(svyby(~reason.notaddressable, by = ~(DHSREGEN), design = my.svydesign.kmeans, FUN = svytotal, na.rm = T, na.rm.all = T), by = "DHSREGEN") %>% rename(se.reason.notaddressable = se) 
  
results.wide.reasons <- kmeans.svy.data.table %>% select(-starts_with("se.")) %>%
  mutate_at(vars(-DHSREGEN, -unmet_1_p, -unmet_1), funs(./unmet_1)) %>% mutate_all(list(~replace_na(., 0))) %>% select(-unmet_1)
region <- results.wide.reasons$DHSREGEN  
results.wide.reasons.nolabel <- results.wide.reasons %>% select(-DHSREGEN)

# try kmeans with varying number of clusters
k2 <- kmeans(results.wide.reasons.nolabel, centers = 2, nstart = 25)
k3 <- kmeans(results.wide.reasons.nolabel, centers = 3, nstart = 25)
k4 <- kmeans(results.wide.reasons.nolabel, centers = 4, nstart = 25)
k5 <- kmeans(results.wide.reasons.nolabel, centers = 5, nstart = 25)
k6 <- kmeans(results.wide.reasons.nolabel, centers = 6, nstart = 25)

# plots to compare
grid.arrange(
  fviz_cluster(k2, geom = "point", data = results.wide.reasons.nolabel) + ggtitle("k = 2"),
  fviz_cluster(k3, geom = "point",  data = results.wide.reasons.nolabel) + ggtitle("k = 3"),
  fviz_cluster(k4, geom = "point",  data = results.wide.reasons.nolabel) + ggtitle("k = 4"),
  fviz_cluster(k5, geom = "point",  data = results.wide.reasons.nolabel) + ggtitle("k = 5"),
  fviz_cluster(k6, geom = "point",  data = results.wide.reasons.nolabel) + ggtitle("k = 6"),
  nrow = 2)

# elbow method
set.seed(123)
fviz_nbclust(results.wide.reasons.nolabel, 
             kmeans, 
             method = "wss")

# Compute k-means clustering with 5 clusters
rownames(results.wide.reasons.nolabel) <- region
fviz_cluster(k5, data = results.wide.reasons.nolabel)
# Plot data
kmeans5 <- results.wide.reasons %>%
  mutate(kc_label = k5$cluster)%>%
  gather(key, value, -DHSREGEN, -kc_label) %>%
  arrange(DHSREGEN) %>%
  as_tibble() 
kmeans5 %>%
  ggplot(aes(x = key, y = value, group = DHSREGEN)) +
  geom_point() +
  facet_wrap(~ kc_label) +
  labs(x = NULL,
       y = 'Percent') +
  theme_bw()
# map data
kmeans5.spatial <- shape2.N %>% left_join(kmeans5, by = "DHSREGEN")
kmeans5.spatial %>% 
  ggplot() +
  geom_sf(aes(fill = as.factor(kc_label))) +
  theme_bw() + scale_fill_brewer('Cluster', palette = "Dark2")

# repeat with 4 clusters
# Plot data
kmeans4 <- results.wide.reasons %>%
  mutate(kc_label = k4$cluster)%>%
  gather(key, value, -DHSREGEN, -kc_label) %>%
  arrange(DHSREGEN) %>%
  as_tibble() 
# map data
kmeans4.spatial <- shape2.N %>% left_join(kmeans4, by = "DHSREGEN")
kmeans4.spatial %>% 
  ggplot() +
  geom_sf(aes(fill = as.factor(kc_label))) +
  theme_bw() + scale_fill_brewer('Cluster', palette = "Dark2")

# -- Summarization by age and parity analysis with unmet need definition -- #

# percentage in each age group of women with unmet need
with(filter(dhs.R, unmet_1==1), prop.table(svytable(~age_cat, my.svydesign.R))) 
with(filter(dhs.N, unmet_1==1), prop.table(svytable(~age_cat, my.svydesign.N))) 

View(svytable( ~parity_cat + age_cat, my.svydesign.N)) # survey totals by age and parity group
View(svyby(~svy_num_women, by = ~(age_cat+parity_cat), design = my.svydesign.N, FUN = svytotal)) # survey totals by age and parity group, ensure same as above

# all population size results with survey weights
svy.results.age.parity <- function(my.svydesign){
  results.age.parity <- svyby(~svy_num_women + num_women, by = ~(age_cat+parity_cat), design = my.svydesign, FUN = svytotal, na.rm = T, na.rm.all = T) %>% 
    left_join(svyby(~unmet.pop_1, by = ~(age_cat+parity_cat), design = my.svydesign, FUN = svytotal, na.rm = T, na.rm.all = T), by = c("age_cat", "parity_cat")) %>% rename(se.unmet.pop_1 = se) %>%
    left_join(svyby(~unmet.pop_2, by = ~(age_cat+parity_cat), design = my.svydesign, FUN = svytotal, na.rm = T, na.rm.all = T), by = c("age_cat", "parity_cat")) %>% rename(se.unmet.pop_2 = se) %>%
    left_join(svyby(~unmet.pop_3, by = ~(age_cat+parity_cat), design = my.svydesign, FUN = svytotal, na.rm = T, na.rm.all = T), by = c("age_cat", "parity_cat")) %>% rename(se.unmet.pop_3 = se) %>%
    left_join(svyby(~unmet.pop_4, by = ~(age_cat+parity_cat), design = my.svydesign, FUN = svytotal, na.rm = T, na.rm.all = T), by = c("age_cat", "parity_cat")) %>% rename(se.unmet.pop_4 = se) %>%
    left_join(svyby(~pregnancy.surveyyr.unmet.pop_1, by = ~(age_cat+parity_cat), design = my.svydesign, FUN = svytotal, na.rm = T, na.rm.all = T), by = c("age_cat", "parity_cat")) %>% rename(se.pregnancy.surveyyr.unmet.pop_1 = se) %>%
    left_join(svyby(~pregnancy.surveyyr.unmet.pop_2, by = ~(age_cat+parity_cat), design = my.svydesign, FUN = svytotal, na.rm = T, na.rm.all = T), by = c("age_cat", "parity_cat")) %>% rename(se.pregnancy.surveyyr.unmet.pop_2 = se) %>%
    left_join(svyby(~pregnancy.surveyyr.unmet.pop_3, by = ~(age_cat+parity_cat), design = my.svydesign, FUN = svytotal, na.rm = T, na.rm.all = T), by = c("age_cat", "parity_cat")) %>% rename(se.pregnancy.surveyyr.unmet.pop_3 = se) %>%
    left_join(svyby(~pregnancy.surveyyr.unmet.pop_4, by = ~(age_cat+parity_cat), design = my.svydesign, FUN = svytotal, na.rm = T, na.rm.all = T), by = c("age_cat", "parity_cat")) %>% rename(se.pregnancy.surveyyr.unmet.pop_4 = se) %>%
    gather(variable, value, -age_cat, -parity_cat, -num_women, -svy_num_women, -se.num_women, -se.svy_num_women) %>% separate(variable, into = c("variable", "definition"), sep = "_") %>% spread(variable, value) # long format for unmet need definition
}
results.age.parity.N <- svy.results.age.parity(my.svydesign = my.svydesign.N)

# histogram plots

# total women
ggplot(svyby(~num_women, by = ~(age_cat+parity_cat), design = my.svydesign.N, FUN = svytotal, na.rm = T, na.rm.all = T),
       aes(y = parity_cat, x = age_cat, height = num_women, group = parity_cat, fill = parity_cat)) + 
  geom_density_ridges(stat = "identity", scale = 1.5, alpha = 0.5, show.legend = F) +
  theme_ridges(center = TRUE)+
  ylab("Parity") + xlab("Age") +
  ggtitle('Total number of women') +
  scale_fill_brewer(guide = guide_legend(reverse = TRUE), palette = "Blues")

# unmet need by definition
ggplot(results.age.parity.N, aes(y = parity_cat, x = age_cat, height = unmet.pop, group = interaction(rev(definition), parity_cat), fill = definition)) + 
  geom_density_ridges(stat = "identity", scale = 3, alpha = 0.5) +
  theme_ridges(center = TRUE)+
  ylab("Parity") + xlab("Age") + 
  ggtitle('Women with unmet need') +
  scale_fill_brewer('Sexually active within', guide = guide_legend(reverse = TRUE), palette = "BuPu", labels = c("4 weeks", "1 year", "Ever", "N/A"))
ggplot(filter(results.age.parity.N, parity_cat == "0"), aes(y = unmet.pop, x = age_cat, color = factor(definition), group = rev(definition))) + # for 0 parity only
  geom_line(size = 1.5) + 
  ylab("Number of women") + xlab("Age") + 
  ggtitle('Zero parity women with unmet need') +
  theme_classic() + scale_y_continuous(labels = comma) +
  scale_color_brewer('Sexually active within', guide = guide_legend(reverse = TRUE), palette = "Set1", labels = c("4 weeks", "1 year", "Ever", "N/A"))

# pregnancies among unmet need by definition
ggplot(filter(results.age.parity.N, parity_cat != "0"), aes(y = parity_cat, x = age_cat, height = pregnancy.surveyyr.unmet.pop, group = interaction(rev(definition), parity_cat), fill = definition)) + 
  geom_density_ridges(stat = "identity", scale = 2, alpha = 0.5) +
  theme_ridges(center = TRUE)+
  ylab("Parity") + xlab("Age") + 
  ggtitle('Pregnancies among women with unmet need') +
  scale_fill_brewer('Sexually active within', guide = guide_legend(reverse = TRUE), palette = "BuPu", labels = c("4 weeks", "1 year", "Ever", "N/A"))
ggplot(filter(results.age.parity.N, parity_cat == "1"), aes(y = pregnancy.surveyyr.unmet.pop, x = age_cat, color = factor(definition), group = rev(definition))) + # for 0 parity only
  geom_line(size = 1.5) + 
  ylab("Number of pregnancies") + xlab("Age") + 
  ggtitle('Pregnancies among women with unmet need (parity 0 -> 1)') +
  theme_classic() + scale_y_continuous(labels = comma) +
  scale_color_brewer('Sexually active within', guide = guide_legend(reverse = TRUE), palette = "Set1", labels = c("4 weeks", "1 year", "Ever", "N/A"))

# repeat with "addressable" unmet need
svy.results.age.parity.address.unmet <- function(dhs.address){ 
  dhs.addressable <- dhs.address %>% mutate(unmet.pop_1 = ifelse(reason.addressable != 1, 0, unmet.pop_1), # unmet need and pregnancies only for those with addressable reasons
                                            unmet.pop_2 = ifelse(reason.addressable != 1, 0, unmet.pop_2),
                                            unmet.pop_3 = ifelse(reason.addressable != 1, 0, unmet.pop_3),
                                            unmet.pop_4 = ifelse(reason.addressable != 1, 0, unmet.pop_4),
                                            pregnancy.surveyyr.unmet.pop_1 = ifelse(reason.addressable != 1, 0, pregnancy.surveyyr.unmet.pop_1),
                                            pregnancy.surveyyr.unmet.pop_2 = ifelse(reason.addressable != 1, 0, pregnancy.surveyyr.unmet.pop_2),
                                            pregnancy.surveyyr.unmet.pop_3 = ifelse(reason.addressable != 1, 0, pregnancy.surveyyr.unmet.pop_3),
                                            pregnancy.surveyyr.unmet.pop_4 = ifelse(reason.addressable != 1, 0, pregnancy.surveyyr.unmet.pop_4))
  my.svydesign.address.unmet <- svy.setup(dhs = dhs.addressable)
  results.age.parity <- svyby(~svy_num_women + num_women, by = ~(age_cat+parity_cat), design = my.svydesign.address.unmet, FUN = svytotal, na.rm = T, na.rm.all = T) %>% 
    left_join(svyby(~unmet.pop_1, by = ~(age_cat+parity_cat), design = my.svydesign.address.unmet, FUN = svytotal, na.rm = T, na.rm.all = T), by = c("age_cat", "parity_cat")) %>% rename(se.unmet.pop_1 = se) %>%
    left_join(svyby(~unmet.pop_2, by = ~(age_cat+parity_cat), design = my.svydesign.address.unmet, FUN = svytotal, na.rm = T, na.rm.all = T), by = c("age_cat", "parity_cat")) %>% rename(se.unmet.pop_2 = se) %>%
    left_join(svyby(~unmet.pop_3, by = ~(age_cat+parity_cat), design = my.svydesign.address.unmet, FUN = svytotal, na.rm = T, na.rm.all = T), by = c("age_cat", "parity_cat")) %>% rename(se.unmet.pop_3 = se) %>%
    left_join(svyby(~unmet.pop_4, by = ~(age_cat+parity_cat), design = my.svydesign.address.unmet, FUN = svytotal, na.rm = T, na.rm.all = T), by = c("age_cat", "parity_cat")) %>% rename(se.unmet.pop_4 = se) %>%
    left_join(svyby(~pregnancy.surveyyr.unmet.pop_1, by = ~(age_cat+parity_cat), design = my.svydesign.address.unmet, FUN = svytotal, na.rm = T, na.rm.all = T), by = c("age_cat", "parity_cat")) %>% rename(se.pregnancy.surveyyr.unmet.pop_1 = se) %>%
    left_join(svyby(~pregnancy.surveyyr.unmet.pop_2, by = ~(age_cat+parity_cat), design = my.svydesign.address.unmet, FUN = svytotal, na.rm = T, na.rm.all = T), by = c("age_cat", "parity_cat")) %>% rename(se.pregnancy.surveyyr.unmet.pop_2 = se) %>%
    left_join(svyby(~pregnancy.surveyyr.unmet.pop_3, by = ~(age_cat+parity_cat), design = my.svydesign.address.unmet, FUN = svytotal, na.rm = T, na.rm.all = T), by = c("age_cat", "parity_cat")) %>% rename(se.pregnancy.surveyyr.unmet.pop_3 = se) %>%
    left_join(svyby(~pregnancy.surveyyr.unmet.pop_4, by = ~(age_cat+parity_cat), design = my.svydesign.address.unmet, FUN = svytotal, na.rm = T, na.rm.all = T), by = c("age_cat", "parity_cat")) %>% rename(se.pregnancy.surveyyr.unmet.pop_4 = se) %>%
    gather(variable, value, -age_cat, -parity_cat, -num_women, -svy_num_women, -se.num_women, -se.svy_num_women) %>% separate(variable, into = c("variable", "definition"), sep = "_") %>% spread(variable, value) # long format for unmet need definition
}
results.age.parity.N.address <- svy.results.age.parity.address.unmet(dhs.address = dhs.N)

# histogram plots

# unmet need by definition
ggplot(filter(results.age.parity.N.address, parity_cat == "0"), aes(y = unmet.pop, x = age_cat, color = factor(definition), group = rev(definition))) + # for 0 parity only
  geom_line(size = 1.5) + 
  ylab("Number of women") + xlab("Age") + 
  ggtitle('Zero parity women with unmet need') +
  theme_classic() + scale_y_continuous(labels = comma) +
  scale_color_brewer('Sexually active within', guide = guide_legend(reverse = TRUE), palette = "Set1", labels = c("4 weeks", "1 year", "Ever", "N/A"))

# pregnancies among unmet need by definition
ggplot(filter(results.age.parity.N.address, parity_cat == "1"), aes(y = pregnancy.surveyyr.unmet.pop, x = age_cat, color = factor(definition), group = rev(definition))) + # for 0 parity only
  geom_line(size = 1.5) + 
  ylab("Number of pregnancies") + xlab("Age") + 
  ggtitle('Pregnancies among women with unmet need (parity 0 -> 1)') +
  theme_classic() + scale_y_continuous(labels = comma) +
  scale_color_brewer('Sexually active within', guide = guide_legend(reverse = TRUE), palette = "Set1", labels = c("4 weeks", "1 year", "Ever", "N/A"))
  
