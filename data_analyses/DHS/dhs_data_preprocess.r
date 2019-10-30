setwd("C:/Users/jochen/Desktop/FP_data/NGIR6ADT")
library(foreign)
library(plyr)
library(dplyr)
library(forcats)
library(stringr)
dat <- read.dta("NGIR6AFL.DTA")


################# some checking of the data ###################
check_contraceptive_missing <- function(dat){
  # v017: Century month code (CMC) for the first month of the calendar
  cmc_first_month <- dat[,'v017']
  # v008: Century month code of date of interview
  cmc_interview <- dat[,'v008']
  # check if contraceptive missing for some months
  contraceptive_str <- gsub(" ", "", dat[,'vcal_1'])
  return(sum(nchar(contraceptive_str)!=(cmc_interview-cmc_first_month+1)))
}
check_contraceptive_missing(dat)


############### helper function to get a list of parity for each individual ######################
get_parity_list <- function(dat, cmc_first_month, cmc_interview, record_length){
  cumulative_parity <- 0
  num <- c()
  parity_list <- c()
  for(j in 1:20){
    # b3_i is the cmc of birth of the ith children for this individual
    var_cmc_birth <- ifelse(j <= 9, paste0('b3_', paste0('0',j)), paste0('b3_', j))
    cmc_birth <- as.numeric(dat[var_cmc_birth])
    if(!is.na(cmc_birth)){
      cumulative_parity <- cumulative_parity+1
      if(cmc_birth >= cmc_first_month){
        num <- c(num, cmc_birth)
      }
    }
  }
  if(length(num)==0){
    parity_list <- rep(cumulative_parity, record_length)
  }  else{
    previous_t <- cmc_interview+1
    for(t in 1:length(num)){
      parity_list <- c(parity_list, rep(cumulative_parity,previous_t-num[t]))
      previous_t <- num[t]
      cumulative_parity <- cumulative_parity-1
    }
    parity_list <- c(parity_list, rep(cumulative_parity,previous_t-cmc_first_month)) 
  }
  return(parity_list)
}


##################### extract and write dataframe to csv #####################
df <- data.frame(matrix(ncol = 7, nrow = 0))
colnames(df) <- c("caseid", "contraceptive", "transition", "transition_reason", "cmc", "age", "parity")

write.table( df,  
             file="NGIR6AFL_extract.csv", 
             append = F, 
             sep=',', 
             row.names=F)

for(i in 1:nrow(dat)){
  # v017: Century month code for the first month of the calendar
  cmc_first_month <- dat[i,'v017']
  # V008: Century month code of date of interview
  cmc_interview <- dat[i,'v008']
  # get a list of cmc where this individual has contraceptive records
  cmc_list <- seq(cmc_interview,cmc_first_month)
  record_length <- length(cmc_list)
  # get contraceptive list
  contraceptive_str <- gsub(" ", "", dat[i,'vcal_1'])
  contraceptive_list <- strsplit(contraceptive_str,'')[[1]]
  # get transition list
  transition_list <- c(paste0(contraceptive_list[2:record_length],'_', contraceptive_list[1:(record_length-1)]),'None')
  # get transition reason list
  temp_vcal_2 <- paste0(dat[i,'vcal_2'],strrep(" ",80 - nchar(dat[i,'vcal_2'])))
  empty_month_count <- str_count(dat$vcal_1[i], " ")
  transition_reason_list <- strsplit(substr(temp_vcal_2,empty_month_count+1,nchar(temp_vcal_2)),'')[[1]]
  # get age list
  age_list <- floor((cmc_list-dat[i,'v011'])/12)
  # get parity list
  parity_list <- get_parity_list(dat[i,], cmc_first_month, cmc_interview, record_length)
  
  df_temp <- cbind.data.frame(rep(dat[i,'caseid'],record_length),
                              contraceptive_list,
                              transition_list,
                              transition_reason_list,
                              cmc_list,
                              age_list,
                              parity_list)
  colnames(df_temp) <- c("caseid", "contraceptive", "transition", "transition_reason", "cmc", "age", "parity")
  write.table( df_temp,
               file="NGIR6AFL_extract.csv",
               append = T,
               sep=',',
               row.names=F,
               col.names=F )
}

################### load data back and staar some analysis ###################
dat <- read.csv("NGIR6AFL_extract.csv")
dat_sub <- cbind.data.frame(as.character(dat$transition[which(dat$transition_reason!=' ')-1]),as.character(dat$transition_reason[which(dat$transition_reason!=' ')]))
colnames(dat_sub) <- c("transition","transition_reason")
dat_sub['from_method'] <- as.factor(substr(dat_sub$transition,1,1))
dat_sub['to_method'] <- as.factor(substr(dat_sub$transition,3,3))
dat_sub$from_method <-
  mapvalues(dat_sub$from_method,
            from=c("1","2","3","4","5","6","8","9","F","K","L","M","N","W"),
            to=c("Pill","IUD","Injct","Diaph","Cond","FSter","Rhyth","Withd","Foam","Unknw","Lact","OModr","Impla","OTrad"))
dat_sub$to_method <- 
  mapvalues(dat_sub$to_method,
            from=c("0","1","2","3","5","6","8","9","C","K","M","N","P","T","W"),
            to=c("None","Pill","IUD","Injct","Cond","FSter","Rhyth","Withd","FCond","Foam","Unknw","OModr","Impla","Term","OTrad"))
dat_sub$transition_reason <- 
  mapvalues(dat_sub$transition_reason,
            from=c("?","1","2","3","4","6","7","8","9","A","C","D","F","K","W"),
            to=c("Miss","PregWhileUse","WantPreg","HusbandDis","SideEffect","Access","WantMoreEffect","Inconvenient","InfreqSex","DifficultPreg","Cost","Separation","UpToGod","Unknw","Other"))

tab_switch_from_reason_prop <- prop.table(table(dat_sub[, c("from_method", "transition_reason")]), 1)
dat_melt <- melt(tab_switch_from_reason_prop)
ggplot(dat_melt,aes(x=from_method,y=transition_reason))+geom_tile(aes(fill=value))+scale_fill_gradient(low="white",high="blue")+labs(x="switch from method",y="reason of switching")
ggsave("switch_out_reason.png")

tab_switch_to_reason_prop <- prop.table(table(dat_sub[, c("to_method", "transition_reason")]),2)
dat_melt <- melt(tab_switch_to_reason_prop)
ggplot(dat_melt,aes(x=to_method,y=transition_reason))+geom_tile(aes(fill=value))+scale_fill_gradient(low="white",high="blue")+labs(x="switch to method",y="reason of switching")
ggsave("switch_into_reason.png")

