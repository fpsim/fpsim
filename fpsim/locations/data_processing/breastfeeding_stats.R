### Data to process DHS data for breastfeeding duration

dir <- "directory" ## directory with data
filename <- "filename" ## must be a KR (child recode) file from the DHS for automatic processing

##Breastfeeding duration 
setwd(dir)
dat <- read.dta13(filename)

require(fitdistrplus)

dat_nonmiss <- subset(dat, m4 < 93) #m5 = months of breastfeeding in previous pregnancy
dat_nonmiss$m4_weighted <- (dat_nonmiss$v005/1000000) * dat_nonmiss$m4 #use this code to apply survey weights


#gumbel distribution definitions 
dgumbel <- function(x, a, b) 1/b*exp((a-x)/b)*exp(-exp((a-x)/b))
pgumbel <- function(q, a, b) exp(-exp((a-q)/b))
qgumbel <- function(p, a, b) a-b*log(-log(p))

fitgumbel <- fitdist(dat_nonmiss$m4, #replace with weighted var here if using survey weights
                     "gumbel", start=list(a=1, b=1))
summary(fitgumbel)
plot(fitgumbel)

bf_stats <- as.data.frame(fitgumbel$estimate)

write.csv(bf_stats, file = "bf_stats.csv") #label with country name for saving

mean(dat_nonmiss$m4)
