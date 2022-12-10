### Data to process DHS data for breastfeeding duration

dir <- "directory" ## directory with data
filename <- "filename" ## must be a KR (child recode) file from the DHS for automatic processing

##Breastfeeding duration 
setwd(dir)
dat <- read.dta13(filename)

require(ggplot2)

dat_nonmiss <- subset(dat, m5 < 93) #m5 = months of breastfeeding in previous pregnancy

fitgumbel <- fitdist(dat_nonmiss$m5, "gumbel", start=list(a=1, b=1))
summary(fitgumbel)
plot(fitgumbel)

bf_stats <- as.data.frame(fitgumbel$estimate)

save(bf_stats, file = "bf_stats.csv")