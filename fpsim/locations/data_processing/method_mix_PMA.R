## Calculate method mix from PMA file
require(readstata13)
require(ggplot2)

dir <- 'directory'
filename <- "PMA2022_KEP3_HQFQ_v2.0_17Aug2022.dta" #replace with recent PMA data file

setwd(dir)
kdat <- readstata13::read.dta13(filename)

table(kdat$current_method, kdat$current_user) # confirm tabulation of any method vs. modern method only

use <- as.data.frame(table(kdat$current_user) %>% prop.table)
use$perc <- use$Freq * 100
use$use <- NA
use$use[use$Var1 == '0. No']  <- 'No current method'
use$use[use$Var1 == '1. Yes'] <- 'Any current method'

ggplot(use, 
       aes(x = use, y = perc, fill = use)) + 
  geom_histogram(stat = 'identity') +
  scale_fill_brewer(palette = 'Paired') +
  coord_flip() + 
  xlab('') + ylab('%') + 
  theme(legend.position = 'none')

write.csv(use, "use.csv")
##recode - note that methods MUST be in order of efficacy
kdat$method_recode <- NA
kdat$method_recode[is.na(kdat$method_recode) == TRUE & kdat$femalester == 1] <- 'BTL/vasectomy'
kdat$method_recode[is.na(kdat$method_recode) == TRUE & kdat$malester == 1] <- 'BTL/vasectomy'
kdat$method_recode[is.na(kdat$method_recode) == TRUE & kdat$implant == 1] <- 'implant'
kdat$method_recode[is.na(kdat$method_recode) == TRUE & kdat$IUD == 1] <- 'IUD'
kdat$method_recode[is.na(kdat$method_recode) == TRUE & kdat$pill == 1] <- 'pill'
kdat$method_recode[is.na(kdat$method_recode) == TRUE & kdat$injectables == 1] <- 'injectables'
kdat$method_recode[is.na(kdat$method_recode) == TRUE & kdat$EC == 1] <- 'other modern'
kdat$method_recode[is.na(kdat$method_recode) == TRUE & kdat$malecondoms == 1] <- 'condoms'
kdat$method_recode[is.na(kdat$method_recode) == TRUE & kdat$femalecondoms == 1] <- 'condoms'
kdat$method_recode[is.na(kdat$method_recode) == TRUE & kdat$diaphragm == 1] <- 'other modern'
kdat$method_recode[is.na(kdat$method_recode) == TRUE & kdat$foamjelly == 1] <- 'other modern'
kdat$method_recode[is.na(kdat$method_recode) == TRUE & kdat$stndrddays == 1] <- 'other modern'
kdat$method_recode[is.na(kdat$method_recode) == TRUE & kdat$LAM == 1] <- 'other modern'
kdat$method_recode[is.na(kdat$method_recode) == TRUE & kdat$withdrawal == 1] <- 'withdrawal'
kdat$method_recode[is.na(kdat$method_recode) == TRUE & kdat$rhythm == 1] <- 'other traditional'
kdat$method_recode[is.na(kdat$method_recode) == TRUE & kdat$othertrad == 1] <- 'other traditional'

mix_subset <- kdat[!(is.na(kdat$method_recode)), ]
mix <- as.data.frame(table(mix_subset$method_recode) %>% prop.table)
mix$perc <- mix$Freq * 100
names(mix)[1] <- 'method'

ggplot(mix, 
       aes(x = reorder(method,(perc)), 
           y = perc, 
           fill = method)) + 
  geom_histogram(stat = 'identity') +
  geom_text(aes(x=method, y= (perc + 2.5), 
                label=round(perc, digits = 2))) + 
  scale_fill_brewer(palette = 1) +
  coord_flip() + 
  xlab('') + ylab('% of method users') + 
  theme(legend.position = 'none') +
  theme_minimal() +
  ylim(0,50)

write.csv(mix, "mix.csv")






