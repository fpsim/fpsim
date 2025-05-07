## Calculate method mix from DHS data
library(tidyverse)
library(survey)

dhs.raw <- withr::with_dir(normalizePath(file.path(Sys.getenv("ONEDRIVE"), "DHS/KEIR8ADT"), "/"), {read_dta("KEIR8AFL.DTA")})

# Recode current method variable
dhs.data <- dhs.raw %>%
  mutate(method_recode = case_when(v312 == 1 ~ "Pill",
                                   v312 == 2 ~ "IUDs",
                                   v312 == 3 ~ "Injectables",
                                   v312 %in% c(4,13,15,16,17,18) ~ "Other modern",
                                   v312 %in% c(5,14) ~ "Condoms",
                                   v312 %in% c(6,7) ~ "BTL",
                                   v312 %in% c(8,10,12) ~ "Other traditional",
                                   v312 == 9 ~ "Withdrawal",
                                   v312 == 11 ~ "Implants"),
         wt = v005/1000000) %>%
  filter(!is.na(method_recode))

# survey weighted table
mix <- as.data.frame(svytable(~ method_recode, svydesign(id=~v021, weights=~wt, strata=~v023, data = dhs.data))) %>% mutate(perc = Freq/sum(Freq)*100)
write.csv(mix, "mix.csv")








