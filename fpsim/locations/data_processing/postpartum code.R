#########################################
# -- Posstpartum analysis - ##
# -- Marita Zimmermann
# -- October 2020
#########################################

rm(list=ls())

# -- Libraries -- #
library(tidyverse)      
library(haven)
library(scales)
library(withr)

# -- Download data -- #
# Source https://dhsprogram.com/data/dataset_admin/index.cfm
# recode https://www.dhsprogram.com/publications/publication-dhsg4-dhs-questionnaires-and-manuals.cfm
dhs.path <- normalizePath(file.path(Sys.getenv("ONEDRIVE"), "DHS"), "/")
dhs.kenya <- with_dir(dhs.path, {read_dta("KE_2014_DHS_09162022_2324_122388/KEIR72DT/KEIR72FL.DTA")}) # read individual recode data

# -- Data cleaning  
dhs.pp <- dhs.kenya %>%
mutate(b19_01 = v008 - b3_01) %>% # have to calculate months pp 
  filter(b19_01 < 24 & b9_01 == 0) %>% # keep if most recent birth is under 24 months, living with mother
  mutate(breastmilk = ifelse(m4_1 == 95, T, F), # breastfeeding https://dhsprogram.com/data/Guide-to-DHS-Statistics/Breastfeeding_and_Complementary_Feeding.htm#Calculation
         water = ifelse(v409 == 1, T, F), 
         non.milk.liquids = ifelse(v409a == 1 | v410 == 1 | v410a == 1 | v412c == 1 | v413 == 1 | v413a == 1 | v413b == 1 | v413c == 1 | v413d == 1, T, F),
         non.milk.liquids = ifelse(is.na(non.milk.liquids), F, non.milk.liquids),
         other.milk = ifelse(v411 == 1 | v411a == 1, T, F),
         solids = ifelse(v412a == 1 | v412b == 1 | v414a == 1 | v414b == 1 | v414c == 1 | v414d == 1 | v414e == 1 | v414f == 1 | v414g == 1 | v414h == 1 | v414i == 1 | v414j == 1 | v414k == 1 | v414l == 1 | v414m == 1 | v414n == 1 | v414o == 1 | v414p == 1 | v414q == 1 | v414r == 1 | v414s == 1 | v414t == 1 | v414u == 1 | v414v == 1 | v414w == 1 | m39a_1 == 1, T, F),
         solids = ifelse(is.na(solids), F, solids),
         exclusive.bf = ifelse(breastmilk == T & non.milk.liquids == F & other.milk == F & solids == F, T, F), # exclusively breastfeeding (only breastmilk and plain water)
         amenorrhea = ifelse((m6_1 >= b19_01 & m6_1 < 96) | m6_1 == 96, T, F),
         abstinent = ifelse(m8_1 >= b19_01 & m8_1 <= 96, T, F), # if duration of abstinence is greater than months pp and 
         exclusive.bf_amerorrhea = ifelse(exclusive.bf == T | amenorrhea == T, T, F),
         exclusive.bf_and_amerorrhea = ifelse(exclusive.bf == T & amenorrhea == T, T, F),
         exclusive.bf_amerorrhea_abstinent = ifelse(exclusive.bf_amerorrhea == T | abstinent == T, T, F),
         s.active_2 = case_when(v536 == 1 ~ T, v536 == 2 | v536 == 3 ~ F),
         wt = v005/1000000)

dhs.pp.results <- dhs.pp %>%
  rename(months_postpartum = b19_01) %>%
  group_by(months_postpartum) %>% 
  mutate_at(c("exclusive.bf", "amenorrhea", "exclusive.bf_and_amerorrhea"), ~ifelse(breastmilk, ., NA)) %>% # take out values for those who are not breastfeeding for these becasue we want percentages among breastfeeding women for the model
  summarise(abstinent = weighted.mean(abstinent, wt, na.rm = T), s.active_1 = 1-abstinent,
            s.active_2 = weighted.mean(s.active_2, wt, na.rm = T),
            exclusive.bf = weighted.mean(exclusive.bf, wt, na.rm = T), 
            amenorrhea = weighted.mean(amenorrhea, wt, na.rm = T), 
            exclusive.bf_and_amerorrhea = weighted.mean(exclusive.bf_and_amerorrhea, wt, na.rm = T), 
            n = sum(!is.na(abstinent)))

# -- sexual activity postpartum

# percentage of women sexually active by month postpartum
# method 1 is using duration of postpartum abstinence with months pp, and method 2 is using sexually active in last 4 weeks with months pp
# Note that method two overlooks if someone was sexually active pp, then stopped having sex in the last month
# We are currently using method 2 in the model
sexually.active.results <- dhs.pp.results %>% select(months_postpartum, n, s.active_1, s.active_2)
write.csv(sexually.active.results, "Postpartum/Results/sexually.active.results_2022-12-09.csv", row.names = F)

# visualize two methodologies
dhs.pp.results %>%
  select(-starts_with("n")) %>% 
  gather(var, val, -months_postpartum) %>%
  mutate(var = factor(var,
                         levels = c("exclusive.bf", "amenorrhea", "exclusive.bf_and_amerorrhea", "abstinent", "s.active_1", "s.active_2"),
                         labels = c("Exclusively breastfeeding", "Amenorrheic", "Exclusively breastfeeding and amenorrheic", "Abstinent", "Sexually active (method 1)", "Sexually active (method 2)"))) %>%
  filter(var == "Sexually active (method 1)" | var == "Sexually active (method 2)") %>%
  #filter(var == "Exclusively breastfeeding" | var ==  "Amenorrheic" | var ==  "Exclusively breastfeeding and amenorrheic") %>%
  ggplot() +
  geom_line(aes(y = val, x = months_postpartum, group = var, color = var), size = 1.5) +
  scale_color_viridis_d() + 
  theme_bw(base_size = 10) +
  xlab("Months post-partum") +
  scale_y_continuous("Percent of women", labels = percent) +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black", size = 0.1), 
        axis.text.x = element_text(colour = "black"), axis.text.y = element_text(colour = "black"),
        text = element_text(family ="Garamond"),
        legend.position = c(0.7, 0.8),
        legend.title = element_blank(),
        panel.border = element_blank()) 

# look at distribution of time abstinent for those who have resumed sex
dhs.pp %>%
  select(starts_with("m8_")) %>%
  gather(birth , months) %>%
  filter(months < 90) %>%
  filter(birth == "m8_1") %>%
  ggplot() +
  geom_bar(aes(x = months))
ggsave("Postpartum/Results/postpartum.percentage.bf_2021-09-03.png", height = 4, width = 8, units = "in", device='png')


# -- LAM postpartum

# percentage of women on LAM by month pp (breastfeeding, amenorrhea, and both)
LAM.results <- dhs.pp.results %>% select(months_postpartum, n, exclusive.bf, amenorrhea, exclusive.bf_and_amerorrhea)
write.csv(LAM.results, "Postpartum/Results/LAM.results_2022-12-09v2.csv", row.names = F)
