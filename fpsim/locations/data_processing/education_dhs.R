################################################################################
# -- Extract education data from DHS and PMA data datasets
# From DHS we extract: 
# - education objective (expressed in percentage of women that will aim to 
#                        complete X years of education, sratified by type of 
#                        residential setting).
# - education initialization (expressed in (edu) number of years a woman aged (age)
#                             will have likely completed).
# 
# From PMS we extract: 
# 
###############################################################################

# -- Libraries -- #
library(tidyverse)
library(haven)
library(survey)
library(withr)

# -- Data -- #

# Kenya 2022 individual recode
home_dir <-
  path.expand("~")   # replace with your own path to the DTA file
dhs_dir <- "DHS"
survey_dir <- "KEIR8BDT"
filename <- "KEIR8BFL.DTA"
filepath <- file.path(home_dir, dhs_dir, survey_dir, filename)

# -- Load all the data
data.raw <- read_dta(filepath)

# -- Make new columns with names that are more readable -- #
data <- data.raw %>%
  mutate(
    age = v012,
    parity = v220,
    edu = v133,
    urban = ifelse(v025 == 1, 1, 0)
  )

svydesign_obj_1 = svydesign(
  id = data$v001,
  strata = data$v023,
  weights = data$v005 / 1000000,
  data = data
)

# -- Preprocess education data -- #
table.edu.mean <-
  as.data.frame(svyby( ~ edu, ~ age, svydesign_obj_1, svymean))

# Define some age-related constants
dhs_min_age <- 15
dhs_max_age <- 49
fpsim_max_age <- 99

# Create projections for older and younger women by slope
age_lower_bound <- 20
age_upper_bound <- 43

table.edu.lag <- table.edu.mean %>%
  mutate(
    slope = (edu - lag(edu, 1)) / (age - lag(age, 1)),
    group = case_when(age < age_lower_bound ~ 1, age > age_upper_bound ~
                        2)
  ) %>%
  group_by(group) %>% mutate(avg.slope = ifelse(group == 1, mean(slope, na.rm = T), NA),
                             avg.edu = ifelse(group == 2, mean(edu), NA))
# Visualize education data
table.edu.mean %>%
  ggplot() +
  geom_line(aes(y = edu, x = age)) +
  geom_ribbon(aes(
    ymin = edu - se,
    ymax = edu + se,
    x = age
  ), alpha = 0.5) +
  geom_segment(aes(
    x = dhs_max_age + 1,
    y = min(table.edu.lag$avg.edu, na.rm = T),
    xend = 70,
    yend = min(table.edu.lag$avg.edu, na.rm = T)
  )) +
  geom_segment(aes(
    y = 0,
    x = dhs_min_age - min(table.edu.lag$edu, na.rm = T) / min(table.edu.lag$avg.slope, na.rm = T),
    xend = dhs_min_age,
    yend = min(table.edu.lag$edu, na.rm = T)
  )) +
  ylab("Mean years of education") + xlab("Age") +
  theme_bw(base_size = 13)


# Generate education table for initialization of education parameters
table.edu.inital <-
  data.frame(age = c(1:dhs_min_age - 1, dhs_max_age:fpsim_max_age)) %>%
  mutate(edu = ifelse(
    age < dhs_min_age,
    pmax(
      min(table.edu.lag$edu, na.rm = T) - min(table.edu.lag$avg.slope, na.rm = T) *
        (dhs_min_age - age),
      0
    ),
    min(table.edu.lag$avg.edu, na.rm = T)
  )) %>%
  bind_rows(table.edu.mean)

# For initialising education objective, we look at the distribution of
# years of education for all women over age 20 and living in urban settings
# (assumed that they have finished edu)
min_age <- 20
data.20 <- data %>% filter(age > min_age)
svydesign_obj_2 = svydesign(
  id = data.20$v001,
  strata = data.20$v023,
  weights = data.20$v005 / 1000000,
  data = data.20
)

table.edu.20 <-
  as.data.frame(svytable( ~ edu + urban, svydesign_obj_2)) %>% group_by(urban) %>% mutate(percent = Freq /
                                                                                            sum(Freq)) %>% select(-Freq)
current_levels_num <- sort(as.numeric(levels(table.edu.20$edu)))

# -- Fill out missing edu level values, edu=22 and edu=23 are not found
# in the data but fpsim needs a continuous range of edu values
all_levels <- expand.grid(edu = as.character(min(current_levels_num):max(current_levels_num)),
                          urban = unique(table.edu.20$urban))

table.edu.20 <-
  right_join(table.edu.20, all_levels, by = c("edu", "urban"))
table.edu.20$percent[is.na(table.edu.20$percent)] <- 0

table.edu.20$edu <- as.numeric(as.character(table.edu.20$edu))
table.edu.20$urban <- as.numeric(as.character(table.edu.20$urban))

table.edu.20 <- table.edu.20 %>%
  arrange(urban, edu)

table.edu.20$urban <- as.factor(table.edu.20$urban)
table.edu.20$edu <- as.factor(table.edu.20$edu)

table.edu.20 %>%
  ggplot() +
  geom_line(aes(
    y = percent,
    x = edu,
    group = urban,
    color = urban
  )) +
  ylab("Percent of women") +
  theme_bw(base_size = 13)

# -- Write files -- #
fpsim_dir <- "fpsim" # path to root directory of fpsim
locations_dir <- "fpsim/locations"
country_dir <- "kenya"
country_path <- file.path(home_dir, fpsim_dir, locations_dir, country_dir)

write.csv(table.edu.inital,
          file.path(country_path, 'edu_initialization.csv'),
          row.names = F)
write.csv(table.edu.20,
          file.path(country_path, 'edu_objective.csv'),
          row.names = F)
#write.csv(stop.school, file.path(country_path, 'edu_stop_.csv'), row.names = F)