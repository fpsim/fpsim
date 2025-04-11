
# Imports
library(readr)
library(ggplot2)
library(dplyr)

# Read data
setwd("~/Documents/git/fpsim/fpsim/locations/")
method_time_coefficients <- read_csv("kenya/data/method_time_coefficients.csv")
age_groups = c('<18', '18-20', '20-25', '25-35', '>35')
times <- seq(0, 12*10, length.out = 100)

#################################################
# Gamma plots: injectables & other traditional
# These ones look ok and are the same as the python versions
#################################################
thismethod = 'Other.trad'  # Injectable, IUD
thisdf <- method_time_coefficients %>% filter(method == thismethod)
age_factors <- thisdf$estimate[3:7]
shape <- exp(thisdf$estimate[2])

# Create empty dataframe
data <- data.frame(times = numeric(0), y = numeric(0), age = factor(), shape=numeric(0))

# Loop through the shapes and calculate the gamma pdf
for (i in 1:5) {
  age_label = age_groups[i]
  age_factor = age_factors[i]
  rate <- exp(thisdf$estimate[1] + age_factor)
  y <- dgamma(times, shape = shape, rate = rate)
  data <- rbind(data, data.frame(times = times / 12, y = y, shape = shape, age=as.factor(age_label)))
}

# Plot 
g <- ggplot(data) +
  geom_line(aes(x = times, y = y, color = age)) +
  labs(title = thismethod, x = "Duration of use", y = "PDF", color = "Age") +
  theme_minimal()  
g


#################################################
# WEIBULL plots: None, Implants, Other traditional
# Need help
#################################################
thismethod = 'None'  # Other.trad, Implant
thisdf <- method_time_coefficients %>% filter(method == thismethod)
age_factors <- thisdf$estimate[3:7]
shape <- exp(thisdf$estimate[1])

# Create empty dataframe
data <- data.frame(times = numeric(0), y = numeric(0), age = factor(), shape=numeric(0))

# Loop through the shapes and calculate the gamma pdf
for (i in 1:5) {
  age_label = age_groups[i]
  age_factor = age_factors[i]
  scale <- exp(thisdf$estimate[2] + age_factor)
  y <- dweibull(times, shape = shape, scale = scale)
  data <- rbind(data, data.frame(times = times / 12, y = y, shape = shape, age=as.factor(age_label)))
}

# Plot 
g <- ggplot(data) +
  geom_line(aes(x = times, y = y, color = age)) +
  labs(title = thismethod, x = "Duration of use", y = "PDF", color = "Age") +
  theme_minimal()  
g




#################################################
# Lognormal plots: None, pill, condoms, other modern
# These ones look wrong!!!
# Perhaps we shouldn't take the exponent of the parameters?
# If I don't take the exponent, they look much more reasonable,
# and I can replicate these in Python.
#################################################
thismethod = "None"
thisdf <- method_time_coefficients %>% filter(method == thismethod)
age_factors <- thisdf$coef[3:7]
sdlog <- exp(thisdf$coef[2])  ## REMOVE EXPONENT??

# Create empty dataframe
data <- data.frame(times = numeric(0), y = numeric(0), age = factor(), shape=numeric(0))

# Loop through the shapes and calculate the gamma pdf
for (i in 1:5) {
  age_label = age_groups[i]
  age_factor = age_factors[i]
  meanlog <- exp(thisdf$coef[1] + age_factor) ## REMOVE EXPONENT??
  y <- dlnorm(times, meanlog = meanlog, sdlog = sdlog)
  data <- rbind(data, data.frame(times = times / 12, y = y, shape = shape, age=as.factor(age_label)))
}

# Plot 
g <- ggplot(data) +
  geom_line(aes(x = times, y = y, color = age)) +
  labs(title = thismethod, x = "Duration of use", y = "PDF", color = "Age") +
  theme_minimal()  
g


#################################################
# Log logistic plots: Withdrawal
# @Marita please help?
#################################################
thismethod = "Withdrawal"
thisdf <- method_time_coefficients %>% filter(method == thismethod)
age_factors <- thisdf$coef[3:7]
rate <- exp(thisdf$coef[2])  

# Create empty dataframe
data <- data.frame(times = numeric(0), y = numeric(0), age = factor(), shape=numeric(0))

# Loop through the shapes and calculate the gamma pdf
for (i in 1:5) {
  age_label = age_groups[i]
  age_factor = age_factors[i]
  shape <- exp(thisdf$coef[1] + age_factor) 
  y <- dllogis(times, shape = shape, rate = rate)  ##TODO
  data <- rbind(data, data.frame(times = times / 12, y = y, shape = shape, age=as.factor(age_label)))
}

# Plot 
g <- ggplot(data) +
  geom_line(aes(x = times, y = y, color = age)) +
  labs(title = thismethod, x = "Duration of use", y = "PDF", color = "Age") +
  theme_minimal()  
g

#################################################
# Gompertz plots: implants, IUDs
# @Marita please help?
#################################################

