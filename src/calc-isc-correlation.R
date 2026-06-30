# Calculate the timeseries correlation
# between ISC results from the same stimuli
# across independent groups of subjects from
# different labs and equipment
install.packages("zoo")

library(RcppCNPy)
library(tidyverse)
library(lme4)
library(lmerTest)
library(zoo)


csv_isc_1sec_timeseries <- read_csv("../out/isc_comparison.csv")
csv_isc_1sec_timeseries
csv_isc_1sec_timeseries$second <- 1:360

dmochowsky_col <- "dmochowski_timeseries"
poulsen_col <- "poulsen_timeseries"

kappel_data <- npyLoad("../in/isc_component1_byd_6min.npy")

length(kappel_data)

# calculate correlation of the timeseries from the reference studies
# Dmochowski et al and Poulsen et al, with the present study Kappel et al
dmochowsky_cor <- cor(csv_isc_1sec_timeseries[[dmochowsky_col]], csv_isc_1sec_timeseries[[poulsen_col]])

# print the correlation values
cat("Dmochowski et al vs Poulsen et al: ", dmochowsky_cor, "\n")

kappel_corr_x_dm <- cor(kappel_data, csv_isc_1sec_timeseries[[dmochowsky_col]])
kappel_corr_x_poulsen <- cor(kappel_data, csv_isc_1sec_timeseries[[poulsen_col]])
