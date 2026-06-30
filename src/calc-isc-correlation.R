# Calculate the timeseries correlation
# between ISC results from the same stimuli
# across independent groups of subjects from
# different labs and equipment
install.packages(c("zoo", "RcppCNPy", "tidyverse", "lme4", "lmerTest"))
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

kappel_dat_seg <- npyLoad(
  "../out/isc_results_byd_segment_isc_component1_bywindow.npy"
)
kappel_dat_seg_chance <- npyLoad(
  "../out/isc_results_byd_segment_chance_level.npy"
)

kappel_dat_full <- npyLoad(
  "../out/isc_results_byd_full_isc_component1_bywindow.npy"
)
kappel_dat_full_chance <- npyLoad(
  "../out/isc_results_byd_full_chance_level.npy"
)

length(kappel_dat_seg)

length(kappel_dat_full)

# calculate correlation of the timeseries from the reference studies
# Dmochowski et al and Poulsen et al, with the present study Kappel et al
dmochowsky_cor <- cor(
  csv_isc_1sec_timeseries[[dmochowsky_col]],
  csv_isc_1sec_timeseries[[poulsen_col]]
)

# print the correlation values
cat("Dmochowski et al vs Poulsen et al: ", dmochowsky_cor, "\n")

kappel_corr_x_dm <- cor(
  kappel_dat_seg,
  csv_isc_1sec_timeseries[[dmochowsky_col]]
)
kappel_corr_x_poulsen <- cor(
  kappel_dat_seg,
  csv_isc_1sec_timeseries[[poulsen_col]]
)

# print the correlation values
cat("Kappel et al vs Dmochowski et al: ", kappel_corr_x_dm, "\n")
cat("Kappel et al vs Poulsen et al: ", kappel_corr_x_poulsen, "\n")

# run a rolling correlation of the FULL Kappel
# timeseries with the Dmochowski et al and Poulsen et al timeseries
sliding_correlation <- rollapply(
  kappel_dat_full,
  width = 360,
  FUN = function(x) cor(x, csv_isc_1sec_timeseries[[dmochowsky_col]]),
  by.column = FALSE,
  align = "center"
)
sliding_correlation_poulsen <- rollapply(
  kappel_dat_full,
  width = 360,
  FUN = function(x) cor(x, csv_isc_1sec_timeseries[[poulsen_col]]),
  by.column = FALSE,
  align = "center"
)

peak_corr_dm_y <- max(sliding_correlation, na.rm = TRUE)
peak_corr_poulsen_y <- max(sliding_correlation_poulsen, na.rm = TRUE)

peak_corr_dm_x <- which.max(sliding_correlation)
peak_corr_poulsen_x <- which.max(sliding_correlation_poulsen)

# plot the rolling correlation results
plot(
  sliding_correlation,
  type = "l",
  col = "blue",
  ylim = c(-1, 1),
  xlab = "Time (s)",
  ylab = "Correlation",
  main = "Rolling Correlation: Kappel (full), Dmochowski, Poulsen"
)
lines(sliding_correlation_poulsen, col = "red")
# mark the peak correlation points
points(peak_corr_dm_x, peak_corr_dm_y, col = "blue", pch = 19)
points(peak_corr_poulsen_x, peak_corr_poulsen_y, col = "red", pch = 19)

# label with value and time
text(
  peak_corr_dm_x,
  peak_corr_dm_y + 0.05,
  labels = paste0(
    "Peak: ",
    round(peak_corr_dm_y, 3),
    " at ",
    peak_corr_dm_x,
    "s"
  ),
  pos = 3,
  col = "blue"
)
text(
  peak_corr_poulsen_x,
  peak_corr_poulsen_y,
  labels = paste0(
    "Peak: ",
    round(peak_corr_poulsen_y, 3),
    " at ",
    peak_corr_poulsen_x,
    "s"
  ),
  pos = 3,
  col = "red"
)

legend(
  "topright",
  legend = c(
    "Kappel et al vs Dmochowski et al",
    "Kappel et al vs Poulsen et al"
  ),
  col = c("blue", "red"),
  lty = 1
)
