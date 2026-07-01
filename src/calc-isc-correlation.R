# Calculate the timeseries correlation
# between ISC results from the same stimuli
# across independent groups of subjects from
# different labs and equipment
# install.packages(c("zoo", "RcppCNPy", "tidyverse", "lme4", "lmerTest"))
library(RcppCNPy)
library(tidyverse)
library(lme4)
library(lmerTest)
library(zoo)


csv_isc_1sec_timeseries <- read_csv("../in/isc_comparison.csv")
csv_isc_1sec_timeseries
csv_isc_1sec_timeseries$second <- 1:360

dmochowsky_col <- "dmochowski_timeseries"
poulsen_col <- "poulsen_timeseries"

kappel_dat_seg <- npyLoad(
  "../in/isc_results_bangbangyouaredead_segment_isc_component1_bywindow.npy"
)
kappel_dat_seg_chance <- npyLoad(
  "../in/isc_results_bangbangyouaredead_segment_chance_comp1.npy"
)

kappel_dat_full <- npyLoad(
  "../in/isc_results_bangbangyouaredead_full_isc_component1_bywindow.npy"
)
kappel_dat_full_chance <- npyLoad(
  "../in/isc_results_bangbangyouaredead_full_chance_comp1.npy"
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

print(
  paste0(
    "Peak correlation with Dmochowski et al: ",
    peak_corr_dm_y,
    " at ",
    peak_corr_dm_x,
    "s"
  )
)


# same plot as above but using ggplot instead

# set pleasent plot colors
color_dmochowski <- "#1f77b4" # blue
color_poulsen <- "#646464" # orange
sliding_correlation

df_sliding <- data.frame(
  time = rep(seq_along(sliding_correlation), 2),
  corr = c(sliding_correlation, sliding_correlation_poulsen),
  study = rep(
    c("Dmochowski et al", "Poulsen et al"),
    each = length(sliding_correlation)
  )
)

df_sliding$study <- factor(
  df_sliding$study,
  levels = c("Dmochowski et al", "Poulsen et al")
)

plt <- df_sliding |>
  ggplot(aes(time, corr, color = study)) +
  geom_line() +
  annotate(
    "text",
    x = peak_corr_dm_x + 10,
    y = peak_corr_dm_y,
    label = paste0(
      "Dmochowski peak:\n",
      round(peak_corr_dm_y, 3),
      " at ",
      peak_corr_dm_x,
      "s"
    ),
    color = color_dmochowski,
    hjust = 0
  ) +
  annotate(
    "point",
    x = peak_corr_dm_x,
    y = peak_corr_dm_y,
    color = color_dmochowski,
    size = 3
  ) +
  annotate(
    "text",
    x = peak_corr_poulsen_x - 10,
    y = peak_corr_poulsen_y,
    label = paste0(
      "Poulsen peak:\n",
      round(peak_corr_poulsen_y, 3),
      " at ",
      peak_corr_poulsen_x,
      "s"
    ),
    color = color_poulsen,
    hjust = 1
  ) +
  annotate(
    "point",
    x = peak_corr_poulsen_x,
    y = peak_corr_poulsen_y,
    color = color_poulsen,
    size = 3
  ) +
  labs(
    x = "Rolling window start time (s)",
    y = "Correlation",
    title = "Windowed correlation of present study with reference studies",
    subtitle = "Rolling window size: 360s"
  ) +
  # add legend
  scale_color_manual(
    name = "Reference Study",
    values = c(
      color_dmochowski,
      color_poulsen
    )
  ) +
  theme_minimal()
ggsave(
  "../out/rolling_correlation_full_ggplot.png",
  width = 10,
  height = 6,
  dpi = 300
)
