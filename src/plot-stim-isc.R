# Plot all the ISC results including chance level
# for both Bang You're dead and StoryCorps
# Each has the top 3 ISC componenents
library(RcppCNPy)
library(tidyverse)

# check current directory
current_dir <- getwd()
# if it ends in "src", go up one level
if (basename(current_dir) == "src") {
  setwd("..")
}

isc_result_files <- c(
  "byd" = c(
    "c1" = c(
      "chance" = "in/isc_results_bangbangyouaredead_full_chance_comp1.npy",
      "isc" = "in/isc_results_bangbangyouaredead_full_isc_component1_bywindow.npy"
    ),

    "c2" = c(
      "chance" = "in/isc_results_bangbangyouaredead_full_chance_comp2.npy",
      "isc" = "in/isc_results_bangbangyouaredead_full_isc_component2_bywindow.npy"
    ),
    "c3" = c(
      "chance" = "in/isc_results_bangbangyouaredead_full_chance_comp3.npy",
      "isc" = "in/isc_results_bangbangyouaredead_full_isc_component3_bywindow.npy"
    )
  ),
  "sc" = c(
    "c1" = c(
      "chance" = "in/isc_results_storycorps_q&a_full_chance_comp1.npy",
      "isc" = "in/isc_results_storycorps_q&a_full_isc_component1_bywindow.npy"
    ),
    "c2" = c(
      "chance" = "in/isc_results_storycorps_q&a_full_chance_comp2.npy",
      "isc" = "in/isc_results_storycorps_q&a_full_isc_component2_bywindow.npy"
    ),
    "c3" = c(
      "chance" = "in/isc_results_storycorps_q&a_full_chance_comp3.npy",
      "isc" = "in/isc_results_storycorps_q&a_full_isc_component3_bywindow.npy"
    )
  )
)


# Load the ISC results and chance level for each component and each stimulus
# read them into a data frame for plotting
df_isc_results <- imap_dfr(isc_result_files, function(isc_path, isc_id) {
  # isc_id is e.g. "byd.c1.chance", so we can split it into stimulus, component, and result_type
  isc_id_parts <- str_split(isc_id, "\\.")[[1]]
  stimulus <- isc_id_parts[1]
  component <- paste0("Component ", str_remove(isc_id_parts[2], "c"))
  result_type <- isc_id_parts[3]
  # load the ISC results from the .npy file
  isc_results <- npyLoad(isc_path)

  # create a data frame with the ISC results and the
  # corresponding stimulus, component, and result_type
  df <- data.frame(
    second = seq_along(isc_results),
    value = isc_results,
    stimulus = stimulus,
    component = component,
    result_type = result_type
  )
})

df_isc_results$stimulus <- factor(
  df_isc_results$stimulus,
  levels = c("byd", "sc"),
  labels = c("Bang! You're Dead", "StoryCorps Q&A")
)
df_isc_results$component <- factor(
  df_isc_results$component,
  levels = c("Component 1", "Component 2", "Component 3")
)
df_isc_results$result_type <- factor(
  df_isc_results$result_type,
  levels = c("chance", "isc"),
  labels = c("Chance Level", "ISC Result")
)


# Plot timeseries of ISC results and
# chance level for each component and each stimulus
# there should be a seperate plot for each stimulus
# each component will be a subplot, and the chance level will  be a filled
# area under the curve, while the ISC results will be a line plot
p1 <- df_isc_results |>
  filter(stimulus == "Bang! You're Dead") |>
  ggplot(aes(
    x = second,
    y = value,
    color = result_type,
    linewidth = result_type
  )) +
  geom_ribbon(
    data = df_isc_results |>
      filter(result_type == "Chance Level", stimulus == "Bang! You're Dead"),
    aes(ymin = 0, ymax = value, x = second),
    fill = "grey",
    alpha = 1,
    inherit.aes = FALSE
  ) +
  geom_vline(xintercept = 300, linetype = "dashed", color = "#606060") +
  annotate(
    "text",
    x = 300,
    y = 0.0,
    label = "Reference segment start",
    color = "#606060",
    size = 3,
    hjust = -0.1,
    vjust = 1.5
  ) +
  geom_vline(xintercept = 660, linetype = "dashed", color = "#606060") +
  annotate(
    "text",
    x = 660,
    y = 0.0,
    label = "Reference segment end",
    color = "#606060",
    size = 3,
    hjust = -0.1,
    vjust = 1.5
  ) +
  geom_line() +
  scale_x_continuous(
    limits = c(0, NA),
    breaks = seq(0, 1300, by = 100),
    expand = expansion(mult = c(0, 0))
  ) +
  scale_y_continuous(
    limits = c(0, NA),
    expand = expansion(mult = c(0.1, NA)),
    breaks = c(0, 0.1, 0.2, 0.3),
    oob = scales::squish
  ) +
  # coord_cartesian(ylim = c(0, 0.32)) +
  scale_color_manual(
    values = c("Chance Level" = "#606060", "ISC Result" = "#1f77b4"),
  ) +
  scale_linewidth_manual(
    values = c("Chance Level" = 0.0, "ISC Result" = 0.5),
    labels = NULL,
    guide = "none"
  ) +
  facet_wrap(vars(component), nrow = 3, scales = "free_y") +
  labs(
    title = "ISC Results: Bang! You're Dead",
    x = "Stimulus Elapsed Time (s)",
    y = "ISC Value",
    color = "Result Type",
    linewidth = NULL
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "bottom",
    legend.title = element_blank()
  )


ggsave(
  "out/isc_results_and_chance_level_byd.png",
  plot = p1,
  width = 8,
  height = 8,
  dpi = 300
)

p2 <- df_isc_results |>
  filter(stimulus == "StoryCorps Q&A") |>
  ggplot(aes(
    x = second,
    y = value,
    color = result_type,
    linewidth = result_type
  )) +
  geom_ribbon(
    data = df_isc_results |>
      filter(result_type == "Chance Level", stimulus == "StoryCorps Q&A"),
    aes(ymin = 0, ymax = value, x = second),
    fill = "grey",
    alpha = 1,
    inherit.aes = FALSE
  ) +
  geom_line() +
  scale_x_continuous(
    limits = c(0, NA),
    breaks = seq(0, 300, by = 20),
    expand = expansion(mult = c(0, 0))
  ) +
  scale_y_continuous(
    limits = c(0, NA),
    breaks = c(0, 0.1, 0.2, 0.3),
    oob = scales::squish
  ) +
  # coord_cartesian(ylim = c(0, 0.32)) +
  scale_color_manual(
    values = c("Chance Level" = "#606060", "ISC Result" = "#1f77b4")
  ) +
  scale_linewidth_manual(
    values = c("Chance Level" = 0.0, "ISC Result" = 0.8),
    labels = NULL,
    guide = "none"
  ) +
  facet_wrap(vars(component), nrow = 3, scales = "free_y") +
  labs(
    title = "ISC Results: StoryCorps Q&A",
    x = "Stimulus Elapsed Time (s)",
    y = "ISC Value",
    color = "Result Type",
    linewidth = NULL
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "bottom",
    legend.title = element_blank()
  )


ggsave(
  "out/isc_results_and_chance_level_sc.png",
  plot = p2,
  width = 8,
  height = 8,
  dpi = 300
)
