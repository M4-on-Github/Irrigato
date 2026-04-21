# ==============================================================================
# CSC 314 Project: Advanced Visualizations & Model Explainability
# Description: Generates presentation-ready high quality visualization charts
#              insights. This script acts as an extension and uses the
#              model/data from analysis.R
# ==============================================================================

# ------------------------------------------------------------------------------
# Setup Packages and Libraries
# ------------------------------------------------------------------------------
required_packages <- c(
  "tidyverse", "randomForest", "corrplot", "pdp", "vip",
  "ggcorrplot", "scales", "rpart", "rpart.plot"
)
new_packages <- required_packages[
  !(required_packages %in% installed.packages()[, "Package"])
]
if (length(new_packages) > 0) {
  cat(
    "Installing missing packages: ",
    paste(new_packages, collapse = ", "),
    "\n"
  )
  install.packages(new_packages, repos = "http://cran.us.r-project.org")
}

library(tidyverse)
library(randomForest)
library(corrplot)
library(pdp)
library(vip)
library(ggcorrplot)
library(scales)
library(rpart)
library(rpart.plot)

# ------------------------------------------------------------------------------
# Dependency Check: Ensure analysis.R has been run to provide data and model
# ------------------------------------------------------------------------------
if (!exists("rf_model") || !exists("train_data")) {
  cat("Model 'rf_model' or 'train_data' not found in environment.\n")
  cat("Sourcing 'analysis.R' to load data and train the initial model...\n")
  source("analysis.R")
} else {
  cat("Successfully detected 'rf_model' and 'train_data' from analysis.R!\n")
}

# Setup Output Directory
out_dir <- "visualization"
if (!dir.exists(out_dir)) {
  dir.create(out_dir)
  cat(sprintf("Created output directory: %s\n", out_dir))
}

# Global Theme for highly readable, professional plots
my_theme <- theme_minimal(base_size = 14) +
  theme(plot.title = element_text(face = "bold", size = 16),
        plot.subtitle = element_text(size = 12, color = "gray40"),
        plot.margin = ggplot2::margin(t = 10, r = 10, b = 10, l = 10),
        legend.position = "bottom")

# ==============================================================================
# SECTION 1: Advanced Exploratory Data Analysis (EDA)
# ==============================================================================
cat("\n[1/3] Generating EDA visualizations...\n")

# 1A. Correlation Heatmap of Numerical Features
cat(" -> Correlation Heatmap...\n")
numeric_data <- train_data |> select(where(is.numeric))
cor_matrix <- cor(numeric_data)

png(file.path(out_dir, "eda_numeric_correlation.png"),
    width = 800, height = 800, res = 110)
corrplot(cor_matrix, method = "color", type = "upper",
         tl.col = "black", tl.srt = 45,
         addCoef.col = "black", number.cex = 0.8,
         title = "\nCorrelation of Environmental Variables",
         mar = c(0, 0, 2, 0),
         col = colorRampPalette(
           c("#E46726", "white", "#6D9EC1")
         )(200))
dev.off()


# 1B. Proportional Stacked Bar Chart for Crop Type
cat(" -> Crop Type Proportions...\n")
p_crop <- ggplot(train_data, aes(x = Crop_Type, fill = Irrigation_Need)) +
  geom_bar(position = "fill", color = "white", linewidth = 0.5) +
  scale_y_continuous(labels = percent) +
  scale_fill_manual(values = c(
    "Low" = "#3CAEA3", "Medium" = "#F2B134", "High" = "#ED553B"
  )) +
  labs(title = "Proportion of Irrigation Need by Crop Type",
       subtitle = "Percentage breakdown highlighting water-demanding crops",
       x = "Crop Type", y = "Proportion", fill = "Irrigation Need") +
  my_theme +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave(file.path(out_dir, "eda_crop_irrigation_prop.png"),
       plot = p_crop, width = 9, height = 6, dpi = 300)

# 1C. Interaction Violin Boxplots
cat(" -> Violin Plots for Moisture and Maturity...\n")
p_violin <- ggplot(
  train_data,
  aes(x = Crop_Growth_Stage, y = Soil_Moisture, fill = Irrigation_Need)
) +
  geom_violin(trim = FALSE, alpha = 0.7) +
  geom_boxplot(
    width = 0.15, position = position_dodge(0.9), color = "black",
    outlier.size = 0.5, alpha = 0.9
  ) +
  scale_fill_manual(values = c(
    "Low" = "#3CAEA3", "Medium" = "#F2B134", "High" = "#ED553B"
  )) +
  labs(title = "Soil Moisture Distribution by Growth Stage",
       subtitle = paste(
         "Assessing how moisture thresholds for irrigation change during",
         "crop maturity"
       ),
       x = "Crop Growth Stage", y = "Soil Moisture (%)",
       fill = "Irrigation Need") +
  my_theme +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave(file.path(out_dir, "eda_moisture_growth_violin.png"),
       plot = p_violin, width = 10, height = 6, dpi = 300)

# ==============================================================================
# SECTION 2: Model Explainability
# ==============================================================================
cat("\n[2/3] Generating Model Explainability visualizations...\n")

# 2A. Random Forest Error Convergence
cat(" -> Random Forest Error Convergence...\n")
err_data <- as.data.frame(rf_model$err.rate)
err_data$Trees <- seq_len(nrow(err_data))
err_data_long <- err_data |>
  pivot_longer(cols = -Trees, names_to = "Error_Type", values_to = "Error_Rate")

p_err <- ggplot(
  err_data_long,
  aes(x = Trees, y = Error_Rate, color = Error_Type)
) +
  geom_line(linewidth = 1) +
  scale_color_manual(values = c(
    "OOB" = "black", "Low" = "#3CAEA3", "Medium" = "#F2B134", "High" = "#ED553B"
  )) +
  labs(title = "Random Forest Training Convergence",
       subtitle = "Out-of-Bag (OOB) and Class Error Rates across Trees",
       x = "Number of Trees", y = "Error Rate", color = "Metric") +
  my_theme

ggsave(file.path(out_dir, "exp_rf_error_convergence.png"),
       plot = p_err, width = 8, height = 5, dpi = 300)

# 2B. Polished Feature Importance
cat(" -> Variable Importance (VIP)...\n")
p_vip <- vip(rf_model, num_features = 12, geom = "col", fill = "#20639B") +
  labs(title = "Random Forest Feature Importance",
       subtitle = "Top Predictors for Determining Irrigation Need",
       y = "Importance Score", x = "Feature") +
  my_theme

ggsave(file.path(out_dir, "exp_feature_importance.png"),
       plot = p_vip, width = 8, height = 6, dpi = 300)

# 2C. Surrogate Decision Tree (Visualizing Splits)
cat(" -> Generating Surrogate Decision Tree (flowchart of split rules)...\n")
# A random forest cannot be plotted as a single visual tree, so we build a
# surrogate rpart tree to structurally capture & visualize its primary
# splitting rules visually!
surrogate_tree <- rpart(
  Irrigation_Need ~ ., data = train_data, maxdepth = 4, method = "class"
)

png(file.path(out_dir, "exp_surrogate_tree.png"),
    width = 1000, height = 800, res = 120)
rpart.plot(surrogate_tree,
           main = "Surrogate Decision Tree (Approximating Random Forest Logic)",
           type = 3, extra = 104, fallen.leaves = TRUE, shadow.col = "gray",
           box.palette = list("#ED553B", "#3CAEA3", "#F2B134"))
dev.off()

cat(" -> Exporting raw structural split rules for Tree #1...\n")
tree_1_rules <- getTree(rf_model, k = 1, labelVar = TRUE)
write.csv(tree_1_rules, file.path(out_dir, "rf_tree_1_raw_splits.csv"))

# 2D & 2E Partial Dependence Plots
cat(" -> Preparing subsets for PDP (to accelerate rendering)...\n")
# Sampling data purely to make the mathematical Partial Dependence
# calculation run faster
pdp_sample <- train_set
if (nrow(pdp_sample) > 2000) {
  set.seed(42)
  pdp_sample <- sample_n(pdp_sample, 2000)
}

# 2B. Partial Dependence Plot (PDP) for Soil Moisture
cat(" -> Calculating Partial Dependence for Soil Moisture...\n")
pdp_moisture <- partial(
  rf_model, pred.var = "Soil_Moisture", prob = TRUE,
  which.class = 1L, train = pdp_sample
)
p_pdp_moist <- autoplot(pdp_moisture, alpha = 0.1) +
  geom_line(color = "#173F5F", linewidth = 2) +
  labs(title = "Partial Dependence: Soil Moisture",
       subtitle = paste(
         "Marginal effect of moisture on the probability of 'High'",
         "irrigation need"
       ),
       x = "Soil Moisture (%)", y = "Predicted Probability (High)") +
  my_theme

ggsave(file.path(out_dir, "exp_pdp_soil_moisture.png"),
       plot = p_pdp_moist, width = 7, height = 5, dpi = 300)

# 2C. Partial Dependence Plot (PDP) for Rainfall
cat(" -> Calculating Partial Dependence for Rainfall...\n")
pdp_rain <- partial(
  rf_model, pred.var = "Rainfall_mm", prob = TRUE,
  which.class = 1L, train = pdp_sample
)
p_pdp_rain <- autoplot(pdp_rain, alpha = 0.1) +
  geom_line(color = "#ED553B", linewidth = 2) +
  labs(title = "Partial Dependence: Rainfall",
       subtitle = paste(
         "Marginal effect of recent precipitation on probability of",
         "'High' need"
       ),
       x = "Rainfall (mm)", y = "Predicted Probability (High)") +
  my_theme

ggsave(file.path(out_dir, "exp_pdp_rainfall.png"),
       plot = p_pdp_rain, width = 7, height = 5, dpi = 300)

# ==============================================================================
# SECTION 3: Multi-Dimensional Insight
# ==============================================================================
cat("\n[3/3] Generating Multi-Dimensional insights and combinations...\n")

cat(" -> Extracting Multi-dimensional facets...\n")
# Sample data for plot clarity without overcrowding points
set.seed(42)
plot_data <- train_data
if (nrow(plot_data) > 3000) {
  plot_data <- sample_n(plot_data, 3000)
}

p_facet <- ggplot(
  plot_data,
  aes(x = Rainfall_mm, y = Soil_Moisture, color = Irrigation_Need)
) +
  geom_point(alpha = 0.6, size = 1.5) +
  facet_wrap(~ Season) +
  scale_color_manual(values = c(
    "Low" = "#3CAEA3", "Medium" = "#F2B134", "High" = "#ED553B"
  )) +
  labs(title = "Environmental Triggers for Irrigation Across Seasons",
       subtitle = paste(
         "Interaction between Moisture, Rainfall, Season and Decision",
         "(Sampled for Clarity)"
       ),
       x = "Rainfall (mm)", y = "Soil Moisture (%)",
       color = "Irrigation Need") +
  my_theme +
  theme(strip.text = element_text(face = "bold", size = 12),
        panel.spacing = unit(1, "lines"))

ggsave(file.path(out_dir, "multi_season_facet.png"),
       plot = p_facet, width = 10, height = 7, dpi = 300)

# ==============================================================================
# SECTION 4: Model Exporting & Metrics
# ==============================================================================
cat("\n[4/4] Exporting Model and Metadata...\n")

# 4A. Export the entire Random Forest Model object
cat(" -> Saving rf_model.rds to analysis folder...\n")
saveRDS(rf_model, file = file.path(out_dir, "rf_model.rds"))

# 4B. Export Feature Importance to CSV
cat(" -> Saving Feature Importance CSV...\n")
imp_df <- as.data.frame(randomForest::importance(rf_model))
imp_df$Feature <- rownames(imp_df)
write.csv(imp_df,
          file.path(out_dir, "rf_feature_importance.csv"),
          row.names = FALSE)

# 4C. High-Quality Confusion Matrix Heatmap (if evaluated in analysis.R)
if (exists("conf_matrix")) {
  cat(" -> Plotting Confusion Matrix...\n")
  cm_data <- as.data.frame(conf_matrix$table)
  p_cm <- ggplot(cm_data, aes(x = Reference, y = Prediction, fill = Freq)) +
    geom_tile(color = "white", linewidth = 1) +
    geom_text(
      aes(label = Freq), color = "white", size = 6, fontface = "bold"
    ) +
    scale_fill_viridis_c(option = "mako", begin = 0.2, end = 0.8) +
    labs(
      title = "Validation Confusion Matrix",
      subtitle = "Performance of Random Forest on held-out validation set",
      x = "Actual Irrigation Need",
      y = "Predicted Irrigation Need",
      fill = "Count"
    ) +
    my_theme +
    theme(legend.position = "right")

  ggsave(
    file.path(out_dir, "exp_confusion_matrix.png"),
    plot = p_cm, width = 7, height = 6, dpi = 300
  )
}

cat("\n", strrep("=", 72), "\n", sep = "")
cat("SUCCESS! All visualizations and model exports saved in 'analysis/'.\n")
cat(strrep("=", 72), "\n", sep = "")
