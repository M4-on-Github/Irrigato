# ==============================================================================
# CSC 314 Project: Irrigation Need Prediction
# Description: Classification analysis to predict irrigation needs based
#              on soil qualities, crop types, and environmental factors.
# ==============================================================================

library(tidyverse)
library(randomForest)
library(caret)

# ------------------------------------------------------------------------------
# 1. Data Import & Cleaning
# ------------------------------------------------------------------------------
cat("Loading data...\n")
test_data <- read.csv(
  "~/dataMining/Final_proj_Irrigation_Need/playground-series-s6e4/test.csv"
)
train_data <- read.csv(
  "~/dataMining/Final_proj_Irrigation_Need/playground-series-s6e4/train.csv"
)

# We drop the 'id' column from the data frame used for modeling
train_data <- train_data |> select(-id)

# Convert all character columns to factors (required by randomForest)
train_data <- train_data |> mutate(across(where(is.character), as.factor))
test_data <- test_data |> mutate(across(where(is.character), as.factor))

# Check for missing values
sum_na <- sum(is.na(train_data))
cat("Total missing values in training set:", sum_na, "\n")
if (sum_na > 0) {
  cat("Dropping rows with missing values...\n")
  train_data <- na.omit(train_data)
}

# ------------------------------------------------------------------------------
# 2. Exploratory Data Analysis (EDA)
# ------------------------------------------------------------------------------
# Create output directory for all analysis artifacts
dir.create("analysis", showWarnings = FALSE)

cat("Generating EDA plots (Saving to analysis/)...\n")

# 2.A Target Variable Distribution
p_target <- ggplot(
  train_data,
  aes(x = Irrigation_Need, fill = Irrigation_Need)
) +
  geom_bar(alpha = 0.8) +
  theme_minimal() +
  labs(title = "Distribution of Irrigation Need Classes",
       x = "Irrigation Need", y = "Count") +
  theme(legend.position = "none") # Clearer plots for presentations
ggsave("analysis/eda_target_distribution.png",
       plot = p_target, width = 6, height = 4)

# 2.B Relationship: Soil Moisture vs Irrigation Need
p_moist <- ggplot(
  train_data,
  aes(x = Irrigation_Need, y = Soil_Moisture, fill = Irrigation_Need)
) +
  geom_boxplot(alpha = 0.8) +
  theme_minimal() +
  labs(title = "Impact of Soil Moisture on Irrigation Need",
       x = "Irrigation Need Classification", y = "Soil Moisture (%)") +
  theme(legend.position = "none")
ggsave("analysis/eda_soil_moisture.png", plot = p_moist, width = 6, height = 4)

# 2.C Relationship: Rainfall vs Irrigation Need
p_rain <- ggplot(
  train_data,
  aes(x = Irrigation_Need, y = Rainfall_mm, fill = Irrigation_Need)
) +
  geom_boxplot(alpha = 0.8) +
  theme_minimal() +
  labs(title = "Impact of Rainfall on Irrigation Need",
       x = "Irrigation Need Classification", y = "Rainfall (mm)") +
  theme(legend.position = "none")
ggsave("analysis/eda_rainfall.png", plot = p_rain, width = 6, height = 4)

# ------------------------------------------------------------------------------
# 3. Modeling & Classification
# ------------------------------------------------------------------------------
cat("Preparing for classification modeling...\n")

# Split the training data into training (80%) and validation sets (20%)
set.seed(123) # For reproducibility
train_index <- createDataPartition(
  train_data$Irrigation_Need, p = 0.8, list = FALSE
)
train_set <- train_data[train_index, ]
valid_set <- train_data[-train_index, ]

cat("Training Random Forest model (this may take a moment)...\n")
# We use ntree = 100 for a solid balance of speed and performance.
rf_model <- randomForest(
  Irrigation_Need ~ ., data = train_set, ntree = 100, importance = TRUE
)

# Show model performance on the validation set
cat("Evaluating on validation set...\n")
valid_predictions <- predict(rf_model, newdata = valid_set)
conf_matrix <- confusionMatrix(
  factor(valid_predictions), factor(valid_set$Irrigation_Need)
)

print(conf_matrix)

# Variable Importance Plot
# Excellent for PowerPoint slide discussions!
png("analysis/model_variable_importance.png", width = 800, height = 600)
varImpPlot(
  rf_model,
  main = "Feature Importance for Predicting Irrigation Need",
  pch = 16, col = "blue"
)
dev.off()

# ------------------------------------------------------------------------------
# 4. Final Inference & Packaging
# ------------------------------------------------------------------------------
cat("Generating predictions for test set...\n")

# Predict on test data
test_predictions <- predict(rf_model, newdata = test_data)

# Create submission data frame
submission <- data.frame(id = test_data$id, Irrigation_Need = test_predictions)

# Save the predictions to a CSV file
write.csv(submission, "analysis/final_submission.csv", row.names = FALSE)
cat("SUCCESS! Results saved to 'analysis/final_submission.csv'.\n")
