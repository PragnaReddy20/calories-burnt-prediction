# Load necessary libraries
library(keras)
library(tensorflow)
library(tibble)
library(readr)
library(dplyr)  # Load dplyr for the pipe operator

# 1. Load Dataset
data <- read.csv("C:/Users/Tejasri/Downloads/combined_dataset.csv", stringsAsFactors = FALSE)

# Check the dataset structure and missing values
str(data)

# Handle missing values and convert columns to numeric
data <- na.omit(data)
data$Age <- as.numeric(data$Age)
data$Weight <- as.numeric(data$Weight)
data$Height <- as.numeric(data$Height)
data$Duration <- as.numeric(data$Duration)
data$HeartRate <- as.numeric(data$Heart_Rate)
data$BodyTemp <- as.numeric(data$Body_Temp)
data$Calories <- as.numeric(data$Calories)

# Normalize input features and target variable
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

# Select relevant columns and normalize them
data$Age <- normalize(data$Age)
data$Weight <- normalize(data$Weight)
data$Height <- normalize(data$Height)
data$Duration <- normalize(data$Duration)
data$HeartRate <- normalize(data$Heart_Rate)
data$BodyTemp <- normalize(data$Body_Temp)

# Split dataset into input (X) and target (y)
X <- as.matrix(data[, c("Age", "Weight", "Height", "Duration", "Heart_Rate", "Body_Temp")])
y <- as.matrix(data$Calories)

# Split into train and test sets
set.seed(123)  # For reproducibility
train_index <- sample(1:nrow(X), 0.8 * nrow(X))
X_train <- X[train_index, ]
y_train <- y[train_index]
X_test <- X[-train_index, ]
y_test <- y[-train_index]

# 2. Build Neural Network Model
model <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = 'relu', input_shape = ncol(X)) %>%
  layer_dense(units = 32, activation = 'relu') %>%
  layer_dense(units = 1)  # Output layer for regression

model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam(),
  metrics = c('mean_absolute_error')
)

# 3. Train the Model
history <- model %>% fit(
  X_train, y_train,
  epochs = 50,
  batch_size = 16,
  validation_split = 0.1
)

# 4. Evaluate the Model
evaluation <- model %>% evaluate(X_test, y_test)
print(paste("Test Loss: ", evaluation$loss))
print(paste("Test MAE: ", evaluation$mean_absolute_error))

# 5. Make Predictions
predictions <- model %>% predict(X_test)

# Denormalize the predictions and actual values
denormalize <- function(x, min_val, max_val) {
  return (x * (max_val - min_val) + min_val)
}

calories_min <- min(data$Calories)
calories_max <- max(data$Calories)

predictions_denorm <- denormalize(predictions, calories_min, calories_max)
y_test_denorm <- denormalize(y_test, calories_min, calories_max)

# 6. Plot Results
plot(y_test_denorm, type = "l", col = "blue", lwd = 2, xlab = "Sample", ylab = "Calories Burnt",
     main = "Actual vs Predicted Calories Burnt")
lines(predictions_denorm, col = "red", lwd = 2)
legend("topright", legend = c("Actual", "Predicted"), col = c("blue", "red"), lty = 1, lwd = 2)


