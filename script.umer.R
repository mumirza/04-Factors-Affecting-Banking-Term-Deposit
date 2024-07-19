# Muhammad U. Mirza
# 12/02/2024, ALY 6015
# Final Project

################################################################################
# Clear the Console
################################################################################
cat("\014")  # Clears the console
rm(list = ls())  # Clears the global environment
try(dev.off(dev.list()["RStudioGD"]), silent = TRUE)  # Clears plots
try(p_unload(p_loaded(), character.only = TRUE), silent = TRUE)  # Clears packages
options(scipen = 100)  # Disables scientific notation for the entire R session

################################################################################
# Load Necessary Packages
################################################################################
library(pacman)
p_load(dplyr, tidyverse, janitor, lubridate, ggplot2, ggthemes, ggeasy, psych, 
       knitr, kableExtra, corrplot, RColorBrewer, car, MASS, leaps, caret,
       gridExtra, pROC, ISLR, glmnet, Metrics)

################################################################################
# Load Data
################################################################################
bank <- read.csv("bank.csv")

################################################################################
# Inspect Data
################################################################################
dim(bank) # check number of rows and columns
colnames(bank) # check column names
head(bank) # check top 10 observations of the data 
str(bank) # check the structure of each column
sapply(bank, function(x) length(unique(x))) # check no. of unique values in each column
sapply(bank, function(x) sum(is.na(x))) # check for n/a values

################################################################################
# Clean Data
################################################################################
bank <- clean_names(bank) # make column names R friendly

# Converting binary categorical variables to factors with 'yes' as the first level
binary_vars <- c("default", "housing", "loan", "y")
bank[binary_vars] <- lapply(bank[binary_vars], function(x) factor(x, levels = c("yes", "no")))

# Converting other categorical variables to factors
other_categorical_vars <- c("job", "marital", "education", "contact", "poutcome")
bank[other_categorical_vars] <- lapply(bank[other_categorical_vars], factor)

# Ensure structure changes are completed 
str(bank)

################################################################################
# EDA
################################################################################
# Descriptive statistics for numerical variables
numerical_vars <- bank[, sapply(bank, is.numeric)]
# Function to calculate descriptive stats for numerical variables
numerical_stats <- sapply(numerical_vars, function(x) {
  round(c(
    Min = min(x, na.rm = TRUE), 
    Max = max(x, na.rm = TRUE), 
    Mean = mean(x, na.rm = TRUE), 
    SD = sd(x, na.rm = TRUE), 
    Median = median(x, na.rm = TRUE), 
    `Q1` = quantile(x, 0.25, na.rm = TRUE), 
    `Q3` = quantile(x, 0.75, na.rm = TRUE)
  ), 2)})

numerical_stats_df <- as.data.frame(t(numerical_stats))
numerical_stats_df$Variable <- rownames(numerical_stats_df)
rownames(numerical_stats_df) <- NULL
numerical_stats_df <- numerical_stats_df[, c(ncol(numerical_stats_df), 1:(ncol(numerical_stats_df)-1))]

# Create a kable table for numerical variables
kable(numerical_stats_df, "html", caption = "Descriptive Statistics for Numerical Variables") %>%
  kable_styling(bootstrap_options = c("striped", "hover"), font_size = 12) %>%
  kable_classic(full_width = F, html_font = "Cambria")

# Function to calculate descriptive stats for categorical variables
calculate_counts_percentages_desc <- function(factor_var) {
  counts <- table(factor_var)
  percentages <- prop.table(counts) * 100
  summary_df <- data.frame(
    Level = names(counts),
    Count = as.integer(counts),
    Percentage = round(percentages, 2))
  # Sort the dataframe in descending order by Count
  summary_df <- summary_df[order(-summary_df$Count),]
  return(summary_df)}

# Apply the function to each categorical (factor) variable in the dataset
categorical_summary_desc <- lapply(bank[sapply(bank, is.factor)], calculate_counts_percentages_desc)

# Name the list elements as the variable names for easier reference
names(categorical_summary_desc) <- names(bank[sapply(bank, is.factor)])
categorical_summary_desc

# List of specified categorical variables
specified_vars <- c("marital", "education", "default", "housing", "loan", "contact", "poutcome", "y")

# Generate summaries for specified variables and consolidate
consolidated_summary <- do.call(rbind, lapply(specified_vars, function(var) {
  summary_df <- calculate_counts_percentages_desc(bank[[var]])
  summary_df$Variable <- var
  return(summary_df)}))

# Order the consolidated summary by Variable, then by Count within each Variable
consolidated_summary <- consolidated_summary[, -which(names(consolidated_summary) == "Percentage.factor_var")]

# Move 'Variable' column to the first position
consolidated_summary <- consolidated_summary[c("Variable", "Level", "Count", "Percentage.Freq")]

# Creating a kable table for the consolidated summary
kable(consolidated_summary, "html", caption = "Summary for Specified Categorical Variables", row.names = FALSE) %>%
  kable_styling(bootstrap_options = c("striped", "hover"), font_size = 12) %>%
  kable_classic(full_width = F, html_font = "Cambria")

################################################################################
# Q1) Can the likelihood of a customer subscribing to a term deposit be predicted based on their demographic profile and account characteristics?
################################################################################
# Encoding the target outcome for term deposit subscription
bank$subscribed <- ifelse(bank$y == "yes", 1, 0)
# Drop the 'y' column
bank <- bank[, names(bank) != "y"]

set.seed(123) # Ensure reproducibility for data split
split_index <- sample(x = nrow(bank), size = nrow(bank) * 0.7)
training_set <- bank[split_index, ]
testing_set <- bank[-split_index, ]

# Preparing feature matrices for regression analysis
features_train <- model.matrix(subscribed ~ age + job + marital + education + default + balance + housing + loan, data=training_set)[,-1]
features_test <- model.matrix(subscribed ~ age + job + marital + education + default + balance + housing + loan, data=testing_set)[,-1]

# Setting up outcome vectors
outcome_train <- training_set$subscribed
outcome_test <- testing_set$subscribed

# Calculate weights - Increase weight for the minority class 
weights_train <- ifelse(outcome_train == 1, sum(outcome_train == 0) / sum(outcome_train == 1), 1)

# Estimating regularization parameters for Ridge analysis
set.seed(123)
cv_fit_ridge <- cv.glmnet(features_train, outcome_train, weights = weights_train, alpha = 0, nfolds = 10)

# Optimal regularization strengths
log(cv_fit_ridge$lambda.min) # Optimal for prediction
log(cv_fit_ridge$lambda.1se) # Within one standard error

# Visualizing cross-validation outcomes
plot(cv_fit_ridge)

# Constructing Ridge regression with optimal parameter
model_ridge_min <- glmnet(features_train, outcome_train, alpha = 0, lambda = cv_fit_ridge$lambda.min)
model_ridge_1se <- glmnet(features_train, outcome_train, alpha = 0, lambda = cv_fit_ridge$lambda.1se)

# Reviewing model coefficients
coef(model_ridge_min)
coef(model_ridge_1se)

# Training set performance (RMSE)
predictions_train_ridge <- predict(model_ridge_1se, newx = features_train) 
rmse_train_ridge <- rmse(outcome_train, predictions_train_ridge)

# Test set performance (RMSE)
predictions_test_ridge <- predict(model_ridge_1se, newx = features_test) 
rmse_test_ridge <- rmse(outcome_test, predictions_test_ridge)

rmse_train_ridge
rmse_test_ridge

# Implementing Lasso regression for comparison
set.seed(123)  
cv_fit_lasso <- cv.glmnet(features_train, outcome_train, weights = weights_train, alpha = 1, nfolds = 10)

# Visual assessment of Lasso model fit
plot(cv_fit_lasso)

# Lasso model construction with determined lambdas
model_lasso_min <- glmnet(features_train, outcome_train, alpha = 1, lambda = cv_fit_lasso$lambda.min)
model_lasso_1se <- glmnet(features_train, outcome_train, alpha = 1, lambda = cv_fit_lasso$lambda.1se)

# Model coefficients inspection
coef(model_lasso_min)
coef(model_lasso_1se)

# RMSE evaluation for Lasso model on training data
lasso_train_preds <- predict(model_lasso_1se, newx = features_train) 
rmse_training_lasso <- rmse(outcome_train, lasso_train_preds)

# RMSE calculation for Lasso model on test data
lasso_test_preds <- predict(model_lasso_1se, newx = features_test) 
rmse_testing_lasso <- rmse(outcome_test, lasso_test_preds)

rmse_training_lasso
rmse_testing_lasso

# Feature selection via stepwise regression to refine model
stepwise_model <- step(lm(subscribed ~ ., data = training_set), direction = 'both')

# Performance metrics for stepwise regression model
stepwise_train_pred <- predict(stepwise_model, newdata = training_set)
rmse_training_stepwise <- rmse(actual = training_set$subscribed, predicted = stepwise_train_pred)

stepwise_test_pred <- predict(stepwise_model, newdata = testing_set)
rmse_testing_stepwise <- rmse(actual = testing_set$subscribed, predicted = stepwise_test_pred)

rmse_training_stepwise
rmse_testing_stepwise

################################################################################
# Q2) Can the likelihood of a customer subscribing to a term deposit be predicted based on their demographic profile and account characteristics?
################################################################################
