# Final_Project_Group_6_Analysis
# Zeel R. Patel, Niyati Virmani, Kimberly C. Dmello, Haotian Zhu, Muhammad U. Mirza

# -------------------------------------------------
#                   Clear the Console
# -------------------------------------------------
cat("\014")  # Clears the console
rm(list = ls())  # Clears the global environment
try(dev.off(dev.list()["RStudioGD"]), silent = TRUE)  # Clears plots
try(p_unload(p_loaded(), character.only = TRUE), silent = TRUE)  # Clears packages
options(scipen = 100)  # Disables scientific notation for the entire R session

# -------------------------------------------------
#                   Load Necessary Packages
# -------------------------------------------------
library(pacman)
p_load(dplyr, tidyverse, janitor, lubridate, ggplot2, ggthemes, ggeasy, psych, 
       knitr, kableExtra, corrplot, RColorBrewer, car, MASS, leaps, caret,
       gridExtra, pROC, ISLR, glmnet, Metrics, reshape2)

# -------------------------------------------------
#           Load the dataset into file
# -------------------------------------------------
file.path("bank.csv")
bank_data <- read.csv("bank.csv")
bank_data

class(bank_data)
# -------------------------------------------------
#                       EDA
# -------------------------------------------------
# 1.Renaming columns
names(bank_data)
bank_data <- rename(bank_data, "job_title" = "job", "marital_status" = "marital", "credit_default" = "default", "housing_loan" = "housing", "contact_type" = "contact", "contact_date" = "day", "contact_month" = "month", "subscribe_term_deposit" = "y")
names(bank_data)

# 2. Managing NAs
# Handle missing values: Remove rows with missing values
bank_data <- na.omit(bank_data)

# 3. Display summary statistics
summary(bank_data)

# 4. EDA

# Combine distribution of 'marital_status' and 'education' in one plot
ggplot(bank_data, aes(x = marital_status, fill = education)) +
  geom_bar(position = "dodge", color = "black", stat = "count") +
  labs(title = "Distribution of Marital Status and Education") +
  scale_fill_brewer(palette = "Set2")


# Function to map a numeric variable to the target variable
map_numeric_to_target <- function(data, numeric_var, target_var) {
  ggplot(data, aes(x = !!sym(numeric_var), fill = !!sym(target_var))) +
    geom_histogram(binwidth = 50, alpha = 0.7, position = "identity") +
    labs(title = paste("Distribution of", numeric_var, "by", target_var),
         x = numeric_var, y = "Frequency", fill = target_var)
}

# Example usage for 'duration'
map_numeric_to_target(bank_data, "duration", "subscribe_term_deposit")

# Scatter plot of balance vs. subscription
ggplot(bank_data, aes(x = balance, y = subscribe_term_deposit, color = subscribe_term_deposit)) +
  geom_point(alpha = 0.7) +
  labs(title = "Client Balance vs. Subscription",
       x = "Client Balance",
       y = "Subscription (0: No, 1: Yes)",
       color = "Subscription")

# Calculate subscription rate by month
subscription_rate <- bank_data %>%
  group_by(contact_month) %>%
  summarize(subscription_rate = sum(subscribe_term_deposit == "yes") / n())

# Bar plot of subscription rate by month
ggplot(subscription_rate, aes(x = contact_month, y = subscription_rate)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "Subscription Rate by Month",
       x = "Contact Month",
       y = "Subscription Rate")

# Select numerical columns for correlation analysis
numerical_columns <- bank_data[, sapply(bank_data, is.numeric)]

# Calculate correlation matrix
correlation_matrix <- cor(numerical_columns)

# Visualize the correlation matrix using a heatmap
correlation_melted <- melt(correlation_matrix)
ggplot(correlation_melted, aes(Var1, Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0, limit = c(-1,1), space = "Lab", name="Correlation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# -------------------------------------------------
#                       Analysis
# -------------------------------------------------
# Question-1: Is there any relationship between housing-loan and subscribe to a term deposit(subscription)?

# a. Hypotheses
# Null hypothesis: There's no relationship between the person currently using a housing loan and the person's subscription.
# Alternative hypothesis: There's a relationship between the person currently using a housing loan and the person's subscription.

# Set significance level
alpha <- 0.05

# Create a contingency table
contingency_table <- table(bank_data$subscribe_term_deposit, bank_data$housing_loan)
contingency_table

# Perform the chi-square test of independence
chi_square_result <- chisq.test(contingency_table)
chi_square_result

# Make the decision
ifelse(chi_square_result$p.value > alpha, "Fail to reject the null hypothesis.", "Reject the null hypothesis")

# Question-2: Can we predict the likelihood of having a housing loan based on the client's job type?

# Split the dataset into train and test sets
set.seed(123)  # for reproducibility

train_indices <- sample(1:nrow(bank_data), 0.7 * nrow(bank_data))

train_data <- bank_data[train_indices, ]
test_data <- bank_data[-train_indices, ]

head(train_data)
head(test_data)

# Create a new binary variable indicating presence or absence of a housing loan
train_data$binary_housing_loan <- ifelse(train_data$housing_loan == "yes", 1, 0)

# Fit logistic regression with the binary outcome
logistic_model <- glm(binary_housing_loan ~ job_title, data = train_data, family = "binomial")
summary(logistic_model)

# Display regression coefficients (log-odds)
coef(logistic_model)

# Display regression coefficients (convert log-odds to odds)
exp(coef(logistic_model))

#Job titles such as retired and student have significantly lower odds of having a housing loan.
#Job titles like blue-collar and services show significant increases in the likelihood of having a housing loan.


# Create a confusion matrix for the train set
predicted_train <- ifelse(predict(logistic_model, newdata = train_data, type = "response") > 0.5, "yes", "no")
conf_matrix_train <- table(predicted_train, train_data$housing_loan)
conf_matrix_train

# Calculate accuracy, precision, recall, and specificity
accuracy_train <- sum(diag(conf_matrix_train)) / sum(conf_matrix_train)
precision_train <- conf_matrix_train[2, 2] / sum(conf_matrix_train[, 2])
recall_train <- conf_matrix_train[2, 2] / sum(conf_matrix_train[2, ])
specificity_train <- conf_matrix_train[1, 1] / sum(conf_matrix_train[1, ])

cat("Accuracy:", accuracy_train, "\n")
cat("Precision:", precision_train, "\n")
cat("Recall:", recall_train, "\n")
cat("Specificity:", specificity_train, "\n")

# Create a confusion matrix for the test set
predicted_test <- ifelse(predict(logistic_model, newdata = test_data, type = "response") > 0.5, "yes", "no")
conf_matrix_test <- table(predicted_test, test_data$housing_loan)
conf_matrix_test

# Plot the ROC curve:
roc_curve <- roc(test_data$housing_loan, as.numeric(predicted_test == "yes"))

plot(roc_curve, main = "ROC Curve for Housing Loan Prediction", col = "blue", lwd = 2)
lines(c(0, 1), c(0, 1), col = "red", lty = 2, lwd = 2)  # Diagonal line for reference
legend("bottomright", legend = paste("AUC =", round(auc(roc_curve), 3)), col = "blue", lwd = 2)

# Calculate the AUC:
auc_score <- auc(roc_curve)
cat("AUC Score:", auc_score, "\n")

# Additional EDA 

# Descriptive statistics for numerical variables
numerical_vars <- bank_data[, sapply(bank_data, is.numeric)]
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
# Convert the list of numerical statistics into a dataframe. 
numerical_stats_df <- as.data.frame(t(numerical_stats))
# Add a new column to the dataframe containing the variable names. 
numerical_stats_df$Variable <- rownames(numerical_stats_df)
# Clear the row names of the dataframe because they are no longer needed and to clean up the presentation.
rownames(numerical_stats_df) <- NULL
# Reorder the columns to move the 'Variable' column to the first position for better readability.
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
categorical_summary_desc <- lapply(bank_data[sapply(bank_data, is.factor)], calculate_counts_percentages_desc)

# Name the list elements as the variable names for easier reference
names(categorical_summary_desc) <- names(bank_data[sapply(bank_data, is.factor)])
categorical_summary_desc

# List of specified categorical variables
specified_vars <- c("marital_status", "education", "credit_default", "housing_loan", "loan", "contact_type", "poutcome", "subscribe_term_deposit")

# Generate summaries for specified variables and consolidate
consolidated_summary <- do.call(rbind, lapply(specified_vars, function(var) {
  summary_df <- calculate_counts_percentages_desc(bank_data[[var]])
  summary_df$Variable <- var
  return(summary_df)}))

# Move 'Variable' column to the first position
consolidated_summary <- consolidated_summary[c("Variable", "Level", "Count", "Percentage.Freq")]
# Creating a kable table for the consolidated summary
kable(consolidated_summary, "html", caption = "Summary for Specified Categorical Variables", row.names = FALSE) %>%
  kable_styling(bootstrap_options = c("striped", "hover"), font_size = 12) %>%
  kable_classic(full_width = F, html_font = "Cambria")

# Question-3: Can the likelihood of a customer subscribing to a term deposit be predicted based on their demographic profile and account characteristics?

# Encoding the target outcome for term deposit subscription
bank_data$subscribed <- ifelse(bank_data$subscribe_term_deposit == "yes", 1, 0)
# Drop the 'subscribe_term_deposit' column
bank_data <- bank_data[, names(bank_data) != "subscribe_term_deposit"]

set.seed(123) # Ensure reproducibility for data split
split_index <- sample(x = nrow(bank_data), size = nrow(bank_data) * 0.7)
training_set <- bank_data[split_index, ]
testing_set <- bank_data[-split_index, ]

# Preparing feature matrices for regression analysis
features_train <- model.matrix(subscribed ~ age + job_title + marital_status + education + credit_default + balance + housing_loan + loan, data=training_set)[,-1]
features_test <- model.matrix(subscribed ~ age + job_title + marital_status + education + credit_default + balance + housing_loan + loan, data=testing_set)[,-1]

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
rmse_train_ridge <- sqrt(mean((outcome_train - predictions_train_ridge)^2))

# Test set performance (RMSE)
predictions_test_ridge <- predict(model_ridge_1se, newx = features_test) 
rmse_test_ridge <- sqrt(mean((outcome_test - predictions_test_ridge)^2))

cat("RMSE Training Ridge:", rmse_train_ridge, "\n")
cat("RMSE Test Ridge:", rmse_test_ridge, "\n")

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
rmse_training_lasso <- sqrt(mean((outcome_train - lasso_train_preds)^2))

# RMSE calculation for Lasso model on test data
lasso_test_preds <- predict(model_lasso_1se, newx = features_test) 
rmse_testing_lasso <- sqrt(mean((outcome_test - lasso_test_preds)^2))

cat("RMSE Training Lasso:", rmse_training_lasso, "\n")
cat("RMSE Testing Lasso:", rmse_testing_lasso, "\n")

# Feature selection via stepwise regression to refine model
stepwise_model <- step(lm(subscribed ~ ., data = training_set), direction = 'both')

# Performance metrics for stepwise regression model
stepwise_train_pred <- predict(stepwise_model, newdata = training_set)
rmse_training_stepwise <- rmse(actual = training_set$subscribed, predicted = stepwise_train_pred)

stepwise_test_pred <- predict(stepwise_model, newdata = testing_set)
rmse_testing_stepwise <- rmse(actual = testing_set$subscribed, predicted = stepwise_test_pred)

rmse_training_stepwise
rmse_testing_stepwise

# Question-4: -------------------------------------






