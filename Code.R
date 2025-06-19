library(infotheo)
library(tidyr)
library(dplyr)
library(infotheo)
library(stargazer)
library(rpart)
library(rpart.plot)
library(FNN)
library(caret)
library(caret)
library("e1071")
library(class)
library(knitr)

data = read.csv("hotel.csv")


str(data)
summary(data)




#Remove duplicates
data <- data[!duplicated(data),]

df <- data

#Remove the index variable
df <- df[, -1]
# Deal with the outlier of distribution_channel
df = df[which(df$distribution_channel != "Undefined"),]
levels(as.factor(df$distribution_channel))
#Check for NA
any(is.na(df))
#There are no NA in the dataframe

### Helping functinons

# Assuming all necessary dataframes and models (train, test, m.logit0, tree_best_Depth,
# knn_train, knn_test, best_k, Accuracy.glm0, Accuracy.PredByTree, Knn_accuracy)
# are already defined and available in your environment from the original script.
match_columns <- function(df1, df2) {
  common_cols <- intersect(colnames(df1), colnames(df2))
  return(df1[, common_cols, drop = FALSE])
}



clean_NAs <- function(data_to_clean){
  if (any(is.na(data_to_clean))){
  drop_rows = max(colSums(is.na(data_to_clean))) < max(rowSums(is.na(data_to_clean)))
  if (drop_rows) {data_to_clean = data_to_clean[complete.cases(data_to_clean),]}
  if (!drop_rows) { data_to_clean = data_to_clean[, colSums(is.na(data_to_clean)) == 0]}
  }
  return(data_to_clean)
}



calculate_model_accuracy <- function(
    train_data,                # Base training dataset
    test_data,                 # Base test dataset
    model_logit,               # Logistic regression model
    model_tree,                # Decision tree model
    logit_data_train,          # Dataset for logistic regression training predictions
    logit_data_test,           # Dataset for logistic regression test predictions
    tree_data_train,           # Dataset for tree training predictions
    tree_data_test,            # Dataset for tree test predictions
    knn_train_features,        # Feature set for KNN training
    knn_test_features,         # Feature set for KNN testing
    train_target,              # Training target variable (is_canceled)
    test_target,               # Test target variable (is_canceled)
    knnPred,                   # Out of sample predictions
    train_knnPred,             # In of sample predictions
    best_k,                    # Optimal k for KNN
    model_accuracies,          # Pre-calculated individual model accuracies
    threshold = 0.5 # Decision threshold
) {
  # Normalize weights to sum to 1
  Weights <- model_accuracies / sum(model_accuracies)
  
  # Output weights for reference
  #message("Model weights: ", paste(round(Weights, 4), collapse = ", "))
  
  #==== OUT-OF-SAMPLE (TEST) ACCURACY ====
  
  # 1. Generate individual model predictions for test data
  # Logistic regression predictions
  log_pred <- predict(model_logit, newdata = logit_data_test, type = "response")
  # log_pred_binary <- ifelse(log_pred > 0.5, 1, 0)
  
  # Decision tree predictions
  tree_pred <- predict(model_tree, newdata = tree_data_test, type = "class")
  
  # 2. Create weighted ensemble prediction for test data
  combined_pred <- (
    Weights[1] * unname(log_pred) +  # Use the continuous probability
      Weights[2] * unname(as.numeric(as.character(tree_pred))) +
      Weights[3] * as.numeric(as.character(knnPred))
  )
  
  # 3. Apply threshold to get final binary predictions
  test_combined_binary <- ifelse(combined_pred < threshold, 0, 1)
  
  # 4. Calculate out-of-sample accuracy
  test_nCorrect <- sum(test_target == test_combined_binary, na.rm = TRUE)
  test_accuracy <- test_nCorrect / length(test_target)
  
  #==== IN-SAMPLE (TRAINING) ACCURACY ====
  
  # 1. Generate individual model predictions for training data
  # Logistic regression predictions
  train_log_pred <- predict(model_logit, newdata = logit_data_train, type = "response")
  # Important: Do NOT binarize here.
  # train_log_pred_binary <- ifelse(train_log_pred > 0.5, 1, 0)
  
  # Decision tree predictions
  train_tree_pred <- predict(model_tree, newdata = tree_data_train, type = "class")
  
  # 2. Create weighted ensemble prediction for training data
  train_combined_pred <- (
    Weights[1] * unname(train_log_pred) + # Use continuous probability
      Weights[2] * unname(as.numeric(as.character(train_tree_pred))) +
      Weights[3] * as.numeric(as.character(train_knnPred))
  )
  
  # 3. Apply threshold to get final binary predictions
  train_combined_binary <- ifelse(train_combined_pred < threshold, 0, 1)
  
  # 4. Calculate in-sample accuracy
  train_nCorrect <- sum(train_target == train_combined_binary, na.rm = TRUE)
  train_accuracy <- train_nCorrect / length(train_target)
  
  # Return all relevant results
  return(list(
    test_accuracy = test_accuracy,
    train_accuracy = train_accuracy,
    weights = Weights,
    threshold = threshold
    # No need to return predictions unless you want to inspect them.
  ))
}


### Finding best threshold for glm model
find_best_threshold <- function(glm_model, test_data, target_var, 
                                metric = "f1", n_thresholds = 100) {
  # Generate predicted probabilities
  pred_probs <- predict(glm_model, newdata = test_data, type = "response")
  actual <- test_data[[target_var]]
  
  # Sequence of thresholds to test
  thresholds <- seq(0.01, 0.99, length.out = n_thresholds)
  
  # Initialize results data frame
  results <- data.frame(
    threshold = thresholds,
    accuracy = numeric(n_thresholds),
    f1 = numeric(n_thresholds),
    precision = numeric(n_thresholds),
    recall = numeric(n_thresholds),
    specificity = numeric(n_thresholds),
    balanced_accuracy = numeric(n_thresholds)
  )
  
  # Calculate metrics for each threshold
  for (i in seq_along(thresholds)) {
    pred_class <- ifelse(pred_probs > thresholds[i], 1, 0)
    cm <- table(Predicted = pred_class, Actual = actual)
    
    # Handle cases with no positive predictions
    if (nrow(cm) == 1) {
      if (rownames(cm) == "0") {
        cm <- rbind(cm, c(0, 0))
        rownames(cm) <- c("0", "1")
      } else {
        cm <- rbind(c(0, 0), cm)
        rownames(cm) <- c("0", "1")
      }
    }
    
    tp <- cm["1", "1"]
    fp <- cm["1", "0"]
    tn <- cm["0", "0"]
    fn <- cm["0", "1"]
    
    # Calculate metrics
    results$accuracy[i] <- (tp + tn) / sum(cm)
    results$precision[i] <- ifelse((tp + fp) == 0, 0, tp / (tp + fp))
    results$recall[i] <- ifelse((tp + fn) == 0, 0, tp / (tp + fn))
    results$specificity[i] <- tn / (tn + fp)
    results$balanced_accuracy[i] <- (results$recall[i] + results$specificity[i]) / 2
    results$f1[i] <- ifelse((results$precision[i] + results$recall[i]) == 0, 0,
                            2 * (results$precision[i] * results$recall[i]) / 
                              (results$precision[i] + results$recall[i]))
  }
  
  # Find threshold that maximizes selected metric
  best_row <- which.max(results[[metric]])
  best_threshold <- thresholds[best_row]
  
  return(list(
    best_threshold = best_threshold,
    best_metric_value = results[best_row, metric],
    all_results = results,
    metric_used = metric
  ))
}



# Prepare data for GLM
factor_to_dummy <- function(data, factor_variable_name, drop_first_level = TRUE) {
  # Check if the input data is a data frame
  if (!is.data.frame(data)) {
    stop("Input 'data' must be a data frame.")
  }
  
  # Check if the factor variable exists in the data frame
  if (!factor_variable_name %in% names(data)) {
    stop(paste("Factor variable '", factor_variable_name, "' not found in data frame.", sep = ""))
  }
  
  # Check if the specified variable is a factor
  if (!is.factor(data[[factor_variable_name]])) {
    stop(paste("Variable '", factor_variable_name, "' is not a factor.", sep = ""))
  }
  
  # Get the factor variable
  factor_variable <- data[[factor_variable_name]]
  
  # Create dummy variables using model.matrix
  # The "+ 0" removes the intercept column that model.matrix usually adds
  dummy_matrix <- model.matrix(~ factor_variable + 0, data = data)
  
  # Convert the dummy matrix to a data frame
  dummy_df <- as.data.frame(dummy_matrix)
  
  # Rename the dummy variables to be more informative
  colnames(dummy_df) <- paste(factor_variable_name, levels(factor_variable), sep = "_")
  
  # Drop the first level if drop_first_level is TRUE
  if (drop_first_level) {
    dummy_df <- dummy_df[, -1, drop = FALSE] # drop = FALSE prevents error if only one column remains
  }
  
  # Combine the dummy variables with the original data frame, excluding the original factor variable
  data_without_factor <- data[, !names(data) %in% factor_variable_name]
  new_data <- cbind(data_without_factor, dummy_df)
  
  return(new_data)
}


to_factor <- function(data) {
  # Check if the input data is a data frame
  if (!is.data.frame(data)) {
    stop("Input 'data' must be a data frame.")
  }
  
  # Identify character columns
  char_cols <- sapply(data, is.character)
  
  # Convert all character columns to factors
  for (col_name in names(data)[char_cols]) {
    data[[col_name]] <- as.factor(data[[col_name]])
  }
  
  return(data)
}


factors_to_dummies <- function(data, drop_first_level = TRUE) {
  # Check if the input data is a data frame
  if (!is.data.frame(data)) {
    stop("Input 'data' must be a data frame.")
  }
  
  # Identify factor variables
  factor_cols <- sapply(data, is.factor)
  factor_names <- names(data)[factor_cols]
  
  # Create a list to store the new data frames
  new_data_list <- list()
  
  # Iterate over each factor variable
  for (factor_variable_name in factor_names) {
    factor_variable <- data[[factor_variable_name]]
    
    # Create dummy variables using model.matrix
    dummy_matrix <- model.matrix(~ factor_variable + 0, data = data)
    dummy_df <- as.data.frame(dummy_matrix)
    colnames(dummy_df) <- paste(factor_variable_name, levels(factor_variable), sep = "_")
    
    # Drop the first level if drop_first_level is TRUE
    if (drop_first_level) {
      dummy_df <- dummy_df[, -1, drop = FALSE]
    }
    
    # Store the dummy data frame
    new_data_list[[factor_variable_name]] <- dummy_df
  }
  
  # Remove original factor columns
  data_without_factors <- data[, !factor_cols]
  
  # Combine the non-factor data with the dummy data frames
  new_data <- cbind(data_without_factors, do.call(cbind, new_data_list))
  
  return(new_data)
}


factored_dataset = to_factor(df)

glm_dataset = factors_to_dummies(factored_dataset, drop_first_level = TRUE)

factored_dataset = factors_to_dummies(factored_dataset, drop_first_level = FALSE)

factor_data = to_factor(data)
factor_data = factors_to_dummies(factor_data, drop_first_level = FALSE)

#### Call the new holdout Data and format it


new_bookings = read.csv("new_bookings.csv")
holdout_data = new_bookings[, -1]
str(new_bookings)
summary(as.factor(holdout_data$reserved_room_type))
sum(new_bookings$reserved_room_type == "P")
sum(new_bookings$reserved_room_type == "B")
holdout_data$reserved_room_type[which(new_bookings$reserved_room_type == "P")] = "A"
holdout_data$reserved_room_type[which(new_bookings$reserved_room_type == "B")] = "A"

factored_holdout = to_factor(holdout_data)

factored_holdout = factors_to_dummies(factored_holdout, drop_first_level = FALSE)
colnames(factored_holdout)



#Function to Prepare Data for knn
knn_prep <- function(data) {
  num_data <- data
  num_data[] <- lapply(num_data, function(x) {
    if(is.factor(x)) as.numeric(x) else x
  })
  scale(num_data)
}


set.seed(123)
### This is where the fun begins
### First lets Preallocate results table
k <- 5
results_table <- data.frame(
  Fold = integer(k),
  TreeAccuracy = numeric(k),
  GLMAccuracy = numeric(k),
  KNNAccuracy = numeric(k),
  CombinedAccuracy = numeric(k),
  CombinedAccuracyRules = numeric(k),
  FinalAccuracy = numeric(k),
  Best_Depth = numeric(k),
  Best_K = numeric(k),
  Optimal_threshold_glm = numeric(k),
  Optimal_final_threshold = numeric(k),
  Optimal_final_threshold_insample = numeric(k)
)


# Preallocate the Weights_Vector column as a list of length k
results_table$Weights_Vector <- vector("list", k)

### now create folds

folds <- createFolds(df$is_canceled, k = k, list = TRUE, returnTrain = FALSE)


### Hi Raanan! If you see this that means for some reason you decided to run my code.
### First off all: My condolences, secondly: at the start of the main loop theres the line
### "for (n in seq(1:5)){" 
### PLEASE REPLACE IT WITH:
###"for (n in seq(1:1)){"
### this should cut the run time to 15 minutes instead of 45


for (n in seq(1:5)){
  results_table$Fold[n] = n
  test_index <- folds[[n]]
  train_index <- setdiff(seq_len(nrow(df)), test_index)


  # 
  # sample_size = floor(0.7 * NROW(df))
  # 
  # train_index = sample(nrow(df), size = sample_size, replace = F)
  train = df[train_index,]
  test = df[-train_index,]
  
  #check for the distribution of the is_canceled variable
  hist(train$is_canceled)
  
  
  ### tree - rpart 
  tree0 = rpart(is_canceled ~ ., data = train, 
                method = "class", ## this is a regression tree!
                parms = list(split = "information") )
  rpart.plot(tree0)
  

  
  library(rpart)
  library(rpart.plot)
  
  ## Now we will find the best settings for our tree prediction
  
  ## Data frames to hold the results for each iteration
  Acc = data.frame(Depth = 1:15,
                   InSample = rep(0,15),
                   OutOfSample = rep(0,15))
  
  ## now loop through Depths
  for (depth in 1:15) {
    ##depth = 3
    tree = rpart(is_canceled ~ ., data = train, 
                 method = "class", ## this is a regression tree!
                 parms = list(split = "information"),
                 control = rpart.control(maxdepth = depth, cp = 0.001)
    )
    ##rpart.plot(tree)
    
    ## compute and save accuracies:
    treePredInSample = predict(tree, newdata= train, type="class") 
    nCorrectInSample = sum(treePredInSample==train$is_canceled)
    Acc[depth,2] = nCorrectInSample/nrow(train)
    
    treePredOutOfSample = predict(tree, newdata= test, type="class")   
    nCorrectOutOfSample = sum(treePredOutOfSample==test$is_canceled)
    Acc[depth,3] = nCorrectOutOfSample/nrow(test)
    cat("Checked depth: ", depth, "\nAccuracy: ", Acc[depth,3])
  }
  
  # Step 1: Compute the overfit gap
  Acc$Gap <- abs(Acc$InSample - Acc$OutOfSample)
  
  # Step 2: Normalize OutOfSample and Gap to range [0,1]
  # Higher OutOfSample is better, lower Gap is better
  Acc$NormOut <- (Acc$OutOfSample - min(Acc$OutOfSample)) / 
    (max(Acc$OutOfSample) - min(Acc$OutOfSample))
  
  Acc$NormGap <- (Acc$Gap - min(Acc$Gap)) / 
    (max(Acc$Gap) - min(Acc$Gap))
  
  # Step 3: Create a score: high out-of-sample, low gap
  # You can tune the weights if needed
  Acc$Score <- Acc$NormOut - Acc$NormGap  # Simple difference score
  
  # Pick best depth based on max score
  best_Depth <- Acc[which(Acc$Score == max(Acc$Score, na.rm = TRUE))[1], 1]
  
  results_table$Best_Depth[n] = best_Depth
  
  library(ggplot2)
  
  write.csv(Acc,"AccuracyInOut.csv")
  
  
  library(reshape2)
  
  meltedAcc= melt(Acc[, c(1,2,3)], id.vars = "Depth", 
                  variable.name = "Type", value.name = "Accuracy")
  
  ## we needed the melted df for plotting:
  ggplot(meltedAcc, aes(x=Depth,y=Accuracy, color = Type)) +
    geom_point() + geom_line() + 
    scale_x_discrete(limits = factor(1:15))
  
  ## Now we will make a model with the best depth - best_Depth, where the Insample and Outsample where close and highest
  
  tree_best_Depth = rpart(is_canceled ~ ., data = train, 
                          method = "class", ## this is a regression tree!
                          parms = list(split = "information"),
                          control = rpart.control(maxdepth = 7, cp = 0.001) )
  rpart.plot(tree_best_Depth)
  
  tree_best_Depth_pred = predict(tree_best_Depth, newdata = test, type = "class")
  
  nCorrect = sum(1*(test$is_canceled == tree_best_Depth_pred) )
  Accuracy.PredByTree = nCorrect/nrow(test) ### right 787% of the time!!
  
  results_table$TreeAccuracy[n] = Accuracy.PredByTree
  cat("Done with tree!\nAccuarcy: ", Accuracy.PredByTree)
  #### GLM ######
  
  str(train)
  train.index = train_index
  factored_test = factored_dataset[-train_index,] ### factored_test
  factored_train = factored_dataset[train_index,] ### factored_train
  
  m.logit0 = glm(is_canceled ~ . , data = factored_train, family = "binomial")
  
  threshold_results <- find_best_threshold(
    glm_model = m.logit0,
    test_data = factored_test,
    target_var = "is_canceled",
    metric = "accuracy"
  )
  
  
  # Get the optimal threshold
  optimal_threshold <- threshold_results$best_threshold
  results_table$Optimal_threshold_glm[n] = optimal_threshold
  print(paste("Optimal threshold:", optimal_threshold))
  
  # View all results
  print(threshold_results$all_results)
  
  # Plot threshold vs metric
  library(ggplot2)
  ggplot(threshold_results$all_results, aes(x = threshold, y = !!sym(threshold_results$metric_used))) +
    geom_line() +
    geom_vline(xintercept = optimal_threshold, color = "red", linetype = "dashed") +
    labs(title = paste("Threshold Optimization for", threshold_results$metric_used),
         y = threshold_results$metric_used)
  log_prob =  predict(m.logit0, factored_test, type = "response")
  log_pred_binary = ifelse(log_prob > optimal_threshold, 1, 0)
  
  nCorrect = sum(1*(factored_test$is_canceled == log_pred_binary) )
  Accuracy.glm0 = nCorrect/nrow(factored_test) ### right 783% of the time!!
  results_table$GLMAccuracy[n] = Accuracy.glm0 
  ### KNN ###
  cat("Done with glm!\nAccuarcy: ", Accuracy.glm0)
  
  knn_train <- knn_prep(factored_train[, -which(names(factored_train) == "is_canceled")])
  knn_test <- knn_prep(factored_test[, -which(names(factored_test) == "is_canceled")])
  knn_test = clean_NAs(knn_test)
  knn_train = clean_NAs(knn_train)
  knn_train = match_columns(knn_train, knn_test)
  knn_test = match_columns(knn_test, knn_train)
  
  train_labels <- train$is_canceled

  
  k_values <- seq(1, 30, by = 2)
  k_accuracies <- data.frame(
    K = k_values,
    Out_acc = numeric(length(k_values)),
    In_acc = numeric(length(k_values)))
  
  for (i in seq_along(k_values)) {
    knn_pred <- knn(train = knn_train, test = knn_test, cl = train$is_canceled, k=k_values[i], prob = T)
    train_knn_pred <- knn(train = knn_train, test = knn_train, cl = train$is_canceled, k=k_values[i], prob = T)
    k_accuracies[i,2] <- mean(knn_pred == factored_test$is_canceled)
    k_accuracies[i, 3] <- mean(train_knn_pred == factored_train$is_canceled)
    cat("\nChecked K: ", k_values[i], "\nAccuracy: ", k_accuracies[i,2], "\n")
  }
  
  # Plot accuracy vs k
  plot(k_values, k_accuracies[,2], type = "b", 
       xlab = "k Value", ylab = "Accuracy",
       main = "KNN Accuracy by k Value")
  
  # Step 1: Compute the overfit gap
  k_accuracies$Gap <- abs(k_accuracies$In_acc - k_accuracies$Out_acc)
  
  # Step 2: Normalize OutOfSample and Gap to range [0,1]
  # Higher OutOfSample is better, lower Gap is better
  k_accuracies$NormOut <- (k_accuracies$Out_acc - min(k_accuracies$Out_acc)) / 
    (max(k_accuracies$Out_acc) - min(k_accuracies$Out_acc))
  
  k_accuracies$NormGap <- (k_accuracies$Gap - min(k_accuracies$Gap)) / 
    (max(k_accuracies$Gap) - min(k_accuracies$Gap))
  
  # Step 3: Create a score: high out-of-sample, low gap
  # You can tune the weights if needed
  k_accuracies$Score <- k_accuracies$NormOut - k_accuracies$NormGap  # Simple difference score
  
  # Pick best depth based on max score
  best_k <- k_accuracies[which(k_accuracies$Score == max(k_accuracies$Score, na.rm = TRUE))[1], 1]
  
  
  # Select best k
  #best_k <- k_values[which.max(k_accuracies)]
  cat("the best k is: ", best_k)
  results_table$Best_K[n] = best_k
  knn_accuracy <- max(k_accuracies)
  cat("\nKNN Performance:\n")
  cat("Best k:", best_k, "\n") #15
  cat("Test Accuracy:", knn_accuracy, "\n")
  
  knnPred = knn(train = knn_train, test = knn_test, cl = train$is_canceled, k=best_k, prob = T)
  train_knnPred = knn(train = knn_train, test = knn_train, cl = train$is_canceled, k=best_k, prob = T)
  caret::confusionMatrix(data = as.factor(knnPred), reference = as.factor(test$is_canceled), positive = "1")
  Knn_accuracy = mean(knnPred == test$is_canceled)
  results_table$KNNAccuracy[n] = Knn_accuracy
  
  cat("Done with knn!\nAccuarcy:", Knn_accuracy)
  
  ### COMBINED ###
  ## train = df[train_index,]
  ## test = df[-train_index,]
  
  threshold_t = 0.2
  
  Weights = c(Accuracy.glm0, Accuracy.PredByTree,Knn_accuracy )
  results_table$Weights_Vector[n] = Weights
  Weights = Weights/sum(Weights)
  
  combined_pred = (Weights[1]*unname(log_pred_binary) + Weights[2]*unname(as.numeric(as.character(tree_best_Depth_pred))) + Weights[3]*as.numeric(as.character(knnPred)))
  
  combined_pred_binary = ifelse(combined_pred < threshold_t, 0, 1)
  any(is.na(combined_pred_binary))
  
  nCorrect = sum((test$is_canceled == combined_pred_binary) )
  Accuracy.Combined_pred = nCorrect/nrow(test) 
  
  results_table$CombinedAccuracy[n] = Accuracy.Combined_pred
  combined_pred_train = (Weights[1]*unname(predict(m.logit0, newdata = factored_train 
                                                   , type = "response")) + 
                           Weights[2]*unname(as.numeric(as.character(predict(tree_best_Depth, newdata = train, type = "class")))) +
                           Weights[3]*as.numeric(as.character(train_knnPred)))
  combined_pred_train_binary = ifelse(combined_pred_train < threshold_t, 0, 1)
  nCorrect = sum((train$is_canceled == combined_pred_train_binary) )
  InSampleAccuracy.Combined_pred = nCorrect/nrow(train)
  results_table$InSampleAccuracy.Combined_pred[n] = InSampleAccuracy.Combined_pred
  # Lets find the best threshold
  # Create empty vectors to store accuracies
  # thresholds_list <- seq(0.01, 1, by = 0.01)
  thresholds_list <- seq(0.1, 1, by = 0.1)
  in_sample_accuracies <- numeric(length(thresholds_list))
  out_of_sample_accuracies <- numeric(length(thresholds_list))
  
  # Loop through each threshold
  for (i in seq_along(thresholds_list)) {
    current_threshold <- thresholds_list[i]
    
    # Call the function with the current threshold
    results <- calculate_model_accuracy(
      train_data = train,
      test_data = test,
      model_logit = m.logit0,
      model_tree = tree_best_Depth,
      logit_data_train = factored_dataset[train_index, ], ### factored_train
      logit_data_test = factored_dataset[-train_index, ], ###factored_test
      tree_data_train = train,
      tree_data_test = test,
      knn_train_features = knn_train,
      knn_test_features = knn_test,
      train_target = train$is_canceled,
      test_target = test$is_canceled,
      knnPred = knn_pred,
      train_knnPred = train_knn_pred,
      best_k = best_k,
      model_accuracies = c(Accuracy.glm0, Accuracy.PredByTree, Knn_accuracy),
      threshold = current_threshold
    )
    
    # Store the accuracies
    in_sample_accuracies[i] <- results$train_accuracy
    out_of_sample_accuracies[i] <- results$test_accuracy
    cat("Threshold:", current_threshold, "\nIn Sample Accuarcy:", in_sample_accuracies[i],
        "\nOut of Sample Accuracy:", out_of_sample_accuracies[i])
  }
  
  # Create a data frame for plotting
  accuracy_df <- data.frame(
    Threshold = thresholds_list,
    InSample = in_sample_accuracies,
    OutOfSample = out_of_sample_accuracies
  )
  
  # Pick best depth based on max acc
  best_final_threshold <- accuracy_df[which(accuracy_df$OutOfSample == max(accuracy_df$OutOfSample, na.rm = TRUE))[1], 1]
  best_final_threshold_insample <- accuracy_df[which(accuracy_df$InSample == max(accuracy_df$InSample, na.rm = TRUE))[1], 1]
  # Plotting the accuracies using ggplot2
  ggplot(accuracy_df, aes(x = Threshold)) +
    geom_line(aes(y = InSample, color = "In-Sample Accuracy")) +
    geom_point(aes(y = InSample, color = "In-Sample Accuracy")) +
    geom_line(aes(y = OutOfSample, color = "Out-of-Sample Accuracy")) +
    geom_point(aes(y = OutOfSample, color = "Out-of-Sample Accuracy")) +
    labs(
      title = "In-Sample vs. Out-of-Sample Accuracy Across Thresholds",
      x = "Prediction Threshold",
      y = "Accuracy",
      color = "Accuracy Type"
    ) +
    scale_color_manual(values = c(
      "In-Sample Accuracy" = "blue",
      "Out-of-Sample Accuracy" = "red"
    )) +
    theme_minimal() +
    theme(legend.position = "bottom")
  
  
  ## Zooming in
  # Plotting the accuracies using ggplot2
  ggplot(accuracy_df, aes(x = Threshold)) +
    geom_line(aes(y = InSample, color = "In-Sample Accuracy")) +
    geom_point(aes(y = InSample, color = "In-Sample Accuracy")) +
    geom_line(aes(y = OutOfSample, color = "Out-of-Sample Accuracy")) +
    geom_point(aes(y = OutOfSample, color = "Out-of-Sample Accuracy")) +
    labs(
      title = "In-Sample vs. Out-of-Sample Accuracy Across Thresholds",
      x = "Prediction Threshold",
      y = "Accuracy",
      color = "Accuracy Type"
    ) +
    scale_color_manual(values = c(
      "In-Sample Accuracy" = "blue",
      "Out-of-Sample Accuracy" = "red"
    )) +
    theme_minimal() +
    theme(legend.position = "bottom") +
    
    xlim(0.1, 0.5)
  ### As we can see the optimal threshold is between 0.2 and 0.4
  ## we will re-run the functions with this in mind
  
  caret::confusionMatrix(data = as.factor(combined_pred_binary), reference = as.factor(test$is_canceled), positive = "1")
  
  
  # Create empty vectors to store accuracies
  # thresholds_list <- seq(0.01, 1, by = 0.01)
  thresholds_list <- seq(max(best_final_threshold-0.1,0), min(best_final_threshold+0.1,1), by = 0.01)
  in_sample_accuracies <- numeric(length(thresholds_list))
  out_of_sample_accuracies <- numeric(length(thresholds_list))
  
  # Loop through each threshold
  for (i in seq_along(thresholds_list)) {
    current_threshold <- thresholds_list[i]
    
    # Call the function with the current threshold
    results <- calculate_model_accuracy(
      train_data = train,
      test_data = test,
      model_logit = m.logit0,
      model_tree = tree_best_Depth,
      logit_data_train = factored_dataset[train_index, ], ##factored_train
      logit_data_test = factored_dataset[-train_index, ], ##factored_test
      tree_data_train = train,
      tree_data_test = test,
      knn_train_features = knn_train,
      knn_test_features = knn_test,
      train_target = train$is_canceled,
      test_target = test$is_canceled,
      knnPred = knnPred,
      train_knnPred = train_knnPred,
      best_k = best_k,
      model_accuracies = c(Accuracy.glm0, Accuracy.PredByTree, Knn_accuracy),
      threshold = current_threshold
    )
    
    # Store the accuracies
    in_sample_accuracies[i] <- results$train_accuracy
    out_of_sample_accuracies[i] <- results$test_accuracy
    cat("Threshold:", current_threshold, "\nIn Sample Accuarcy:", in_sample_accuracies[i],
        "\nOut of Sample Accuracy:", out_of_sample_accuracies[i])
  }
  
  # Create a data frame for plotting
  accuracy_df <- data.frame(
    Threshold = thresholds_list,
    InSample = in_sample_accuracies,
    OutOfSample = out_of_sample_accuracies
  )
  
  
  # Pick best depth based on max acc
  best_final_threshold <- accuracy_df[which(accuracy_df$OutOfSample == max(accuracy_df$OutOfSample, na.rm = TRUE))[1], 1]
  best_final_threshold_insample <- accuracy_df[which(accuracy_df$InSample == max(accuracy_df$InSample, na.rm = TRUE))[1], 1]
  
  results_table$Optimal_final_threshold[n] = best_final_threshold
  results_table$Optimal_final_threshold_insample[n] = best_final_threshold_insample
  
  cat("Done with threshold loop!\nBest Threshold: ", best_final_threshold)
  
  combined_pred_binary = ifelse(combined_pred < best_final_threshold, 0, 1)
  
  ### Combined with rules ###
  library(dplyr)
  
  combined_pred_with_rules  <- case_when(
    test$deposit_type == "Non Refund" ~ 1,
    test$lead_time < 12 ~ 0,
    TRUE ~ combined_pred_binary
  )
  
  
  sum(combined_pred_with_rules)
  
  sum(test$is_canceled)
  
  nCorrect = sum((test$is_canceled == combined_pred_with_rules) )
  Accuracy.combined_pred_with_rules = nCorrect/nrow(test) ### right 75% of the time!!
  
  results_table$CombinedAccuracyRules[n] = Accuracy.combined_pred_with_rules
  
  caret::confusionMatrix(data = as.factor(combined_pred_with_rules), reference = as.factor(test$is_canceled), positive = "1")
  caret::confusionMatrix(data = as.factor(tree_best_Depth_pred), reference = as.factor(test$is_canceled), positive = "1")
  caret::confusionMatrix(data = as.factor(knnPred), reference = as.factor(test$is_canceled), positive = "1")
  caret::confusionMatrix(data = as.factor(log_pred_binary), reference = as.factor(test$is_canceled), positive = "1")
  
  ### Run the model on all the data #########
  Final_tree = rpart(is_canceled ~ ., data = df, 
                    method = "class", ## this is a regression tree!
                    parms = list(split = "information"),
                    control = rpart.control(maxdepth = 7, cp = 0.001) )
  rpart.plot(Final_tree)
  Final_tree_pred = predict(Final_tree, newdata = df, type = "class")
  
  nCorrect_FTree = sum(1*(df$is_canceled == Final_tree_pred) )
  Acc_FTree = nCorrect_FTree/nrow(df)
  
  Final_glm = glm(is_canceled ~ . , data = factored_dataset, family = "binomial")
  Final_glm_pred = predict(Final_glm, newdata = factored_dataset, type = "response")
  
  knn_final_data <- knn_prep(factored_dataset[, -which(names(test) == "is_canceled")])
  train_labels <- factored_dataset$is_canceled
  knn_final_data = clean_NAs(knn_final_data)
  Final_knn_Pred = knn(train = knn_final_data, test = knn_final_data, cl = factored_dataset$is_canceled, k=best_k, prob = T)
  
  nCorrect_FKnn = sum(1*(factored_dataset$is_canceled == Final_knn_Pred) )
  Acc_FKnn = nCorrect_FKnn/nrow(factored_dataset)
  
  threshold_results <- find_best_threshold(
    glm_model = Final_glm,
    test_data = factored_dataset,
    target_var = "is_canceled",
    metric = "accuracy"
  )
  
  
  # Get the optimal threshold
  GLM_Final_threshold <- threshold_results$best_threshold
  print(paste("Optimal threshold:", GLM_Final_threshold))
  
  Final_glm_pred_binary = ifelse(Final_glm_pred > GLM_Final_threshold, 1, 0)
  
  nCorrect_FGLM = sum(1*(factored_dataset$is_canceled == Final_glm_pred_binary) )
  Acc_FGLM = nCorrect_FGLM/nrow(factored_dataset)
  
  threshold_Final = best_final_threshold
  
  Weights = c(Acc_FGLM, Acc_FTree,Acc_FKnn )
  Weights = Weights/sum(Weights)
  
  Final_pred = (Weights[1]*unname(Final_glm_pred) + Weights[2]*unname(as.numeric(as.character(Final_tree_pred))) + Weights[3]*as.numeric(as.character(Final_knn_Pred)))
  
  Final_pred_binary = ifelse(Final_pred < threshold_Final, 0, 1)
  any(is.na(combined_pred_binary))
  
  Final_pred_with_rules <- case_when(
    df$deposit_type == "Non Refund" ~ 1,
    df$lead_time < 12 ~ 0,
    TRUE ~ Final_pred_binary
  )
  nCorrect = sum((df$is_canceled == Final_pred_with_rules) )
  Accuracy.Final = nCorrect/nrow(df) 
  results_table$FinalAccuracy[n] = Accuracy.Final
  mean(Final_pred_with_rules)
  mean(df$is_canceled)
  
  ## We are still missing a large percent of positives, on avrage 5%, but assuimng that
  ## The amount of positives is largely the same within the holdout data, we shouldn't get bad results.
  
  caret::confusionMatrix(data = as.factor(Final_pred_with_rules), reference = as.factor(df$is_canceled), positive = "1")
  message("########## Finished with round ", n, "##########")
}

write.csv(results_table[, -13], file = "Results_cross_validation.csv")
##########################################################################################

### Analyze the results table

summary(results_table)
### The best k seems to be 11
### The best depth really is 7
sd(results_table$TreeAccuracy)
sd(results_table$GLMAccuracy)
sd(results_table$KNNAccuracy)
sd(results_table$CombinedAccuracy)
sd(results_table$CombinedAccuracyRules)
sd(results_table$FinalAccuracy)
sd(results_table$Best_Depth)
sd(results_table$Best_K)
sd(results_table$Optimal_threshold_glm)
sd(results_table$Optimal_final_threshold)
sd(results_table$InSampleAccuracy.Combined_pred)

##########################################################################################

### Run the model on all the data #########
Final_tree = rpart(is_canceled ~ ., data = df, 
                   method = "class", ## this is a regression tree!
                   parms = list(split = "information"),
                   control = rpart.control(maxdepth = 8, cp = 0.001) )
rpart.plot(Final_tree)
Final_tree_pred = predict(Final_tree, newdata = df, type = "class")

nCorrect_FTree = sum(1*(df$is_canceled == Final_tree_pred) )
Acc_FTree = nCorrect_FTree/nrow(df)

Final_glm = glm(is_canceled ~ . , data = factored_dataset, family = "binomial")
Final_glm_pred = predict(Final_glm, newdata = factored_dataset, type = "response")

knn_final_data <- knn_prep(factored_dataset[, -which(names(test) == "is_canceled")])
train_labels <- factored_dataset$is_canceled
knn_final_data = clean_NAs(knn_final_data)
Final_knn_Pred = knn(train = knn_final_data, test = knn_final_data, cl = factored_dataset$is_canceled, k=best_k, prob = T)

nCorrect_FKnn = sum(1*(factored_dataset$is_canceled == Final_knn_Pred) )
Acc_FKnn = nCorrect_FKnn/nrow(factored_dataset)

threshold_results <- find_best_threshold(
  glm_model = Final_glm,
  test_data = factor_data,
  target_var = "is_canceled",
  metric = "accuracy"
)


# Get the optimal threshold
GLM_Final_threshold <- threshold_results$best_threshold
print(paste("Optimal threshold:", GLM_Final_threshold))

Final_glm_pred_binary = ifelse(Final_glm_pred > GLM_Final_threshold, 1, 0)

nCorrect_FGLM = sum(1*(factored_dataset$is_canceled == Final_glm_pred_binary) )
Acc_FGLM = nCorrect_FGLM/nrow(factored_dataset)

### The best threshoold was 0.47, HOWEVER, the pos predictive value 
### has been pretty low all along, so we can asume that when trying to maximise the 
### Total accuaracy some overfitting has occured, to remedey this and 
### To maximise our overall accuracy the solution is simple
### We will set the threshold to be lower, guessing less ones and reducing the acuracy of 0s 
### is a good price to pay when our ned prediction values are 80+
### and in return we get a few extra precents in pos predictive value, overall improving our accuracy
### Just as we wanted
threshold_Final = 0.2

Weights = c(Acc_FGLM, Acc_FTree,Acc_FKnn )
Weights = Weights/sum(Weights)

Final_pred = (Weights[1]*unname(Final_glm_pred) + Weights[2]*unname(as.numeric(as.character(Final_tree_pred))) + Weights[3]*as.numeric(as.character(Final_knn_Pred)))

Final_pred_binary = ifelse(Final_pred < threshold_Final, 0, 1)
any(is.na(combined_pred_binary))

Final_pred_with_rules <- case_when(
  df$deposit_type == "Non Refund" ~ 1,
  df$lead_time < 12 ~ 0,
  TRUE ~ Final_pred_binary
)

nCorrect = sum((df$is_canceled == Final_pred_with_rules) )
Accuracy.Final = nCorrect/nrow(df) 

mean(Final_pred_with_rules)
mean(df$is_canceled)

## We are still missing a large percent of positives, on avrage 5%, but assuimng that
## The amount of positives is largely the same within the holdout data, we shouldn't get bad results.
Final_pred_with_rules = append(Final_pred_with_rules, 0, 22332)
caret::confusionMatrix(data = as.factor(Final_pred_with_rules), reference = as.factor(data$is_canceled), positive = "1")


### Now its time to output the submission


### Run te models on the new data ###
Submit_tree_pred = predict(Final_tree, newdata = holdout_data, type = "class")

Submit_glm_pred = predict(Final_glm, newdata = factored_holdout, type = "response")

knn_final_data <- knn_prep(factor_data[, -which(names(factor_data) == "is_canceled")])
knnFinalTest <- knn_prep(factored_holdout)
knn_final_data = clean_NAs(knn_final_data)
knnFinalTest = clean_NAs(knnFinalTest)
knn_final_data = match_columns(knn_final_data, knnFinalTest)
knnFinalTest = match_columns(knnFinalTest, knn_final_data)
train_labels <- factor_data$is_canceled
ncol(knn_final_data)
ncol(knnFinalTest)
nrow(knn_final_data)
nrow(knnFinalTest)
Submit_knn_Pred = knn(train = knn_final_data, test = knnFinalTest, cl = factor_data$is_canceled, k=best_k, prob = T)

Submit_glm_pred_binary = ifelse(Submit_glm_pred > GLM_Final_threshold, 1, 0)

#### THIS IS THE FINAL DECISION
threshold_Final = 0.2

Weights = c(Acc_FGLM, Acc_FTree,Acc_FKnn )
Weights = Weights/sum(Weights)

Submit_pred = (Weights[1]*unname(Submit_glm_pred) + Weights[2]*unname(as.numeric(as.character(Submit_tree_pred))) + Weights[3]*as.numeric(as.character(Submit_knn_Pred)))

Submit_pred_binary = ifelse(Submit_pred < threshold_Final, 0, 1)
any(is.na(Submit_pred_binary))

Prediction <- case_when(
  holdout_data$deposit_type == "Non Refund" ~ 1,
  holdout_data$lead_time < 12 ~ 0,
  TRUE ~ Submit_pred_binary
)

mean(Prediction)
mean(Final_pred_with_rules)
mean(data$is_canceled)
any(is.na(Prediction))
output = cbind(Number_Id = new_bookings[,1], Prediction = Prediction)
output = data.frame(output)
summary(output)
write.csv(output, "Predictions.csv", row.names = FALSE)
