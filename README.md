# Business-analytics-Final-Exam
My solution (`Code.R`) implements a comprehensive machine learning pipeline for predicting hotel booking cancellations. It leverages an ensemble approach combining Logistic Regression, Decision Trees, and K-Nearest Neighbors (KNN) models. The script also includes data preprocessing, hyperparameter tuning, cross-validation, and a final prediction generation for new, unseen booking data.

## Table of Contents

* [Project Overview](#project-overview)

* [Files Included](#files-included)

* [Key Libraries](#key-libraries)

* [Data](#data)

* [Preprocessing Steps](#preprocessing-steps)

* [Functions Overview](#functions-overview)

* [Model Training and Evaluation](#model-training-and-evaluation)

* [Ensemble Modeling](#ensemble-modeling)

* [Prediction Generation](#prediction-generation)

* [How to Run](#how-to-run)

* [Important Notes](#important-notes)

## Project Overview

The main objective of this project is to predict whether a hotel booking will be canceled (`is_canceled` variable). The script performs the following key tasks:

1. **Data Loading and Cleaning**: Reads the hotel booking data, removes duplicates, handles outliers, and checks for missing values.

2. **Feature Engineering**: Converts categorical variables to factors and then to dummy variables suitable for modeling.

3. **Model Development**:

   * Trains and tunes a Decision Tree model.

   * Trains a Logistic Regression model and optimizes its prediction threshold.

   * Trains and tunes a K-Nearest Neighbors (KNN) model.

4. **Ensemble Prediction**: Combines the predictions from the three models using a weighted average, with weights derived from individual model accuracies. It also applies business rules to enhance prediction accuracy.

5. **Cross-Validation**: Evaluates the models' performance using k-fold cross-validation.

6. **Final Prediction**: Generates predictions on a new holdout dataset and exports them to a CSV file.

## Files Included

* `Code.R`: The main R script containing all the data processing, model training, and prediction logic.

* `hotel.csv`: (Expected) The primary dataset containing historical hotel booking information, including the `is_canceled` target variable.

* `new_bookings.csv`: (Expected) A holdout dataset containing new booking information for which predictions need to be generated.

## Key Libraries

The script utilizes several R libraries for data manipulation, modeling, and visualization:

* `infotheo`: For information theory-based calculations (though not extensively used in the main logic presented).

* `tidyr`: For tidying data.

* `dplyr`: For data manipulation.

* `stargazer`: For producing well-formatted regression tables (not directly used in output, but imported).

* `rpart`: For building decision trees.

* `rpart.plot`: For plotting decision trees.

* `FNN`: For K-Nearest Neighbors algorithm.

* `caret`: For machine learning utilities, including `createFolds` for cross-validation and `confusionMatrix` for evaluation.

* `e1071`: For support vector machines and other statistical functions (not directly used for main models, but loaded).

* `class`: For the `knn` function.

* `knitr`: For dynamic report generation (not directly used in output, but loaded).

* `ggplot2`: For creating visualizations (e.g., accuracy plots).

* `reshape2`: For reshaping data frames.

## Data

The script expects two CSV files:

* `hotel.csv`: Contains the training data. The target variable is `is_canceled`.

* `new_bookings.csv`: Contains the data for which predictions are to be made. It should have the same structure as `hotel.csv` excluding the `is_canceled` column.

## Preprocessing Steps

1. **Duplicate Removal**: Removes duplicate rows from the `data` dataframe.

2. **Index Removal**: The first column (assumed to be an index) is removed.

3. **Outlier Handling**: Filters out rows where `distribution_channel` is "Undefined".

4. **NA Check**: Confirms there are no `NA` values (as per the code's comment).

5. **Categorical to Factor**: Converts all character columns to factor variables.

6. **Factor to Dummy Variables**: Converts factor variables into binary dummy variables.

7. **KNN Data Preparation**: Numerical features are scaled, and factor variables are converted to numeric for KNN. Missing values are handled by either dropping rows or columns with `NA`s, based on the distribution of `NA`s.

## Functions Overview

The `Code.R` script contains several custom R functions to facilitate data preprocessing, model evaluation, and threshold optimization.

### `match_columns(df1, df2)`

* **Purpose**: Ensures that two dataframes have the same set of columns by retaining only the common columns. This is crucial for consistency when preparing data for models, especially between training and test sets.

* **Parameters**:

  * `df1`: The first dataframe.

  * `df2`: The second dataframe.

* **Returns**: `df1` with only the columns present in both `df1` and `df2`.

### `clean_NAs(data_to_clean)`

* **Purpose**: Handles missing (`NA`) values in a dataframe. It decides whether to remove rows or columns containing `NA`s based on which approach removes fewer `NA`s overall (i.e., if row-wise removal is more efficient than column-wise, it removes rows, otherwise columns).

* **Parameters**:

  * `data_to_clean`: The dataframe to be cleaned.

* **Returns**: The dataframe with `NA` values handled.

### `calculate_model_accuracy(...)`

* **Purpose**: A comprehensive function to calculate the combined (ensemble) accuracy for both in-sample (training) and out-of-sample (test) predictions across the Logistic Regression, Decision Tree, and KNN models. It takes individual model accuracies to weight their contributions to the ensemble.

* **Parameters**:

  * `train_data`: Base training dataset.

  * `test_data`: Base test dataset.

  * `model_logit`: Trained Logistic Regression model.

  * `model_tree`: Trained Decision Tree model.

  * `logit_data_train`: Dataset for logistic regression training predictions.

  * `logit_data_test`: Dataset for logistic regression test predictions.

  * `tree_data_train`: Dataset for tree training predictions.

  * `tree_data_test`: Dataset for tree test predictions.

  * `knn_train_features`: Feature set for KNN training.

  * `knn_test_features`: Feature set for KNN testing.

  * `train_target`: Training target variable (`is_canceled`).

  * `test_target`: Test target variable (`is_canceled`).

  * `knnPred`: Out-of-sample KNN predictions.

  * `train_knnPred`: In-sample KNN predictions.

  * `best_k`: Optimal `k` value for KNN.

  * `model_accuracies`: Vector of pre-calculated individual model accuracies (e.g., `c(Accuracy.glm0, Accuracy.PredByTree, Knn_accuracy)`).

  * `threshold`: Decision threshold to convert combined probabilities into binary predictions (default is 0.5).

* **Returns**: A list containing `test_accuracy`, `train_accuracy`, `weights`, and `threshold`.

### `find_best_threshold(glm_model, test_data, target_var, metric = "f1", n_thresholds = 100)`

* **Purpose**: Identifies the optimal prediction threshold for a given logistic regression model. It evaluates various thresholds (from 0.01 to 0.99) and selects the one that maximizes a specified evaluation `metric` (e.g., "accuracy", "f1", "precision", "recall", "specificity", "balanced_accuracy"). The function also generates a plot to visualize the metric's performance across different thresholds, indicating the chosen optimal threshold.

* **Parameters**:

  * `glm_model`: The trained logistic regression model.

  * `test_data`: The dataset used to evaluate thresholds.

  * `target_var`: The name of the target variable (e.g., "is_canceled").

  * `metric`: The evaluation metric to optimize (default is "f1").

  * `n_thresholds`: Number of thresholds to evaluate (default is 100).

* **Returns**: A list containing `best_threshold`, `best_metric_value`, `all_results` (a dataframe of metrics for all thresholds), and `metric_used`. A `ggplot2` plot is also generated as a side effect.

### `factor_to_dummy(data, factor_variable_name, drop_first_level = TRUE)`

* **Purpose**: Converts a single specified factor variable in a dataframe into binary dummy variables. Optionally, it can drop the first level of the factor to avoid multicollinearity.

* **Parameters**:

  * `data`: The input dataframe.

  * `factor_variable_name`: The name of the factor variable to convert.

  * `drop_first_level`: Logical, whether to drop the first dummy variable level (default is `TRUE`).

* **Returns**: The dataframe with the original factor variable replaced by its dummy counterparts.

### `to_factor(data)`

* **Purpose**: Iterates through a dataframe and converts all character columns to factor variables. This is a common preprocessing step before creating dummy variables or fitting models that require factors.

* **Parameters**:

  * `data`: The input dataframe.

* **Returns**: The dataframe with character columns converted to factors.

### `factors_to_dummies(data, drop_first_level = TRUE)`

* **Purpose**: Extends `factor_to_dummy` by converting *all* factor variables in a dataframe into binary dummy variables.

* **Parameters**:

  * `data`: The input dataframe.

  * `drop_first_level`: Logical, whether to drop the first dummy variable level for each factor (default is `TRUE`).

* **Returns**: The dataframe with all original factor columns replaced by their dummy counterparts.

### `knn_prep(data)`

* **Purpose**: Prepares a dataframe for use with the K-Nearest Neighbors (KNN) algorithm. It converts any factor variables to numeric and then scales all numerical columns.

* **Parameters**:

  * `data`: The dataframe to prepare for KNN.

* **Returns**: The scaled numerical dataframe suitable for KNN.

## Model Training and Evaluation

The script trains and evaluates three distinct models:

### 1. Decision Tree (`rpart`)

* Hyperparameter tuning is performed to find the `maxdepth` that balances in-sample and out-of-sample accuracy.

* **Graph Generated**: A `ggplot2` line plot (`meltedAcc`) is generated to visualize the in-sample and out-of-sample accuracies across different `maxdepth` values (from 1 to 15). This plot helps in visually identifying the optimal depth where the accuracy is high and the gap between in-sample and out-of-sample accuracy (indicating overfitting) is minimal.

* Accuracy is calculated for both training and test sets.

### 2. Logistic Regression (`glm`)

* A logistic regression model is built using dummy-encoded features.

* An optimal probability threshold for binary classification is determined by maximizing accuracy on the test set using the `find_best_threshold` function.

* **Graph Generated**: A `ggplot2` line plot (generated by `find_best_threshold`) illustrates how the chosen evaluation metric (e.g., accuracy) changes across different prediction thresholds, with a vertical dashed red line marking the optimal threshold.

### 3. K-Nearest Neighbors (`FNN::knn`)

* The optimal `k` value for KNN is found by evaluating accuracy across a range of `k` values (from 1 to 30, with a step of 2), aiming to minimize the gap between in-sample and out-of-sample accuracy. A scoring mechanism is used to pick the best `k` based on maximizing normalized out-of-sample accuracy and minimizing normalized gap.

* **Graph Generated**: A base R plot is generated to visualize the out-of-sample accuracy against different `k` values, helping to identify the `k` that yields the best performance.

## Ensemble Modeling

The predictions from Logistic Regression, Decision Tree, and KNN models are combined using a weighted average. The weights for each model are proportional to their individual test accuracies during cross-validation.

* **Threshold Optimization for Ensemble**: The script iteratively searches for the optimal combined prediction threshold. It evaluates a range of thresholds and generates a `ggplot2` plot comparing in-sample and out-of-sample accuracies for the ensemble across these thresholds. This helps in selecting a threshold that maximizes overall accuracy and reduces potential overfitting. The search is refined by focusing on a narrower range around an initial best threshold.

Additionally, specific business rules are applied to the final predictions:

* If `deposit_type` is "Non Refund", the prediction is forced to `1` (canceled).

* If `lead_time` is less than 12 days, the prediction is forced to `0` (not canceled).

* Otherwise, the ensemble model's binary prediction is used.

## Prediction Generation

After cross-validation, the models are re-trained on the entire dataset (`df`) to maximize the use of available data. The optimal parameters (e.g., `best_k`, `best_Depth`, `optimal_threshold_glm`, `best_final_threshold`) found during cross-validation are then applied to these final models.

The final ensemble model, including the business rules, is used to predict cancellations on the `new_bookings.csv` (holdout) dataset. The predictions are then saved to `Predictions.csv`.

## How to Run

1. **Prerequisites**: Ensure you have R installed and an R environment (like RStudio) set up.

2. **Install Libraries**: Make sure all the necessary R packages listed under [Key Libraries](#key-libraries) are installed. You can install them using `install.packages("package_name")`.

3. **Data Files**: Place `hotel.csv` and `new_bookings.csv` in the same directory as `Code.R`, or modify the `read.csv` paths accordingly.

4. **Execute Script**: Run the `Code.R` script. The cross-validation loop is set to `5` folds (`seq(1:5)`). For faster execution during testing, you might want to temporarily change `for (n in seq(1:5))` to `for (n in seq(1:1))` as suggested in the code comments.

5. **Output**: The script will generate:

   * `Results_cross_validation.csv`: A CSV file detailing the accuracy and optimal parameters found during each fold of cross-validation.

   * `Predictions.csv`: A CSV file with two columns: `Number_Id` (from `new_bookings.csv`) and `Prediction` (0 for not canceled, 1 for canceled).

## Important Notes

* **Runtime**: The cross-validation loop (currently set to 5 folds) can take a significant amount of time to run (e.g., 45 minutes). Adjusting the loop to `seq(1:1)` will reduce this to approximately 15 minutes for a quick test run.

* **Threshold Optimization**: The script includes a detailed process for finding the optimal prediction thresholds for both the GLM model and the final ensemble, considering both in-sample and out-of-sample accuracies, visualized by generated plots.

* **Business Rules**: The `case_when` statements for `deposit_type` and `lead_time` apply specific business logic that overrides model predictions under certain conditions, which can be crucial for real-world scenarios.

* **Model Re-training**: The final models are re-trained on the full dataset (`df`) after hyperparameter tuning using cross-validation to leverage all available data for the most robust final predictions.
