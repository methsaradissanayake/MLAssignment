# Machine Learning Project Report Outline: Sri Lanka Vehicle Price Prediction

## 1. Introduction and Problem Definition
- **Objective:** Predict the price of used and new vehicles in Sri Lanka using scraped data.
- **Task Type:** Regression.
- **Target Variable:** `price`.
- **Dataset Context:** Data collected from ikman.lk vehicle advertisements, preprocessed to handle Sri Lankan price formats (Rs, Lakh, Mn), currency symbols, missing values, and location encoding.

## 2. Algorithm Selection
- **Algorithm Chosen:** XGBoost (Extreme Gradient Boosting Regressor).
- **Justification:**
  - **Handling Non-linearities:** Vehicle prices are heavily non-linear (e.g., depreciation initially happens quickly and then plateaus). XGBoost captures these complex, non-linear interactions natively compared to basic models like Linear Regression.
  - **Handling Categorical & Missing Data:** XGBoost can handle sparse data structures efficiently, and with proper encoding (Target encoding/One-Hot), it performs exceptionally well on categorical variables like Brand, Model, and Location.
  - **Performance and Speed:** XGBoost leverages tree pruning, parallel tree building, and hardware optimization, making it faster and more scalable than Random Forest.
  - **Versus Basic Models:** A basic Linear Regression model assumes a linear relationship and struggles with high-cardinality categorical variables. A standard Decision Tree severely overfits. XGBoost mitigates overfitting via regularization parameters (`alpha`, `lambda`, `eta`) while maintaining high predictive power.

## 3. Data Preprocessing and Setup
- **Data Splitting Strategy:**
  - Training Data: 70%
  - Validation Data: 15% (Used for early stopping during training and hyperparameter tuning).
  - Test Data: 15% (Strictly kept unseen for final evaluation to avoid data leakage).
- **Feature Engineering:** Handling text anomalies, location binary encodings, extracting years, and frequency/target encoding for high-cardinality features.

## 4. Model Training and Hyperparameter Tuning
- **Optimization Strategy:** Optuna.
  - Optuna is chosen over `RandomizedSearchCV` due to its use of the Tree-structured Parzen Estimator (TPE) algorithm. TPE uses sequential model-based optimization, meaning it learns from previous trials to narrow down the search space more effectively than random guessing.
- **Early Stopping:** Implemented using the Validation set. If the validation error does not improve for a set number of rounds, training is halted to prevent overfitting.

## 5. Model Evaluation and Results
- **Metrics Table:**
  - **RMSE (Root Mean Squared Error):** Indicates the average magnitude of the error in the predicted prices, penalizing larger errors heavily.
  - **MAE (Mean Absolute Error):** Indicates the absolute average of errors across all predictions; highly interpretable as the "average off-by amount".
  - **R-squared (R2):** The proportion of the variance in the predicted prices that can be explained by the dataset features.
- **Visualizations (Included in final report):**
  - Predicted vs. Actual Scatter Plot: To visualize heteroscedasticity and overall fit.
  - Residuals Histogram: To check if prediction errors are normally distributed.

## 6. Model Explainability
- **Method Chosen:** SHAP (SHapley Additive exPlanations).
- **Global Explanations:**
  - **Feature Importance Bar Chart:** Shows the absolute SHAP value impact of features across the entire dataset.
  - **SHAP Summary Plot:** Shows not only importance but also the directional impact (e.g., higher 'Year' pushes 'Price' up).
  - **SHAP Dependence Plot:** Specifically examines the interaction and isolated effect of the single most important feature on the prediction output.
- **Local Explanations:**
  - Utilizing SHAP Force Plots / Waterfall Plots in the Streamlit App to explain individual price predictions to the end-user.

## 7. Operationalization (Bonus)
- **Front-End Deployment:** A Streamlit web application providing a user-friendly interface to input vehicle parameters (Brand, Model, Year, Condition) and outputting both the prediction and its corresponding SHAP explanation.
- **Scraping Script:** Documentation on the `scrape.py` utility for scalable data acquisition.
