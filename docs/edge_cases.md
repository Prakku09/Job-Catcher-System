# Edge Cases Analysis

When evaluating the `production_binary_classifier.pkl` against live traffic, several edge cases may arise that are not adequately represented in the training data.

## 1. NaN or Null Values
If numeric scores (e.g., `python_score`, `sql_score`) are missing, the current `StandardScaler` does not natively impute them, which will crash the pipeline.
- **Resolution:** A `SimpleImputer(strategy='median')` must be appended prior to the scaler in the `ColumnTransformer`, OR the API backend must reject nulls. Currently, the system assumes the backend API strictly enforces non-null types.

## 2. Extreme Outliers
If a user submits an expected salary of `$1,000,000,000`, the scaled feature will skew massively, potentially forcing the logistic regression into an extreme uncalibrated probability zone.
- **Resolution:** Implement boundary clipping in the API (e.g., cap salary to reasonable market limits) or rely on the `JobMatchPreprocessor` to clip data during feature engineering.

## 3. Empty Prediction Requests
An empty JSON `{}` submitted to the API.
- **Resolution:** This is treated as a validation failure. The API will intercept this and return an HTTP 422 before the data reaches the ML pipeline.

## 4. Unknown Categorical Features
If a new `education_level` (e.g., `Post-Doctorate`) appears that wasn't in the training data, the `OneHotEncoder` (if present) might throw a `ValueError`.
- **Resolution:** Ensure any categorical encoders are initialized with `handle_unknown='ignore'`. (Note: In this specific feature set, we are manually mapping strings or focusing purely on numeric feature differences, but this rule applies generally to production ML).

## 5. Negative Feature Values
Scores like `years_experience` cannot be negative.
- **Resolution:** The API schema must enforce `min_value=0`. If it bypasses the API, the model will treat it mathematically, which is undefined behavior for the business logic.
