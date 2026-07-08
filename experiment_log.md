# Experiment Log: PlaceMux Difficulty Predictor

## Run 1: First Predictive Model

**Date:** 2026-07-02
**Target:** Question difficulty (Easy, Medium, Hard based on 33.3% and 66.7% percentiles of `difficulty_level`)

### Architecture & Setup
- **Model:** Logistic Regression (class_weight='balanced', max_iter=1000)
- **Features:** TF-IDF on combined text (`domain` + `topic` + `question_text`), max_features=5000
- **Data Splits:** Train: 629 | Validation: 136 | Test: 135
- **Random Seed:** 42

### Baseline (Majority Class: Easy)
- **Validation Accuracy:** 0.3456
- **Validation Macro F1:** 0.1712

### Model Results
- **Validation Accuracy:** 0.4191
- **Validation Macro F1:** 0.3987

> [!TIP]
> **Conclusion:** The first model successfully beats the dumb majority-class baseline. Accuracy improved by ~7.3% and Macro F1 more than doubled.

### Error Analysis
Top 5 most confident errors on the validation set:

1. **True:** Medium | **Predicted:** Hard | **Confidence:** 0.6841
   - Text: `AI & ML ML Basics What is the silhouette score?...`
2. **True:** Hard | **Predicted:** Easy | **Confidence:** 0.6579
   - Text: `AI & ML Preprocessing What is PCA whitening?...`
3. **True:** Easy | **Predicted:** Hard | **Confidence:** 0.6423
   - Text: `AI & ML ML Formula Variance: Var(X) = ?...`
4. **True:** Hard | **Predicted:** Easy | **Confidence:** 0.6422
   - Text: `AI & ML Preprocessing MaxAbsScaler:...`
5. **True:** Medium | **Predicted:** Easy | **Confidence:** 0.6422
   - Text: `AI & ML Preprocessing What is sentence embedding?...`

**Observations on Errors:**
The model seems to struggle with the nuances of AI & ML topics. Some concepts like "PCA whitening" (Hard) or "MaxAbsScaler" (Hard) are incorrectly predicted as Easy, perhaps because preprocessing is generally considered a fundamental topic, or because the text is very short. Formula-based questions ("Variance: Var(X) = ?") are actually Easy but predicted as Hard, maybe due to special characters.

### Next Improvements
- Include the answer choices (`option_a` through `d`) and `correct_answer` into the text features.
- Calculate text length or complexity metrics (e.g., presence of math symbols) as additional numerical features.
- Explore non-linear models (e.g., Random Forest or Gradient Boosting) since vocabulary alone might not capture difficulty linearly.
- Handle class imbalance during TF-IDF (some topics might dominate).

## Task 9/10: Job Match Random Forest Tuning Metrics

- **Run Timestamp**: 2026-07-07T06:16:24Z
- **Git Commit**: d6e5de3
- **Model**: RandomForestClassifier (tuned via RandomizedSearchCV, 15 candidates, cv=3)
- **Best Parameters**: `n_estimators: 50`, `min_samples_split: 10`, `max_depth: null`
- **CV ROC-AUC**: 0.8393

### Test Metrics
- **Accuracy**: 0.7733
- **Precision**: 0.7727
- **Recall**: 0.5862
- **F1 Score**: 0.6667
- **ROC-AUC**: 0.8523

### Thresholding & Gates
- **Optimal F1 Threshold**: 0.3597 (Achieves F1 of 0.75)
- **Gates**: All passed (ROC > 0.75, Acc > 0.65)
- **Artifact**: `models/rf_job_match_pipeline_tuned.pkl`
