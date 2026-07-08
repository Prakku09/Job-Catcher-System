# Model Card: Production Binary Classifier

## 1. Model Details
- **Model Name:** `production_binary_classifier.pkl`
- **Model Type:** Logistic Regression with `CalibratedClassifierCV`
- **Task:** Binary Classification (Predicting if a candidate is a good job match)
- **Framework:** Scikit-Learn

## 2. Intended Use
- **Primary Use Case:** Filtering incoming job applications by matching their skill sets against job requirements to predict `is_good_match`.
- **Out-of-Scope Use Cases:** Not intended for automated hiring without human review. Not intended for evaluating candidate personality or cultural fit.

## 3. Training Data
- **Source:** Synthetic `clean_modelling_table.csv` derived from candidate profiles and job descriptions.
- **Preprocessing:** Engineered difference features (e.g., `diff_python`, `diff_experience`) calculated via `JobMatchPreprocessor`. Null values assume strict API validation prior to ingest.

## 4. Evaluation Metrics
The model was evaluated using a strict 70-15-15 split (Train, Calib, Test). 
Key metrics tracked during productionization:
- **Accuracy, Precision, Recall, F1-Score**
- **ROC-AUC & PR-AUC**
- **Brier Score** (for probability calibration effectiveness)
*See `reports/binary_metrics.json` for live calculated values.*

## 5. Calibration & Operating Point
- **Calibration Method:** Automatically selects between Platt Scaling (Sigmoid) and Isotonic Regression based on minimizing the Brier Score.
- **Decision Threshold:** Shifted away from default `0.5` based on cost-sensitive analysis (e.g., penalizing False Negatives heavier than False Positives). See `reports/operating_point.json`.

## 6. Limitations & Bias
- **Segment Bias:** Evaluated against segments (e.g., Freshers vs. Experienced). The model shows varying degrees of F1 performance across these segments. See `reports/segment_analysis.json`.
- **Generalization:** Heavily dependent on the continuous quality of the input data API. It assumes the distribution of test data reflects reality.
