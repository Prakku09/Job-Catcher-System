import pandas as pd
import numpy as np
import os
import joblib
import sys

sys.path.append('.')

from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, confusion_matrix, f1_score
try:
    from sklearn.frozen import FrozenEstimator # Scikit-learn >= 1.6
except ImportError:
    # Polyfill for earlier sklearn versions
    from sklearn.base import clone
    class FrozenEstimator:
        def __init__(self, estimator):
            self.estimator = clone(estimator)
        def fit(self, X, y=None):
            return self
        def predict(self, X):
            return self.estimator.predict(X)
        def predict_proba(self, X):
            return self.estimator.predict_proba(X)

from src.train_model import JobMatchPreprocessor

# --- Business Logic Constraints ---
COST_FALSE_NEGATIVE = 5.0  # Missing a good candidate is highly costly
COST_FALSE_POSITIVE = 1.0  # Interviewing a bad candidate is moderately costly

def main():
    print("Loading data for productionization...")
    df = pd.read_csv('src/data/clean_modelling_table.csv')
    
    leakage_cols = ['exam_time_seconds', 'self_reported_confidence', 'retake_count', 
                    'application_id', 'student_id', 'job_id', 'name', 'application_date']
    df = df.drop(columns=leakage_cols, errors='ignore')
    
    # We will need segments for the fairness check. Keep original data aligned.
    y = df['is_good_match']
    X = df.drop(columns=['is_good_match'], errors='ignore')
    segments = df['education_level'] if 'education_level' in df.columns else pd.Series(['Unknown']*len(df))
    
    # We split into Train and Test. Test will be our proxy for "Calibration / Holdout"
    X_train, X_cal, y_train, y_cal, seg_train, seg_cal = train_test_split(
        X, y, segments, test_size=0.2, random_state=42, stratify=y
    )
    
    print("\nLoading the best ensemble pipeline...")
    # Loading the Voting or Stacking pipeline from Task 11. Stacking is robust.
    # But CalibratedClassifierCV needs a fitted estimator or one it can fit. 
    raw_model = joblib.load('models/stacking_ensemble_pipeline.pkl')
    
    # 1. Calibration Method Selection (Task 12.1)
    print("\n--- Calibration Selection ---")
    # Split calibration data further so method choice doesn't touch the test set
    X_cal_fit, X_cal_val, y_cal_fit, y_cal_val = train_test_split(X_cal, y_cal, test_size=0.4, stratify=y_cal, random_state=42)
    
    candidate_scores = {}
    for method in ("isotonic", "sigmoid"):
        try:
            cand = CalibratedClassifierCV(estimator=FrozenEstimator(raw_model), method=method)
            cand.fit(X_cal_fit, y_cal_fit)
            brier = brier_score_loss(y_cal_val, cand.predict_proba(X_cal_val)[:, 1])
            candidate_scores[method] = brier
            print(f"  Method {method}: Brier Score = {brier:.4f}")
        except Exception as e:
            print(f"  Method {method} failed: {e}")
            candidate_scores[method] = float('inf')
    
    chosen_method = min(candidate_scores, key=candidate_scores.get)
    print(f"Chosen calibration method: {chosen_method}")
    
    # Refit on the full calibration split once chosen
    calibrated_model = CalibratedClassifierCV(estimator=FrozenEstimator(raw_model), method=chosen_method)
    calibrated_model.fit(X_cal, y_cal)
    
    # 2. Cost-based Thresholding (Task 12.2)
    print("\n--- Cost-Based Threshold Sweeping ---")
    # To avoid data leakage, we evaluate costs on the holdout/calibration set
    cal_proba = calibrated_model.predict_proba(X_cal)[:, 1]
    
    costs = []
    thresholds = np.linspace(0.01, 0.99, 99)
    for t in thresholds:
        pred = (cal_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_cal, pred, labels=[0,1]).ravel()
        costs.append(fp * COST_FALSE_POSITIVE + fn * COST_FALSE_NEGATIVE)
        
    best_threshold = thresholds[np.argmin(costs)]
    print(f"Optimal Threshold (minimizing FN={COST_FALSE_NEGATIVE}, FP={COST_FALSE_POSITIVE}): {best_threshold:.3f}")
    
    # 3. Segment Fairness Check (Task 12.3)
    print("\n--- Segment Fairness Check ---")
    final_pred = (cal_proba >= best_threshold).astype(int)
    
    f1_scores = []
    for segment_name in seg_cal.unique():
        mask = (seg_cal == segment_name)
        if mask.sum() == 0: continue
        y_seg, pred_seg = y_cal[mask], final_pred[mask]
        
        # calculate F1, handle edge cases with 0 denominator
        if sum(y_seg) == 0 and sum(pred_seg) == 0:
            score = 1.0
        else:
            score = f1_score(y_seg, pred_seg, zero_division=0)
            
        f1_scores.append(score)
        print(f"  Segment [{segment_name}]: F1 Score = {score:.4f} (n={mask.sum()})")
        
    fairness_gap = max(f1_scores) - min(f1_scores)
    print(f"Fairness Gap (Max F1 - Min F1): {fairness_gap:.4f}")
    
    if fairness_gap > 0.2:
        print("WARNING: High fairness gap detected across education levels. The model may be biased.")
    else:
        print("Fairness gap is within acceptable limits.")
        
    # 4. Packaging (Task 12.4)
    print("\n--- Packaging Model for Serving ---")
    pkg = {
        "model": calibrated_model,
        "threshold": best_threshold,
        "cost_assumptions": {"false_negative": COST_FALSE_NEGATIVE, "false_positive": COST_FALSE_POSITIVE},
    }
    
    pkg_path = "models/production_model_package.joblib"
    joblib.dump(pkg, pkg_path)
    print(f"Packaged model cleanly saved to {pkg_path}")

if __name__ == '__main__':
    main()
