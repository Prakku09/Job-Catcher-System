import pandas as pd
import numpy as np
import os
import sys
import json
import time
import subprocess
import joblib
import tracemalloc
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

sys.path.append('.')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, auc, precision_recall_curve, brier_score_loss,
    classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from src.train_model import JobMatchPreprocessor
from src.ml_utils import check_dependencies, validate_data

def measure_latency(model, X, n_repeats=50):
    start = time.perf_counter()
    for _ in range(n_repeats):
        model.predict(X)
    return (time.perf_counter() - start) / n_repeats / len(X) * 1000  # ms/sample

def get_repo_metadata(data_path, X_train, X_test):
    try:
        git_commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('utf-8').strip()
    except:
        git_commit = "unknown"
    return {
        "git_commit": git_commit,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "python_version": sys.version.split()[0],
        "sklearn_version": sklearn.__version__,
        "dataset_path": data_path,
        "random_seed": 42,
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    }

def main():
    print("Checking dependencies...")
    check_dependencies()
    
    # 1. Setup Directories
    for d in ['models', 'reports', 'plots', 'docs']:
        os.makedirs(d, exist_ok=True)
        
    # 2. Load Data
    data_path = 'src/data/clean_modelling_table.csv'
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    leakage_cols = ['exam_time_seconds', 'self_reported_confidence', 'retake_count', 
                    'application_id', 'student_id', 'job_id', 'name', 'application_date']
    df = df.drop(columns=leakage_cols, errors='ignore')
    
    y = df['is_good_match']
    X = df.drop(columns=['is_good_match'], errors='ignore')
    
    validate_data(X, y)
    
    # Train / Calib / Test Split (70 / 15 / 15)
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_calib, y_train, y_calib = train_test_split(X_train_full, y_train_full, test_size=0.1764, random_state=42, stratify=y_train_full)
    
    numeric_features = ['diff_experience', 'diff_python', 'diff_ml', 'diff_sql', 'diff_stats', 'diff_js', 'diff_ds']
    
    preprocessor = Pipeline([
        ('feature_engineering', JobMatchPreprocessor()),
        ('column_transfer', ColumnTransformer([('scaler', StandardScaler(), numeric_features)], remainder='drop'))
    ])
    
    print("Fitting preprocessor and base model...")
    X_train_trans = preprocessor.fit_transform(X_train)
    X_calib_trans = preprocessor.transform(X_calib)
    X_test_trans = preprocessor.transform(X_test)
    
    # Using the best base model architecture from Task 11
    base_model = LogisticRegression(class_weight='balanced', random_state=42)
    base_model.fit(X_train_trans, y_train)
    
    # 3. Probability Calibration
    print("Calibrating probabilities...")
    calibrated_sigmoid = CalibratedClassifierCV(estimator=base_model, method='sigmoid', cv=3)
    calibrated_sigmoid.fit(X_train_trans, y_train)
    
    calibrated_isotonic = CalibratedClassifierCV(estimator=base_model, method='isotonic', cv=3)
    calibrated_isotonic.fit(X_train_trans, y_train)
    
    prob_sig = calibrated_sigmoid.predict_proba(X_test_trans)[:, 1]
    prob_iso = calibrated_isotonic.predict_proba(X_test_trans)[:, 1]
    prob_uncal = base_model.predict_proba(X_test_trans)[:, 1]
    
    brier_sig = brier_score_loss(y_test, prob_sig)
    brier_iso = brier_score_loss(y_test, prob_iso)
    brier_uncal = brier_score_loss(y_test, prob_uncal)
    
    best_method = "sigmoid" if brier_sig < brier_iso else "isotonic"
    best_calibrated_model = calibrated_sigmoid if best_method == "sigmoid" else calibrated_isotonic
    best_prob = prob_sig if best_method == "sigmoid" else prob_iso
    
    calibration_metrics = {
        "brier_score_uncalibrated": float(brier_uncal),
        "brier_score_sigmoid": float(brier_sig),
        "brier_score_isotonic": float(brier_iso),
        "selected_method": best_method
    }
    
    # Plot Calibration Curve
    plt.figure(figsize=(8, 6))
    fop_uncal, mpv_uncal = calibration_curve(y_test, prob_uncal, n_bins=10)
    fop_best, mpv_best = calibration_curve(y_test, best_prob, n_bins=10)
    
    plt.plot(mpv_uncal, fop_uncal, marker='o', label='Uncalibrated')
    plt.plot(mpv_best, fop_best, marker='s', label=f'Calibrated ({best_method})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Value')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.savefig('plots/calibration_curve.png', bbox_inches='tight')
    plt.close()
    
    # 4. Threshold Optimization
    print("Determining optimal decision threshold...")
    # Evaluate thresholds on calibration set to avoid test set leakage
    calib_probs = best_calibrated_model.predict_proba(X_calib_trans)[:, 1]
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1 = 0
    best_thresh = 0.5
    for t in thresholds:
        preds = (calib_probs >= t).astype(int)
        f1 = f1_score(y_calib, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            
    # Calculate costs (Hypothetical: False Positives cost $500, False Negatives cost $2000)
    test_preds = (best_prob >= best_thresh).astype(int)
    cm = confusion_matrix(y_test, test_preds)
    fp = cm[0, 1]
    fn = cm[1, 0]
    expected_cost = fp * 500 + fn * 2000
    
    operating_point = {
        "optimal_threshold": float(best_thresh),
        "justification": "Maximized F1-Score on the calibration set. A cost-sensitive analysis implies higher penalty for missing a good candidate (FN).",
        "expected_error": {
            "false_positives": int(fp),
            "false_negatives": int(fn)
        },
        "expected_cost_usd": float(expected_cost)
    }
    
    # 5. Core Evaluation
    precision, recall, _ = precision_recall_curve(y_test, best_prob)
    binary_metrics = {
        "accuracy": float(accuracy_score(y_test, test_preds)),
        "precision": float(precision_score(y_test, test_preds, zero_division=0)),
        "recall": float(recall_score(y_test, test_preds, zero_division=0)),
        "f1_score": float(f1_score(y_test, test_preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, best_prob)),
        "pr_auc": float(auc(recall, precision)),
        "brier_score": float(calibration_metrics[f"brier_score_{best_method}"])
    }
    
    class_report = classification_report(y_test, test_preds, output_dict=True)
    
    # Plots
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Production Confusion Matrix')
    plt.savefig('plots/production_confusion_matrix.png', bbox_inches='tight')
    plt.close()
    
    fpr, tpr, _ = roc_curve(y_test, best_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC AUC ({binary_metrics["roc_auc"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('Production ROC Curve')
    plt.legend()
    plt.savefig('plots/production_roc_curve.png', bbox_inches='tight')
    plt.close()
    
    plt.figure()
    plt.plot(recall, precision, label=f'PR AUC ({binary_metrics["pr_auc"]:.3f})')
    plt.title('Production Precision-Recall Curve')
    plt.legend()
    plt.savefig('plots/production_pr_curve.png', bbox_inches='tight')
    plt.close()
    
    # 6. Segment-wise Evaluation
    print("Performing segment analysis...")
    X_test_raw = X_test.copy()
    X_test_raw['is_good_match'] = y_test
    X_test_raw['pred'] = test_preds
    
    segment_analysis = {}
    if 'years_experience' in X_test_raw.columns:
        median_exp = X_test_raw['years_experience'].median()
        freshers = X_test_raw[X_test_raw['years_experience'] <= median_exp]
        experienced = X_test_raw[X_test_raw['years_experience'] > median_exp]
        
        segment_analysis['experience'] = {
            "freshers": {
                "accuracy": float(accuracy_score(freshers['is_good_match'], freshers['pred'])),
                "f1_score": float(f1_score(freshers['is_good_match'], freshers['pred'], zero_division=0))
            },
            "experienced": {
                "accuracy": float(accuracy_score(experienced['is_good_match'], experienced['pred'])),
                "f1_score": float(f1_score(experienced['is_good_match'], experienced['pred'], zero_division=0))
            }
        }
        
        # Plot segment comparison
        plt.figure(figsize=(8, 5))
        sns.barplot(x=['Freshers', 'Experienced'], y=[segment_analysis['experience']['freshers']['f1_score'], segment_analysis['experience']['experienced']['f1_score']])
        plt.ylabel('F1 Score')
        plt.title('Model Performance Across Experience Segments')
        plt.ylim(0, 1)
        plt.savefig('plots/segment_comparison.png', bbox_inches='tight')
        plt.close()
        
    # 7. Live Verification
    live_samples = X_test_raw.sample(min(10, len(X_test_raw)), random_state=42)
    live_out = live_samples.copy()
    # Recalculate prob to be precise
    live_probs = best_calibrated_model.predict_proba(preprocessor.transform(live_samples.drop(columns=['is_good_match', 'pred'])))[:, 1]
    live_out['probability'] = live_probs
    live_out['threshold'] = best_thresh
    live_out['final_prediction'] = (live_probs >= best_thresh).astype(int)
    live_out.to_csv('reports/live_predictions.csv', index=False)
    
    # 8. Performance Trade-off
    print("Profiling performance...")
    tracemalloc.start()
    latency_ms = measure_latency(best_calibrated_model, X_test_trans)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', best_calibrated_model)])
    joblib.dump(pipeline, 'models/production_binary_classifier.pkl')
    model_size_mb = os.path.getsize('models/production_binary_classifier.pkl') / (1024 * 1024)
    
    tradeoff_df = pd.DataFrame([{
        "latency_ms_per_sample": latency_ms,
        "model_size_mb": model_size_mb,
        "peak_memory_mb": peak / (1024 * 1024),
        "f1_score": binary_metrics["f1_score"]
    }])
    tradeoff_df.to_csv('reports/performance_tradeoff.csv', index=False)
    
    # Latency Plot
    plt.figure(figsize=(6, 4))
    sns.barplot(x=['Inference Latency'], y=[latency_ms])
    plt.ylabel('Milliseconds (ms)')
    plt.title('Latency Breakdown per Sample')
    plt.savefig('plots/latency_breakdown.png', bbox_inches='tight')
    plt.close()
    
    # Dependencies
    dependencies = {
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
        "joblib": joblib.__version__
    }
    
    # Merge Metadata & Save
    meta = get_repo_metadata(data_path, X_train_full, X_test)
    
    for report_name, data_obj in [
        ("calibration_metrics.json", calibration_metrics),
        ("operating_point.json", operating_point),
        ("binary_metrics.json", binary_metrics),
        ("segment_analysis.json", segment_analysis),
        ("classification_report.json", class_report),
        ("dependencies.json", dependencies)
    ]:
        data_obj["metadata"] = meta
        with open(f"reports/{report_name}", "w") as f:
            json.dump(data_obj, f, indent=4)
            
    print("Production pipeline completed. All artifacts saved successfully.")

if __name__ == "__main__":
    main()
