import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime
import time
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sys.path.append('.')

import optuna
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, auc, precision_recall_curve, roc_curve
from optuna.integration import XGBoostPruningCallback
import joblib

from src.train_model import JobMatchPreprocessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_metrics(y_true, y_pred, y_prob):
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(auc(recall_curve, precision_curve))
    }

def main():
    logging.info("Starting Task 17: Bayesian Hyperparameter Optimization")
    
    # Setup Data
    data_path = 'src/data/clean_modelling_table.csv'
    df = pd.read_csv(data_path)
    
    leakage_cols = ['exam_time_seconds', 'self_reported_confidence', 'retake_count', 
                    'application_id', 'student_id', 'job_id', 'name', 'application_date']
    df = df.drop(columns=leakage_cols, errors='ignore')
    
    y = df['is_good_match']
    X = df.drop(columns=['is_good_match'], errors='ignore')
    
    numeric_features = ['diff_experience', 'diff_python', 'diff_ml', 'diff_sql', 'diff_stats', 'diff_js', 'diff_ds']
    
    preprocessor = Pipeline([
        ('feature_engineering', JobMatchPreprocessor()),
        ('column_transfer', ColumnTransformer([('scaler', StandardScaler(), numeric_features)], remainder='drop'))
    ])
    
    # Train/Test Split (keeping 15% out for final evaluation)
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    
    logging.info("Applying preprocessing to training data for optimization...")
    X_train_trans = preprocessor.fit_transform(X_train_full)
    X_test_trans = preprocessor.transform(X_test)
    
    if isinstance(X_train_trans, np.ndarray):
        X_train_trans = pd.DataFrame(X_train_trans, columns=numeric_features)
        X_test_trans = pd.DataFrame(X_test_trans, columns=numeric_features)
        
    os.makedirs('plots', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Evaluate Baseline Model
    logging.info("Evaluating Baseline XGBoost Model...")
    baseline_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    
    start_time = time.perf_counter()
    baseline_model.fit(X_train_trans, y_train_full)
    baseline_train_time = time.perf_counter() - start_time
    
    start_time = time.perf_counter()
    base_preds = baseline_model.predict(X_test_trans)
    base_probs = baseline_model.predict_proba(X_test_trans)[:, 1]
    baseline_infer_time = time.perf_counter() - start_time
    
    baseline_metrics = calculate_metrics(y_test, base_preds, base_probs)
    
    trial_history = []
    
    # Optuna Objective Function
    def objective(trial):
        param = {
            "verbosity": 0,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
            "random_state": 42
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        
        start_t = time.perf_counter()
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train_trans, y_train_full)):
            X_tr, y_tr = X_train_trans.iloc[train_idx], y_train_full.iloc[train_idx]
            X_va, y_va = X_train_trans.iloc[val_idx], y_train_full.iloc[val_idx]
            
            dtrain = xgb.DMatrix(X_tr, label=y_tr)
            dval = xgb.DMatrix(X_va, label=y_va)
            
            pruning_callback = XGBoostPruningCallback(trial, "validation-auc")
            
            evals = [(dval, "validation")]
            try:
                bst = xgb.train(
                    param, 
                    dtrain, 
                    num_boost_round=param['n_estimators'], 
                    evals=evals, 
                    callbacks=[pruning_callback],
                    early_stopping_rounds=20,
                    verbose_eval=False
                )
                
                preds_prob = bst.predict(dval)
                score = roc_auc_score(y_va, preds_prob)
                cv_scores.append(score)
            except optuna.TrialPruned:
                # Log pruned
                trial_history.append({
                    "trial": trial.number,
                    "status": "Pruned",
                    "score": None,
                    "time_sec": time.perf_counter() - start_t,
                    "params": json.dumps(trial.params)
                })
                raise
                
        mean_score = np.mean(cv_scores)
        
        trial_history.append({
            "trial": trial.number,
            "status": "Completed",
            "score": mean_score,
            "time_sec": time.perf_counter() - start_t,
            "params": json.dumps(trial.params)
        })
        
        return mean_score

    # Run Optuna
    logging.info("Starting Optuna Study...")
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
    study.optimize(objective, n_trials=50)
    
    logging.info(f"Number of finished trials: {len(study.trials)}")
    logging.info(f"Best trial: {study.best_trial.value}")
    
    # Save best parameters
    best_params = study.best_trial.params
    with open('reports/best_parameters.json', 'w') as f:
        json.dump(best_params, f, indent=4)
        
    pd.DataFrame(trial_history).to_csv('reports/trial_history.csv', index=False)
    
    # Retrain on full train set with best params
    logging.info("Retraining best model on full training set...")
    best_xgb_model = xgb.XGBClassifier(**best_params, random_state=42, eval_metric='logloss')
    
    start_time = time.perf_counter()
    best_xgb_model.fit(X_train_trans, y_train_full)
    tuned_train_time = time.perf_counter() - start_time
    
    start_time = time.perf_counter()
    tuned_preds = best_xgb_model.predict(X_test_trans)
    tuned_probs = best_xgb_model.predict_proba(X_test_trans)[:, 1]
    tuned_infer_time = time.perf_counter() - start_time
    
    tuned_metrics = calculate_metrics(y_test, tuned_preds, tuned_probs)
    
    # Compare Baseline vs Tuned
    comparison_data = [
        {"Model": "Baseline XGBoost", **baseline_metrics, "Training_Time_s": baseline_train_time, "Inference_Time_s": baseline_infer_time},
        {"Model": "Optuna Tuned XGBoost", **tuned_metrics, "Training_Time_s": tuned_train_time, "Inference_Time_s": tuned_infer_time}
    ]
    pd.DataFrame(comparison_data).to_csv('reports/baseline_vs_tuned.csv', index=False)
    
    test_results = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "baseline_metrics": baseline_metrics,
        "tuned_metrics": tuned_metrics,
        "improvement_roc_auc": tuned_metrics['roc_auc'] - baseline_metrics['roc_auc']
    }
    with open('reports/test_results.json', 'w') as f:
        json.dump(test_results, f, indent=4)
        
    # Save Pipeline
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', best_xgb_model)
    ])
    joblib.dump(full_pipeline, 'models/best_tuned_model.pkl')
    logging.info("Saved tuned pipeline to models/best_tuned_model.pkl")
    
    # Generate Plots
    logging.info("Generating plots...")
    
    # 1. Optuna built-in plots
    try:
        from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances, plot_parallel_coordinate
        
        fig = plot_optimization_history(study)
        plt.savefig('plots/optimization_history.png', bbox_inches='tight')
        plt.close()
        
        fig = plot_param_importances(study)
        plt.savefig('plots/parameter_importance.png', bbox_inches='tight')
        plt.close()
        
        fig = plot_parallel_coordinate(study)
        plt.savefig('plots/parallel_coordinates.png', bbox_inches='tight')
        plt.close()
    except Exception as e:
        logging.warning(f"Could not generate Optuna matplotlib plots: {e}")
        
    # 2. ROC Curve
    plt.figure(figsize=(8, 6))
    fpr_base, tpr_base, _ = roc_curve(y_test, base_probs)
    fpr_tuned, tpr_tuned, _ = roc_curve(y_test, tuned_probs)
    plt.plot(fpr_base, tpr_base, label=f"Baseline (AUC = {baseline_metrics['roc_auc']:.3f})")
    plt.plot(fpr_tuned, tpr_tuned, label=f"Tuned (AUC = {tuned_metrics['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison (Test Set)')
    plt.legend()
    plt.savefig('plots/roc_curve.png')
    plt.close()
    
    # 3. PR Curve
    plt.figure(figsize=(8, 6))
    prec_base, rec_base, _ = precision_recall_curve(y_test, base_probs)
    prec_tuned, rec_tuned, _ = precision_recall_curve(y_test, tuned_probs)
    plt.plot(rec_base, prec_base, label=f"Baseline (AUC = {baseline_metrics['pr_auc']:.3f})")
    plt.plot(rec_tuned, prec_tuned, label=f"Tuned (AUC = {tuned_metrics['pr_auc']:.3f})")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve Comparison (Test Set)')
    plt.legend()
    plt.savefig('plots/pr_curve.png')
    plt.close()
    
    # 4. Model Comparison Bar Chart
    plt.figure(figsize=(10, 6))
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    
    base_vals = [baseline_metrics[m] for m in metrics_to_plot]
    tuned_vals = [tuned_metrics[m] for m in metrics_to_plot]
    
    plt.bar(x - width/2, base_vals, width, label='Baseline')
    plt.bar(x + width/2, tuned_vals, width, label='Tuned')
    
    plt.ylabel('Scores')
    plt.title('Baseline vs Tuned Model Metrics (Test Set)')
    plt.xticks(x, [m.upper() for m in metrics_to_plot])
    plt.legend()
    plt.ylim(0.5, 1.0)
    plt.savefig('plots/model_comparison.png')
    plt.close()
    
    logging.info("All outputs generated successfully!")

if __name__ == '__main__':
    main()
