import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sys.path.append('.')

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, auc, precision_recall_curve, roc_curve

# Import custom preprocessor
from src.train_model import JobMatchPreprocessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_metrics(y_true, y_pred, y_prob):
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": auc(recall_curve, precision_curve)
    }

def main():
    logging.info("Starting Task 16: Cross-Validation & Generalization Evaluation")
    
    # 1. Setup Data
    data_path = 'src/data/clean_modelling_table.csv'
    df = pd.read_csv(data_path)
    
    leakage_cols = ['exam_time_seconds', 'self_reported_confidence', 'retake_count', 
                    'application_id', 'student_id', 'job_id', 'name', 'application_date']
    df = df.drop(columns=leakage_cols, errors='ignore')
    
    y = df['is_good_match']
    X = df.drop(columns=['is_good_match'], errors='ignore')
    
    numeric_features = ['diff_experience', 'diff_python', 'diff_ml', 'diff_sql', 'diff_stats', 'diff_js', 'diff_ds']
    
    # Base preprocessing steps
    preprocessor = Pipeline([
        ('feature_engineering', JobMatchPreprocessor()),
        ('column_transfer', ColumnTransformer([('scaler', StandardScaler(), numeric_features)], remainder='drop'))
    ])
    
    # Pre-transform the data to save time across folds since the transformation doesn't leak target info
    logging.info("Applying preprocessing to data...")
    X_trans = preprocessor.fit_transform(X)
    # Ensure X_trans is a DataFrame or Numpy array
    if isinstance(X_trans, np.ndarray):
        X_trans = pd.DataFrame(X_trans, columns=numeric_features)
    
    # 2. Define Models
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
        "SVM": SVC(probability=True, random_state=42, class_weight='balanced'),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42)
    }
    
    fold_configs = [5, 10]
    
    all_fold_scores = []
    
    # Ensure directories exist
    os.makedirs('plots', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    # 3. Standard Cross-Validation
    for k in fold_configs:
        logging.info(f"Running {k}-Fold Cross-Validation...")
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        
        for model_name, model in models.items():
            logging.info(f"  Evaluating {model_name} ({k}-fold)")
            
            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_trans, y), 1):
                X_train, y_train = X_trans.iloc[train_idx], y.iloc[train_idx]
                X_val, y_val = X_trans.iloc[val_idx], y.iloc[val_idx]
                
                # Check for stratification
                assert len(np.unique(y_train)) == 2, "Train fold missing a class"
                assert len(np.unique(y_val)) == 2, "Val fold missing a class"
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                y_prob = model.predict_proba(X_val)[:, 1]
                
                metrics = calculate_metrics(y_val, y_pred, y_prob)
                metrics.update({
                    "Model": model_name,
                    "CV_Folds": k,
                    "Fold_ID": fold_idx
                })
                all_fold_scores.append(metrics)
                
    fold_scores_df = pd.DataFrame(all_fold_scores)
    fold_scores_df.to_csv('fold_scores.csv', index=False)
    logging.info("Saved fold_scores.csv")
    
    # Aggregate results
    cv_results_list = []
    for (model_name, k), group in fold_scores_df.groupby(['Model', 'CV_Folds']):
        stats = {
            "Model": model_name,
            "CV_Folds": k,
            "Mean_Accuracy": group['accuracy'].mean(),
            "Std_Accuracy": group['accuracy'].std(),
            "Mean_ROC_AUC": group['roc_auc'].mean(),
            "Std_ROC_AUC": group['roc_auc'].std(),
            "Var_ROC_AUC": group['roc_auc'].var(),
            "Best_Fold_ROC_AUC": group['roc_auc'].max(),
            "Worst_Fold_ROC_AUC": group['roc_auc'].min(),
            "Mean_F1": group['f1'].mean(),
            "Mean_Precision": group['precision'].mean(),
            "Mean_Recall": group['recall'].mean(),
            "Mean_PR_AUC": group['pr_auc'].mean()
        }
        cv_results_list.append(stats)
        
    cv_results_df = pd.DataFrame(cv_results_list)
    cv_results_df.to_csv('cross_validation_results.csv', index=False)
    logging.info("Saved cross_validation_results.csv")
    
    # Best Model Selection
    # Choose 10-fold results for final comparison as it's generally more stable
    comp_df = cv_results_df[cv_results_df['CV_Folds'] == 10].copy()
    comp_df = comp_df.sort_values(by=['Mean_ROC_AUC', 'Var_ROC_AUC'], ascending=[False, True])
    comp_df.to_csv('model_comparison.csv', index=False)
    
    best_model_row = comp_df.iloc[0]
    best_model_name = best_model_row['Model']
    
    # 4. Nested CV for Random Forest
    logging.info("Running Nested CV for Random Forest...")
    nested_scores = []
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    rf = RandomForestClassifier(random_state=42)
    param_dist = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5]
    }
    
    for outer_fold, (train_idx, val_idx) in enumerate(outer_cv.split(X_trans, y), 1):
        X_train_out, y_train_out = X_trans.iloc[train_idx], y.iloc[train_idx]
        X_val_out, y_val_out = X_trans.iloc[val_idx], y.iloc[val_idx]
        
        search = RandomizedSearchCV(rf, param_dist, cv=inner_cv, scoring='roc_auc', n_iter=5, random_state=42, n_jobs=-1)
        search.fit(X_train_out, y_train_out)
        
        best_rf = search.best_estimator_
        y_pred = best_rf.predict(X_val_out)
        y_prob = best_rf.predict_proba(X_val_out)[:, 1]
        
        metrics = calculate_metrics(y_val_out, y_pred, y_prob)
        metrics.update({
            "Outer_Fold": outer_fold,
            "Best_Params": json.dumps(search.best_params_)
        })
        nested_scores.append(metrics)
        
    nested_df = pd.DataFrame(nested_scores)
    nested_df.to_csv('nested_cv_results.csv', index=False)
    logging.info("Saved nested_cv_results.csv")
    
    # 5. Generalization Report
    gen_report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "dataset_samples": len(df),
        "validation_strategy": "StratifiedKFold (shuffle=True, random_state=42)",
        "models_evaluated": list(models.keys()),
        "best_generalization_model": best_model_name,
        "best_model_metrics_10fold": {
            "mean_roc_auc": float(best_model_row['Mean_ROC_AUC']),
            "var_roc_auc": float(best_model_row['Var_ROC_AUC']),
            "mean_accuracy": float(best_model_row['Mean_Accuracy']),
            "best_fold_roc_auc": float(best_model_row['Best_Fold_ROC_AUC']),
            "worst_fold_roc_auc": float(best_model_row['Worst_Fold_ROC_AUC'])
        },
        "nested_cv_rf_mean_roc_auc": float(nested_df['roc_auc'].mean()),
        "conclusion": f"{best_model_name} demonstrated the best generalization based on high mean ROC-AUC and low variance across 10 folds."
    }
    
    with open('generalization_report.json', 'w') as f:
        json.dump(gen_report, f, indent=4)
        
    logging.info("Saved generalization_report.json")
    
    # 6. Visualizations
    logging.info("Generating plots...")
    
    # 1. CV Score Comparison (Bar Plot)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=cv_results_df[cv_results_df['CV_Folds']==10], x='Model', y='Mean_ROC_AUC')
    plt.title('10-Fold CV Mean ROC-AUC by Model')
    plt.ylim(0.5, 1.0)
    plt.savefig('plots/cv_score_comparison.png')
    plt.close()
    
    # 2. Fold-wise Performance Plots
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=fold_scores_df[fold_scores_df['CV_Folds']==10], x='Fold_ID', y='roc_auc', hue='Model', marker='o')
    plt.title('Fold-wise ROC-AUC (10-Fold)')
    plt.xticks(range(1, 11))
    plt.savefig('plots/fold_wise_performance.png')
    plt.close()
    
    # 3. Boxplot of Model Scores
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=fold_scores_df[fold_scores_df['CV_Folds']==10], x='Model', y='roc_auc')
    plt.title('Distribution of ROC-AUC Scores (10-Fold)')
    plt.savefig('plots/boxplot_model_scores.png')
    plt.close()
    
    # 4. Mean +/- Std Dev Comparison
    plt.figure(figsize=(10, 6))
    df_10f = cv_results_df[cv_results_df['CV_Folds']==10]
    plt.errorbar(df_10f['Model'], df_10f['Mean_ROC_AUC'], yerr=df_10f['Std_ROC_AUC'], fmt='o', capsize=5)
    plt.title('Mean ROC-AUC ± Standard Deviation')
    plt.savefig('plots/mean_std_comparison.png')
    plt.close()
    
    # 5. ROC-AUC comparison chart (simulated curves for top model on last fold for simplicity)
    # Just grab the last fold out of the loop for plotting ROC curves for all models
    plt.figure(figsize=(8, 6))
    for model_name, model in models.items():
        y_prob = model.predict_proba(X_val)[:, 1]
        fpr, tpr, _ = roc_curve(y_val, y_prob)
        auc_val = roc_auc_score(y_val, y_prob)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_val:.3f})")
        
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (Last Fold of 10-Fold CV)')
    plt.legend()
    plt.savefig('plots/roc_auc_comparison_chart.png')
    plt.close()
    
    logging.info("All outputs generated successfully!")

if __name__ == '__main__':
    main()
