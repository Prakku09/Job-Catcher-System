import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, precision_recall_curve, f1_score, auc
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import joblib
import os

class JobMatchPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X_out = pd.DataFrame(index=X.index)
        
        # 1. Domain Feature
        def get_domain_score(row):
            title = str(row.get('title', '')).lower()
            if 'data' in title or 'ml' in title or 'ai' in title or 'machine learning' in title or 'analyst' in title:
                return max(row.get('python_score', 0), row.get('sql_score', 0), row.get('ml_score', 0), row.get('statistics_score', 0))
            else:
                return max(row.get('javascript_score', 0), row.get('data_structures_score', 0), row.get('python_score', 0))
                
        X_filled = X.copy()
        for col in ['python_score', 'sql_score', 'ml_score', 'statistics_score', 'javascript_score', 'data_structures_score']:
            if col in X_filled:
                X_filled[col] = X_filled[col].fillna(0)
            else:
                X_filled[col] = 0
                
        for col in ['python_required', 'sql_required', 'ml_required', 'statistics_required', 'javascript_required', 'data_structures_required']:
            if col in X_filled:
                X_filled[col] = X_filled[col].fillna(0)
            else:
                X_filled[col] = 0
        
        X_out['domain_match'] = X_filled.apply(get_domain_score, axis=1)
        
        # 2. Diff Experience
        exp_student = X['years_experience'].fillna(0) if 'years_experience' in X else pd.Series(0, index=X.index)
        exp_job = X['exp_required_years'].fillna(0) if 'exp_required_years' in X else pd.Series(0, index=X.index)
        X_out['diff_experience'] = exp_student - exp_job
        
        # 3. Diff Skills
        X_out['diff_python'] = X_filled['python_score'] - X_filled['python_required']
        X_out['diff_ml'] = X_filled['ml_score'] - X_filled['ml_required']
        X_out['diff_sql'] = X_filled['sql_score'] - X_filled['sql_required']
        X_out['diff_stats'] = X_filled['statistics_score'] - X_filled['statistics_required']
        X_out['diff_js'] = X_filled['javascript_score'] - X_filled['javascript_required']
        X_out['diff_ds'] = X_filled['data_structures_score'] - X_filled['data_structures_required']
        
        # 5. Salary diff
        salary_exp = X['expected_salary_inr'].fillna(0) if 'expected_salary_inr' in X else pd.Series(0, index=X.index)
        salary_off = X['salary_offered_inr'].fillna(0) if 'salary_offered_inr' in X else pd.Series(0, index=X.index)
        X_out['diff_salary'] = salary_off - salary_exp
        
        # 6. Education diff
        edu_map = {'UG': 1, 'PG': 2, 'PhD': 3}
        stu_edu = X['education_level'].map(edu_map).fillna(1) if 'education_level' in X else pd.Series(1, index=X.index)
        job_edu = X['edu_minimum'].map(edu_map).fillna(1) if 'edu_minimum' in X else pd.Series(1, index=X.index)
        X_out['diff_education'] = stu_edu - job_edu
        
        # 7. Location match
        loc_stu = X['location_student'].fillna('') if 'location_student' in X else pd.Series('', index=X.index)
        loc_job = X['location_job'].fillna('') if 'location_job' in X else pd.Series('', index=X.index)
        X_out['location_match'] = (loc_stu == loc_job).astype(int)
        
        return X_out[['diff_experience', 'diff_python', 'diff_ml', 'diff_sql', 'diff_stats', 'diff_js', 'diff_ds']]

def main():
    # 1. Load Data
    data_path = 'src/data/clean_modelling_table.csv'
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # 2. Remove Leakage Features
    leakage_cols = ['exam_time_seconds', 'self_reported_confidence', 'retake_count', 
                    'application_id', 'student_id', 'job_id', 'name', 'application_date']
    df = df.drop(columns=leakage_cols, errors='ignore')
    
    y = df['is_good_match']
    X = df.drop(columns=['is_good_match'], errors='ignore')
    
    # 3. Train/Test Split
    # We will use X_train_full for Cross-Validation which splits it into train/val folds internally
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    
    print(f"\nData splits - Train/Val (CV): {len(X_train_full)}, Test: {len(X_test)}")
    
    # 4. Build Full Pipeline
    print("\nBuilding Final Job Match RF Pipeline...")
    numeric_features = ['diff_experience', 'diff_python', 'diff_ml', 'diff_sql', 'diff_stats', 'diff_js', 'diff_ds']
    
    col_transformer = ColumnTransformer(
        transformers=[('scaler', StandardScaler(), numeric_features)],
        remainder='drop'
    )
    
    pipeline = Pipeline([
        ('feature_engineering', JobMatchPreprocessor()),
        ('column_transfer', col_transformer),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # 5. Hyperparameter Tuning
    print("\n--- Hyperparameter Tuning ---")
    # Random Search
    param_dist = {
        'classifier__n_estimators': [50, 100, 200, 300],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10]
    }
    
    print("Running RandomizedSearchCV...")
    random_search = RandomizedSearchCV(
        pipeline, param_distributions=param_dist, 
        n_iter=15, cv=3, scoring='roc_auc', random_state=42, n_jobs=-1
    )
    # Fit on train_full
    random_search.fit(X_train_full, y_train_full)
    
    print(f"Best Random Search AUC: {random_search.best_score_:.4f}")
    
    best_pipeline = random_search.best_estimator_
    
    # 6. Evaluate on Test Set
    test_preds = best_pipeline.predict(X_test)
    test_probs = best_pipeline.predict_proba(X_test)[:, 1]
    
    test_acc = accuracy_score(y_test, test_preds)
    test_roc_auc = roc_auc_score(y_test, test_probs)
    
    print("\n--- Final Tuned Model on TEST Set ---")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test ROC-AUC: {test_roc_auc:.4f}")
    print("\nClassification Report (Test):")
    print(classification_report(y_test, test_preds))
    
    # Feature Importances
    rf_model = best_pipeline.named_steps['classifier']
    importances = rf_model.feature_importances_
    
    print("\n--- Feature Importances (RF - Tuned) ---")
    for feat, imp in zip(numeric_features, importances):
        print(f"{feat}: {imp:.4f}")
        
    # 7. Check Gates
    print("\n--- Checking Evaluation Gates on TEST SET ---")
    assert test_roc_auc > 0.75, f"GATE FAILED: Test ROC-AUC ({test_roc_auc:.4f}) is not > 0.75"
    assert test_acc > 0.65, f"GATE FAILED: Test Accuracy ({test_acc:.4f}) is not > 0.65"
    print("ALL GATES PASSED ON TEST SET! SUCCESS")
        
    # Plotting ROC and PR Curves on Test Set
    print("\n--- Plotting Curves and Sweeping Thresholds (Test Set) ---")
    fpr, tpr, _ = roc_curve(y_test, test_probs)
    precision, recall, thresholds = precision_recall_curve(y_test, test_probs)
    
    os.makedirs('plots', exist_ok=True)
    plt.figure(figsize=(12, 5))
    
    # ROC Curve
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f'ROC curve (area = {test_roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Test)')
    plt.legend(loc="lower right")
    
    # PR Curve
    pr_auc = auc(recall, precision)
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Test)')
    plt.legend(loc="lower left")
    
    plt.tight_layout()
    plt.savefig('plots/rf_roc_pr_curves_tuned.png')
    print("Saved Tuned ROC and PR curves to plots/rf_roc_pr_curves_tuned.png")
    
    # Sweep thresholds to find best F1 on Test Set
    best_threshold = 0.5
    best_f1 = 0
    for threshold in thresholds:
        preds_at_thresh = (test_probs >= threshold).astype(int)
        f1 = f1_score(y_test, preds_at_thresh)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            
    print(f"Optimal Threshold for F1-Score on Test: {best_threshold:.4f} (F1: {best_f1:.4f})")
        
    # 8. Save final pipeline
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_pipeline, 'models/rf_job_match_pipeline_tuned.pkl')
    print("\nTuned RF Pipeline saved to models/rf_job_match_pipeline_tuned.pkl")

if __name__ == '__main__':
    main()
