import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import sys
sys.path.append('.')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.inspection import PartialDependenceDisplay

from src.train_model import JobMatchPreprocessor

def main():
    # 1. Load Data
    data_path = 'src/data/clean_modelling_table.csv'
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # 2. Preprocess & Split
    leakage_cols = ['exam_time_seconds', 'self_reported_confidence', 'retake_count', 
                    'application_id', 'student_id', 'job_id', 'name', 'application_date']
    df = df.drop(columns=leakage_cols, errors='ignore')
    
    y = df['is_good_match']
    X = df.drop(columns=['is_good_match'], errors='ignore')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    
    numeric_features = ['diff_experience', 'diff_python', 'diff_ml', 'diff_sql', 'diff_stats', 'diff_js', 'diff_ds']
    
    # 3. Train Baseline Logistic Regression
    print("\nTraining Linear Baseline (Logistic Regression)...")
    lr_pipeline = Pipeline([
        ('feature_engineering', JobMatchPreprocessor()),
        ('column_transfer', ColumnTransformer([('scaler', StandardScaler(), numeric_features)], remainder='drop')),
        ('classifier', LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000))
    ])
    lr_pipeline.fit(X_train, y_train)
    
    lr_preds = lr_pipeline.predict(X_test)
    lr_probs = lr_pipeline.predict_proba(X_test)[:, 1]
    lr_acc = accuracy_score(y_test, lr_preds)
    lr_roc = roc_auc_score(y_test, lr_probs)
    
    # 4. Load Random Forest
    print("\nLoading Tuned Random Forest...")
    rf_pipeline = joblib.load('models/rf_job_match_pipeline_tuned.pkl')
    
    rf_preds = rf_pipeline.predict(X_test)
    rf_probs = rf_pipeline.predict_proba(X_test)[:, 1]
    rf_acc = accuracy_score(y_test, rf_preds)
    rf_roc = roc_auc_score(y_test, rf_probs)
    
    # 5. Evaluate and Print LIFT
    print("\n" + "="*40)
    print("MODEL COMPARISON (LINEAR vs COMPLEX)")
    print("="*40)
    print(f"Linear Baseline (LogReg) | Acc: {lr_acc:.4f} | ROC-AUC: {lr_roc:.4f}")
    print(f"Random Forest (Tuned)    | Acc: {rf_acc:.4f} | ROC-AUC: {rf_roc:.4f}")
    print("-" * 40)
    print(f"LIFT (Accuracy): +{(rf_acc - lr_acc)*100:.2f}%")
    print(f"LIFT (ROC-AUC):  +{(rf_roc - lr_roc)*100:.2f}%")
    print("="*40)
    
    # 6. Generate Partial Dependence Plots (PDP)
    print("\nGenerating Partial Dependence Plots (PDP)...")
    os.makedirs('plots', exist_ok=True)
    
    # Pass the preprocessed data to the remaining steps
    X_train_transformed = rf_pipeline.named_steps['feature_engineering'].transform(X_train)
    
    pdp_pipeline = Pipeline([
        ('column_transfer', rf_pipeline.named_steps['column_transfer']),
        ('classifier', rf_pipeline.named_steps['classifier'])
    ])
    
    fig, ax = plt.subplots(figsize=(15, 5))
    features_to_plot = ['diff_experience', 'diff_js', 'diff_python']
    
    disp1 = PartialDependenceDisplay.from_estimator(
        pdp_pipeline, 
        X_train_transformed, 
        features=features_to_plot,
        feature_names=numeric_features,
        ax=ax,
        grid_resolution=50
    )
    plt.suptitle("1-D Partial Dependence Plots: Impact on 'Good Match' Probability")
    plt.tight_layout()
    plt.savefig('plots/pdp_1d_analysis.png')
    print("Saved 1-D PDP to plots/pdp_1d_analysis.png")
    
    # Generate 2D PDP
    fig, ax = plt.subplots(figsize=(8, 6))
    features_2d = [('diff_experience', 'diff_js')]
    disp2 = PartialDependenceDisplay.from_estimator(
        pdp_pipeline, 
        X_train_transformed, 
        features=features_2d,
        feature_names=numeric_features,
        ax=ax,
        grid_resolution=30
    )
    plt.suptitle("2-D Partial Dependence: Interaction of Experience and JS Skills")
    plt.tight_layout()
    plt.savefig('plots/pdp_2d_analysis.png')
    print("Saved 2-D PDP to plots/pdp_2d_analysis.png")

if __name__ == '__main__':
    main()
