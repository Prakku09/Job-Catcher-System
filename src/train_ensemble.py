import pandas as pd
import numpy as np
import time
import sys
import os
import joblib

sys.path.append('.')

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

from src.train_model import JobMatchPreprocessor
from src.ml_utils import check_dependencies, validate_data

def measure_latency(model, X, n_repeats=50):
    start = time.perf_counter()
    for _ in range(n_repeats):
        model.predict(X)
    return (time.perf_counter() - start) / n_repeats / len(X) * 1000  # ms/sample

def main():
    print("Checking dependencies...")
    check_dependencies()
    
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
    
    validate_data(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    
    numeric_features = ['diff_experience', 'diff_python', 'diff_ml', 'diff_sql', 'diff_stats', 'diff_js', 'diff_ds']
    
    # Common preprocessing step
    preprocessor = Pipeline([
        ('feature_engineering', JobMatchPreprocessor()),
        ('column_transfer', ColumnTransformer([('scaler', StandardScaler(), numeric_features)], remainder='drop'))
    ])
    
    # Pre-transform training data to feed into base models easily
    print("Transforming training data...")
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    
    cv_scheme = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # 3. Define Diverse Base Models
    print("\nDefining base models...")
    base_models = {
        "logistic_regression": LogisticRegression(random_state=42, class_weight='balanced'),
        "decision_tree":       DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, random_state=42),
        "knn":                 KNeighborsClassifier(n_neighbors=15),
        "naive_bayes":         GaussianNB(),
    }
    
    # Evaluate individual base models on validation folds
    best_single_roc = 0
    best_single_name = ""
    for name, model in base_models.items():
        roc = cross_val_score(model, X_train_trans, y_train, cv=cv_scheme, scoring='roc_auc').mean()
        print(f"  {name:20s} CV ROC-AUC: {roc:.4f}")
        if roc > best_single_roc:
            best_single_roc = roc
            best_single_name = name
            
    print(f"Best single model: {best_single_name} ({best_single_roc:.4f})")
    
    # 4. Ensemble Learning
    estimators = list(base_models.items())
    
    print("\nTraining Ensembles...")
    voting = VotingClassifier(estimators=estimators, voting="soft")
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(class_weight='balanced', random_state=42),
        cv=cv_scheme,   # <- critical to prevent data leakage
        n_jobs=-1
    )
    
    voting.fit(X_train_trans, y_train)
    stacking.fit(X_train_trans, y_train)
    
    # 5. Measure Diversity
    print("\nMeasuring Base Model Diversity...")
    preds = {name: model.fit(X_train_trans, y_train).predict(X_test_trans) for name, model in base_models.items()}
    names = list(preds.keys())
    disagreements = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            disag = np.mean(preds[names[i]] != preds[names[j]])
            disagreements.append(disag)
            print(f"  {names[i]} vs {names[j]}: {disag:.1%} disagreement")
    print(f"Average pairwise disagreement: {np.mean(disagreements):.1%}")
    
    # 6. Evaluate Ensembles vs Best Single Model on TEST set
    print("\nEvaluating Ensembles on TEST SET...")
    # Best single model test eval
    best_single_model = base_models[best_single_name].fit(X_train_trans, y_train)
    single_probs = best_single_model.predict_proba(X_test_trans)[:, 1]
    single_test_roc = roc_auc_score(y_test, single_probs)
    
    vote_probs = voting.predict_proba(X_test_trans)[:, 1]
    stack_probs = stacking.predict_proba(X_test_trans)[:, 1]
    
    vote_test_roc = roc_auc_score(y_test, vote_probs)
    stack_test_roc = roc_auc_score(y_test, stack_probs)
    
    print(f"  Best Single ({best_single_name}): {single_test_roc:.4f}")
    print(f"  Voting Classifier:     {vote_test_roc:.4f}")
    print(f"  Stacking Classifier:   {stack_test_roc:.4f}")
    
    # 7. Measure Latency Trade-off
    print("\nMeasuring Latency Overhead...")
    single_ms = measure_latency(best_single_model, X_test_trans)
    vote_ms = measure_latency(voting, X_test_trans)
    stack_ms = measure_latency(stacking, X_test_trans)
    
    print(f"  Single Model Latency:   {single_ms:.4f} ms/sample")
    print(f"  Voting Latency:         {vote_ms:.4f} ms/sample (Overhead: {(vote_ms/single_ms - 1)*100:.1f}%)")
    print(f"  Stacking Latency:       {stack_ms:.4f} ms/sample (Overhead: {(stack_ms/single_ms - 1)*100:.1f}%)")
    
    # Save the Stacking pipeline
    os.makedirs('models', exist_ok=True)
    full_stack_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('ensemble', stacking)
    ])
    joblib.dump(full_stack_pipeline, 'models/stacking_ensemble_pipeline.pkl')
    print("\nSaved full stacking pipeline to models/stacking_ensemble_pipeline.pkl")

if __name__ == '__main__':
    main()
