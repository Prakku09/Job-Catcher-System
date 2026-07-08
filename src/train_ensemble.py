import pandas as pd
import numpy as np
import time
import sys
import os
import joblib
import json
import subprocess
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

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
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, precision_recall_curve, auc
)

from src.train_model import JobMatchPreprocessor
from src.ml_utils import check_dependencies, validate_data

def measure_latency(model, X, n_repeats=50):
    start = time.perf_counter()
    for _ in range(n_repeats):
        model.predict(X)
    return (time.perf_counter() - start) / n_repeats / len(X) * 1000  # ms/sample

def calculate_metrics(y_true, y_pred, y_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(auc(recall, precision))
    }

def save_confusion_matrix(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def main():
    print("Checking dependencies...")
    check_dependencies()
    
    # Setup directories
    for d in ['models', 'reports', 'plots']:
        os.makedirs(d, exist_ok=True)
        
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
    best_single_model = base_models[best_single_name].fit(X_train_trans, y_train)
    
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
    # Fit all models on the full training set
    for name, model in base_models.items():
        if name != best_single_name:
            model.fit(X_train_trans, y_train)
            
    preds = {name: model.predict(X_train_trans) for name, model in base_models.items()}
    names = list(preds.keys())
    disagreements = []
    disagreement_matrix = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            disag = float(np.mean(preds[names[i]] != preds[names[j]]))
            disagreements.append(disag)
            pair_name = f"{names[i]}_vs_{names[j]}"
            disagreement_matrix[pair_name] = disag
            print(f"  {names[i]} vs {names[j]}: {disag:.1%} disagreement")
            
    avg_disagreement = float(np.mean(disagreements))
    print(f"Average pairwise disagreement: {avg_disagreement:.1%}")
    
    diversity_report = {
        "disagreement_matrix": disagreement_matrix,
        "average_disagreement": avg_disagreement,
        "explanation": "High disagreement indicates that base models are capturing different patterns, which is ideal for an ensemble to perform well."
    }
    with open('reports/diversity_report.json', 'w') as f:
        json.dump(diversity_report, f, indent=4)
    
    # 6. Evaluate Ensembles vs Best Single Model on TEST set
    print("\nEvaluating Ensembles on TEST SET...")
    
    # Predictions
    single_preds = best_single_model.predict(X_test_trans)
    single_probs = best_single_model.predict_proba(X_test_trans)[:, 1]
    
    vote_preds = voting.predict(X_test_trans)
    vote_probs = voting.predict_proba(X_test_trans)[:, 1]
    
    stack_preds = stacking.predict(X_test_trans)
    stack_probs = stacking.predict_proba(X_test_trans)[:, 1]
    
    # Metrics
    metrics_single = calculate_metrics(y_test, single_preds, single_probs)
    metrics_voting = calculate_metrics(y_test, vote_preds, vote_probs)
    metrics_stacking = calculate_metrics(y_test, stack_preds, stack_probs)
    
    # 7. Measure Latency Trade-off
    print("\nMeasuring Latency Overhead...")
    single_ms = measure_latency(best_single_model, X_test_trans)
    vote_ms = measure_latency(voting, X_test_trans)
    stack_ms = measure_latency(stacking, X_test_trans)
    
    # 8. Classification Reports
    class_reports = {
        "best_single_model": classification_report(y_test, single_preds, output_dict=True),
        "voting_classifier": classification_report(y_test, vote_preds, output_dict=True),
        "stacking_classifier": classification_report(y_test, stack_preds, output_dict=True)
    }
    with open('reports/classification_reports.json', 'w') as f:
        json.dump(class_reports, f, indent=4)
        
    # 9. Confusion Matrices
    save_confusion_matrix(y_test, single_preds, "Best Single Model", "plots/confusion_matrix_single.png")
    save_confusion_matrix(y_test, vote_preds, "Voting Classifier", "plots/confusion_matrix_voting.png")
    save_confusion_matrix(y_test, stack_preds, "Stacking Classifier", "plots/confusion_matrix_stacking.png")
    
    # 10. ROC Curves
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, single_probs)
    plt.plot(fpr, tpr, label=f'Best Single ({metrics_single["roc_auc"]:.3f})')
    fpr, tpr, _ = roc_curve(y_test, vote_probs)
    plt.plot(fpr, tpr, label=f'Voting ({metrics_voting["roc_auc"]:.3f})')
    fpr, tpr, _ = roc_curve(y_test, stack_probs)
    plt.plot(fpr, tpr, label=f'Stacking ({metrics_stacking["roc_auc"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Ensemble ROC Curves')
    plt.legend()
    plt.savefig('plots/ensemble_roc_curve.png', bbox_inches='tight')
    plt.close()
    
    # 11. PR Curves
    plt.figure(figsize=(8, 6))
    prec, rec, _ = precision_recall_curve(y_test, single_probs)
    plt.plot(rec, prec, label=f'Best Single ({metrics_single["pr_auc"]:.3f})')
    prec, rec, _ = precision_recall_curve(y_test, vote_probs)
    plt.plot(rec, prec, label=f'Voting ({metrics_voting["pr_auc"]:.3f})')
    prec, rec, _ = precision_recall_curve(y_test, stack_probs)
    plt.plot(rec, prec, label=f'Stacking ({metrics_stacking["pr_auc"]:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Ensemble Precision-Recall Curves')
    plt.legend()
    plt.savefig('plots/ensemble_pr_curve.png', bbox_inches='tight')
    plt.close()
    
    # 12. Model Comparison CSV
    comparison_data = [
        {"Model": "Best Single", **metrics_single, "Latency(ms)": single_ms},
        {"Model": "Voting", **metrics_voting, "Latency(ms)": vote_ms},
        {"Model": "Stacking", **metrics_stacking, "Latency(ms)": stack_ms}
    ]
    pd.DataFrame(comparison_data).to_csv('reports/model_comparison.csv', index=False)
    
    # 13. Feature Importance (if applicable)
    if hasattr(best_single_model, 'feature_importances_'):
        importances = best_single_model.feature_importances_
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances, y=numeric_features)
        plt.title(f'Feature Importance ({best_single_name})')
        plt.savefig('plots/feature_importance.png', bbox_inches='tight')
        plt.close()
    elif hasattr(best_single_model, 'coef_'):
        importances = np.abs(best_single_model.coef_[0])
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances, y=numeric_features)
        plt.title(f'Feature Importance Absolute Coefs ({best_single_name})')
        plt.savefig('plots/feature_importance.png', bbox_inches='tight')
        plt.close()
        
    # 14. Overfitting Analysis (Train vs Test for Stacking)
    train_preds = stacking.predict(X_train_trans)
    train_probs = stacking.predict_proba(X_train_trans)[:, 1]
    
    train_acc = accuracy_score(y_train, train_preds)
    train_roc = roc_auc_score(y_train, train_probs)
    
    overfitting = {
        "train_accuracy": float(train_acc),
        "test_accuracy": float(metrics_stacking['accuracy']),
        "train_roc_auc": float(train_roc),
        "test_roc_auc": float(metrics_stacking['roc_auc']),
        "generalization_gap": float(train_roc - metrics_stacking['roc_auc'])
    }
    with open('reports/overfitting_analysis.json', 'w') as f:
        json.dump(overfitting, f, indent=4)
        
    # 15. Save Models
    joblib.dump(Pipeline([('preprocessor', preprocessor), ('ensemble', stacking)]), 'models/stacking_ensemble_pipeline.pkl')
    joblib.dump(Pipeline([('preprocessor', preprocessor), ('ensemble', voting)]), 'models/voting_classifier.pkl')
    joblib.dump(Pipeline([('preprocessor', preprocessor), ('ensemble', best_single_model)]), 'models/best_single_model.pkl')

    # 16. Reproducibility & ensemble_metrics.json
    try:
        git_commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('utf-8').strip()
    except:
        git_commit = "unknown"
        
    ensemble_metrics = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "git_commit": git_commit,
        "python_version": sys.version.split()[0],
        "sklearn_version": sklearn.__version__,
        "random_seed": 42,
        "dataset": data_path,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "best_single_model": {
            "name": best_single_name,
            **metrics_single
        },
        "voting_classifier": metrics_voting,
        "stacking_classifier": metrics_stacking,
        "average_pairwise_disagreement": avg_disagreement,
        "latency": {
            "single_model_ms": single_ms,
            "voting_ms": vote_ms,
            "stacking_ms": stack_ms
        }
    }
    with open('reports/ensemble_metrics.json', 'w') as f:
        json.dump(ensemble_metrics, f, indent=4)
        
    print("\nAll evaluation artifacts generated successfully in models/, reports/, and plots/.")

if __name__ == '__main__':
    main()
