import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

df = pd.read_csv('src/data/clean_modelling_table.csv')

def get_domain_score(row):
    title = str(row.get('title', '')).lower()
    if 'data' in title or 'ml' in title or 'ai' in title or 'machine learning' in title or 'analyst' in title:
        return max(row.get('python_score', 0), row.get('sql_score', 0), row.get('ml_score', 0), row.get('statistics_score', 0))
    else:
        return max(row.get('javascript_score', 0), row.get('data_structures_score', 0), row.get('python_score', 0))

df['domain_match'] = df.apply(get_domain_score, axis=1)

edu_map = {'UG': 1, 'PG': 2, 'PhD': 3}
df['student_edu_num'] = df['education_level'].map(edu_map).fillna(1)
df['job_edu_num'] = df['edu_minimum'].map(edu_map).fillna(1)
df['diff_education'] = df['student_edu_num'] - df['job_edu_num']

df['location_match'] = (df['location_student'].fillna('') == df['location_job'].fillna('')).astype(int)
df['diff_salary'] = df['salary_offered_inr'].fillna(0) - df['expected_salary_inr'].fillna(0)

df['diff_python'] = df['python_score'].fillna(0) - df['python_required'].fillna(0)
df['diff_ml'] = df['ml_score'].fillna(0) - df['ml_required'].fillna(0)
df['diff_sql'] = df['sql_score'].fillna(0) - df['sql_required'].fillna(0)
df['diff_stats'] = df['statistics_score'].fillna(0) - df['statistics_required'].fillna(0)
df['diff_js'] = df['javascript_score'].fillna(0) - df['javascript_required'].fillna(0)
df['diff_ds'] = df['data_structures_score'].fillna(0) - df['data_structures_required'].fillna(0)
df['diff_experience'] = df['years_experience'].fillna(0) - df['exp_required_years'].fillna(0)

features = [
    'diff_experience', 'diff_python', 'diff_ml', 'diff_sql', 'diff_stats', 
    'diff_js', 'diff_ds', 'diff_salary', 'domain_match', 'diff_education', 'location_match'
]

X = df[features].fillna(0)
y = df['is_good_match']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

print("Training SVC...")
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(probability=True, random_state=42))
])
param_dist = {
    'svc__C': [0.1, 1, 10, 50, 100],
    'svc__gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
    'svc__kernel': ['rbf', 'poly', 'sigmoid']
}
rs = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=30, cv=5, scoring='accuracy', random_state=42, n_jobs=-1)
rs.fit(X_train, y_train)
preds = rs.predict(X_test)
print(f"SVC Acc: {accuracy_score(y_test, preds):.4f}")
print("Best params:", rs.best_params_)
