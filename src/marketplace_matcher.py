import pandas as pd
import logging
import sys
import os

# Ensure src module can be found if running from root
sys.path.append('.')

from src.train_model import JobMatchPreprocessor

# Setup basic logging
logger = logging.getLogger(__name__)

def generate_match_vector(student: dict, job: dict) -> dict:
    """
    Converts an existing Student + Job into the feature vector consumed by the ML model.
    Reuses the existing feature engineering logic.
    """
    # Combine into a single dictionary
    combined = {**student, **job}
    df = pd.DataFrame([combined])
    
    # 1. Base ML Features (used by the model)
    preprocessor = JobMatchPreprocessor()
    df_transformed = preprocessor.transform(df)
    ml_features = df_transformed.iloc[0].to_dict()
    
    # 2. We can also add other gaps directly from the raw data
    # (since preprocessor only returns the subset used by RF)
    exp_student = student.get('years_experience', 0)
    exp_job = job.get('exp_required_years', 0)
    diff_salary = job.get('salary_offered_inr', 0) - student.get('expected_salary_inr', 0)
    
    edu_map = {'UG': 1, 'PG': 2, 'PhD': 3}
    stu_edu = edu_map.get(student.get('education_level', 'UG'), 1)
    job_edu = edu_map.get(job.get('edu_minimum', 'UG'), 1)
    
    full_vector = {
        **ml_features,
        'diff_salary': diff_salary,
        'diff_education': stu_edu - job_edu,
        'location_match': 1 if student.get('location_student', '') == job.get('location_job', 'X') else 0
    }
    
    return full_vector

def validate_thresholds(student: dict, job: dict, thresholds: dict = None) -> dict:
    """
    Validates every competency before prediction based on configurable thresholds.
    """
    if thresholds is None:
        thresholds = {}

    results = {}
    
    # 1. Experience
    min_exp = thresholds.get('min_experience', job.get('exp_required_years', 0))
    results['experience'] = student.get('years_experience', 0) >= min_exp
    
    # 2. Education
    edu_map = {'UG': 1, 'PG': 2, 'PhD': 3}
    stu_edu = edu_map.get(student.get('education_level', 'UG'), 1)
    req_edu_level = thresholds.get('min_education', job.get('edu_minimum', 'UG'))
    min_edu = edu_map.get(req_edu_level, 1)
    results['education'] = stu_edu >= min_edu
    
    # 3. Technical Skills
    # Use thresholds if provided, else use job requirements
    req_python = thresholds.get('min_python', job.get('python_required', 0))
    req_sql = thresholds.get('min_sql', job.get('sql_required', 0))
    req_ml = thresholds.get('min_ml', job.get('ml_required', 0))
    
    results['python'] = student.get('python_score', 0) >= req_python
    results['sql'] = student.get('sql_score', 0) >= req_sql
    results['machine_learning'] = student.get('ml_score', 0) >= req_ml
    
    # 4. Salary
    min_salary = thresholds.get('min_salary', student.get('expected_salary_inr', 0))
    results['salary'] = job.get('salary_offered_inr', 0) >= min_salary
    
    # 5. Location
    req_loc = thresholds.get('location_match_required', False)
    if req_loc:
        results['location'] = student.get('location_student', '') == job.get('location_job', '')
    else:
        results['location'] = True # Not required to match exactly
        
    return results

def map_competencies(validation_results: dict) -> dict:
    """
    Maps validation results into competency groups.
    """
    return {
        'Technical Skills': {
            'python': validation_results.get('python', False),
            'sql': validation_results.get('sql', False),
            'machine_learning': validation_results.get('machine_learning', False)
        },
        'Professional Experience': {
            'experience': validation_results.get('experience', False)
        },
        'Education': {
            'degree': validation_results.get('education', False)
        },
        'Domain Fit': {
            'salary_expectation': validation_results.get('salary', False),
            'location': validation_results.get('location', False)
        }
    }

if __name__ == "__main__":
    # Sample Output Generation
    sample_student = {
        "years_experience": 3.0,
        "python_score": 8.0,
        "sql_score": 7.0,
        "ml_score": 6.5,
        "statistics_score": 6.0,
        "javascript_score": 0.0,
        "data_structures_score": 7.0,
        "expected_salary_inr": 800000,
        "education_level": "PG",
        "location_student": "Bangalore"
    }
    
    sample_job = {
        "title": "Data Scientist",
        "exp_required_years": 2.0,
        "python_required": 7.0,
        "sql_required": 6.0,
        "ml_required": 5.0,
        "statistics_required": 6.0,
        "javascript_required": 0.0,
        "data_structures_required": 5.0,
        "salary_offered_inr": 1200000,
        "edu_minimum": "UG",
        "location_job": "Bangalore"
    }
    
    sample_thresholds = {
        "min_python": 7.5,
        "location_match_required": True
    }
    
    print("--- 1. Generated Match Vector ---")
    vector = generate_match_vector(sample_student, sample_job)
    for k, v in vector.items():
        print(f"  {k}: {v}")
        
    print("\n--- 2. Threshold Validation ---")
    val_results = validate_thresholds(sample_student, sample_job, sample_thresholds)
    for k, v in val_results.items():
        print(f"  {k}: {'Pass' if v else 'Fail'}")
        
    print("\n--- 3. Competency Mapping ---")
    mapping = map_competencies(val_results)
    import json
    print(json.dumps(mapping, indent=2))
