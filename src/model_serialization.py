import os
import json
import logging
import time
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import sklearn
import sys
import subprocess
from pydantic import BaseModel, Field, ValidationError
from typing import Optional, List, Dict, Any

# Ensure correct path resolution for custom classes (e.g. JobMatchPreprocessor)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.train_model import JobMatchPreprocessor

# ---------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/model_loading.log'),
        logging.StreamHandler()
    ]
)

# ---------------------------------------------------------
# Pydantic Schema for Input Validation
# ---------------------------------------------------------
class JobMatchInput(BaseModel):
    application_id: Optional[str] = "unknown"
    student_id: Optional[str] = "unknown"
    job_id: Optional[str] = "unknown"
    years_experience: float = Field(default=0.0, ge=0)
    exp_required_years: float = Field(default=0.0, ge=0)
    python_score: float = Field(default=0.0, ge=0, le=100)
    sql_score: float = Field(default=0.0, ge=0, le=100)
    ml_score: float = Field(default=0.0, ge=0, le=100)
    javascript_score: float = Field(default=0.0, ge=0, le=100)
    data_structures_score: float = Field(default=0.0, ge=0, le=100)
    statistics_score: float = Field(default=0.0, ge=0, le=100)
    python_required: float = Field(default=0.0, ge=0, le=100)
    sql_required: float = Field(default=0.0, ge=0, le=100)
    ml_required: float = Field(default=0.0, ge=0, le=100)
    javascript_required: float = Field(default=0.0, ge=0, le=100)
    data_structures_required: float = Field(default=0.0, ge=0, le=100)
    statistics_required: float = Field(default=0.0, ge=0, le=100)
    expected_salary_inr: float = Field(default=0.0, ge=0)
    salary_offered_inr: float = Field(default=0.0, ge=0)
    education_level: str = "UG"
    edu_minimum: str = "UG"
    location_student: str = "Unknown"
    location_job: str = "Unknown"


# ---------------------------------------------------------
# Model Loader Class
# ---------------------------------------------------------
class JobMatchModelLoader:
    def __init__(self, model_path: str, metadata_path: str):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.model = None
        self.metadata = None
        
    def load(self):
        logging.info(f"Attempting to load model from {self.model_path}")
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file {self.model_path} does not exist.")
            if not os.path.exists(self.metadata_path):
                raise FileNotFoundError(f"Metadata file {self.metadata_path} does not exist.")
                
            self.model = joblib.load(self.model_path)
            logging.info("Model loaded successfully.")
            
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
            logging.info("Metadata loaded successfully.")
            
            self._verify_compatibility()
            
        except Exception as e:
            logging.error(f"Failed to load model or metadata: {str(e)}")
            raise
            
    def _verify_compatibility(self):
        # Python version check (major.minor)
        current_py = f"{sys.version_info.major}.{sys.version_info.minor}"
        meta_py = self.metadata.get('environment', {}).get('python_version', '')
        if meta_py and not meta_py.startswith(current_py):
            logging.warning(f"Python version mismatch: Running {current_py} vs Model {meta_py}")
            
        # Sklearn version check
        current_sk = sklearn.__version__
        meta_sk = self.metadata.get('environment', {}).get('sklearn_version', '')
        if current_sk != meta_sk:
            logging.warning(f"Scikit-learn version mismatch: Running {current_sk} vs Model {meta_sk}")
            
        logging.info("Compatibility verification completed.")

# ---------------------------------------------------------
# Predict Function
# ---------------------------------------------------------
def predict_match(input_data: Dict[str, Any], loader: JobMatchModelLoader):
    start_time = time.perf_counter()
    
    # Validation
    try:
        validated_data = JobMatchInput(**input_data)
    except ValidationError as e:
        logging.error(f"Input validation failed: {e.errors()}")
        return {"status": "error", "error_type": "ValidationError", "details": e.errors()}
        
    if loader.model is None:
        logging.error("Model is not loaded.")
        return {"status": "error", "error_type": "ModelNotLoaded"}
        
    try:
        # Preprocess single record by wrapping in DataFrame
        df_input = pd.DataFrame([validated_data.model_dump()])
        
        probs = loader.model.predict_proba(df_input)
        prob_match = probs[0][1]
        prediction = 1 if prob_match >= 0.5 else 0
        
        # Confidence score (distance from decision boundary)
        confidence = abs(prob_match - 0.5) * 2  # Scales to [0, 1]
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        logging.info(f"Prediction successful. Result: {prediction}, Confidence: {confidence:.3f}, Latency: {latency_ms:.2f}ms")
        
        return {
            "status": "success",
            "prediction": int(prediction),
            "probability": float(prob_match),
            "confidence_score": float(confidence),
            "latency_ms": round(latency_ms, 2)
        }
        
    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        return {"status": "error", "error_type": "PredictionFailure", "details": str(e)}

# ---------------------------------------------------------
# Serialization and Verification Logic
# ---------------------------------------------------------
def main():
    os.makedirs('models', exist_ok=True)
    os.makedirs('artifacts', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    source_model_path = 'models/best_tuned_model.pkl'
    target_model_path = 'models/job_match_pipeline_v1.0.pkl'
    metadata_path = 'artifacts/metadata.json'
    
    logging.info("Starting Serialization process...")
    
    # 1. Load source model
    if not os.path.exists(source_model_path):
        logging.error(f"Source model {source_model_path} not found. Ensure Task 17 was completed.")
        sys.exit(1)
        
    model_pipeline = joblib.load(source_model_path)
    logging.info(f"Loaded source model from {source_model_path}")
    
    # 2. Extract metrics (simulate or read from previous task)
    test_results_path = 'reports/test_results.json'
    try:
        with open(test_results_path, 'r') as f:
            metrics_data = json.load(f)
        train_metrics = metrics_data.get('tuned_metrics', {})
    except Exception:
        logging.warning("Could not read test_results.json. Using empty metrics.")
        train_metrics = {}
        
    try:
        git_commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('utf-8').strip()
    except Exception:
        git_commit = "unknown"
        
    # 3. Create Metadata
    metadata = {
        "model_name": "Job-Catcher-Match-Pipeline",
        "model_version": "1.0",
        "training_timestamp": datetime.utcnow().isoformat() + "Z",
        "git_commit": git_commit,
        "dataset_name": "clean_modelling_table.csv",
        "n_training_samples": 425, # approx 85% of 500
        "selected_features": [
            'years_experience', 'exp_required_years', 'python_score', 'python_required',
            'sql_score', 'sql_required', 'ml_score', 'ml_required', 'javascript_score',
            'javascript_required', 'data_structures_score', 'data_structures_required',
            'statistics_score', 'statistics_required', 'expected_salary_inr', 'salary_offered_inr',
            'education_level', 'edu_minimum', 'location_student', 'location_job'
        ],
        "model_type": "Pipeline(JobMatchPreprocessor -> StandardScaler -> XGBClassifier)",
        "hyperparameters": getattr(model_pipeline.named_steps.get('classifier'), 'get_params', lambda: {})(),
        "training_metrics": train_metrics,
        "environment": {
            "python_version": sys.version.split()[0],
            "sklearn_version": sklearn.__version__,
            "joblib_version": joblib.__version__
        },
        "random_seed": 42
    }
    
    # 4. Serialize Model & Metadata
    joblib.dump(model_pipeline, target_model_path)
    logging.info(f"Serialized model saved to {target_model_path}")
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    logging.info(f"Metadata saved to {metadata_path}")
    
    # 5. Verification (Fresh Load)
    logging.info("Verifying serialization integrity...")
    loader = JobMatchModelLoader(target_model_path, metadata_path)
    loader.load()
    
    # Generate mock sample representing a real row
    sample_data = {
        "years_experience": 2.5,
        "exp_required_years": 3.0,
        "python_score": 85.0,
        "python_required": 80.0,
        "education_level": "PG",
        "edu_minimum": "UG",
        "location_student": "Bengaluru",
        "location_job": "Bengaluru",
        "expected_salary_inr": 800000,
        "salary_offered_inr": 850000
    }
    
    # Pre-serialization output (using original pipeline object)
    df_sample = pd.DataFrame([JobMatchInput(**sample_data).model_dump()])
    pre_prob = float(model_pipeline.predict_proba(df_sample)[0][1])
    
    # Post-serialization output (using loaded object)
    post_pred = predict_match(sample_data, loader)
    post_prob = post_pred['probability']
    
    is_identical = (abs(pre_prob - post_prob) < 1e-6)
    
    validation_res = {
        "pre_serialization_probability": pre_prob,
        "post_serialization_probability": post_prob,
        "is_identical": is_identical,
        "status": "PASS" if is_identical else "FAIL"
    }
    
    with open('artifacts/model_validation.json', 'w') as f:
        json.dump(validation_res, f, indent=4)
        
    logging.info(f"Serialization Verification: {validation_res['status']}")
    assert is_identical, "Serialization failed: Pre and Post probabilities do not match!"
    
    # 6. Prediction Examples CSV
    samples = [
        sample_data,
        {**sample_data, "python_score": 20.0, "python_required": 90.0, "location_student": "Delhi"},
        {**sample_data, "years_experience": 10.0, "exp_required_years": 2.0}
    ]
    
    results = []
    for s in samples:
        res = predict_match(s, loader)
        results.append({**s, "Predicted_Probability": res.get("probability"), "Prediction": res.get("prediction")})
        
    pd.DataFrame(results).to_csv('artifacts/prediction_examples.csv', index=False)
    
    # 7. Serialization Report
    ser_report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_path": target_model_path,
        "metadata_path": metadata_path,
        "file_size_bytes": os.path.getsize(target_model_path),
        "verification": validation_res
    }
    with open('artifacts/serialization_report.json', 'w') as f:
        json.dump(ser_report, f, indent=4)
        
    # 8. Markdown Reports
    with open('reports/serialization_summary.md', 'w') as f:
        f.write("# Model Serialization Summary\n\n")
        f.write(f"- **Model Saved As**: `{target_model_path}`\n")
        f.write(f"- **File Size**: {os.path.getsize(target_model_path) / 1024:.2f} KB\n")
        f.write(f"- **Verification Status**: {validation_res['status']}\n")
        f.write("\n## Metadata\n```json\n" + json.dumps(metadata, indent=2) + "\n```\n")
        
    with open('reports/compatibility_report.md', 'w') as f:
        f.write("# Environment Compatibility Report\n\n")
        f.write(f"- **Python Version**: {sys.version.split()[0]}\n")
        f.write(f"- **Scikit-Learn**: {sklearn.__version__}\n")
        f.write(f"- **Joblib**: {joblib.__version__}\n\n")
        f.write("> [!IMPORTANT]\n> Deployments using this artifact must closely match these package versions to guarantee consistency.\n")
        
    logging.info("Task 19 Complete!")


if __name__ == "__main__":
    main()
