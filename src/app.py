from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import sys
import os
import json
import time
import uuid
import logging
from typing import List

# Ensure src module can be found if running from root
sys.path.append('.')

app = FastAPI(
    title="Job Match API",
    description="Live REST API for binary classification of Job/Candidate matches.",
    version="1.0.0"
)

# Global variables for model artifacts
model_pkg = None

# Metrics state
metrics_state = {
    "total_requests": 0,
    "total_errors": 0,
    "latency_sum": 0.0
}

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("job_catcher_api")

class MatchRequest(BaseModel):
    # Candidate Features
    years_experience: float = Field(..., description="Candidate's years of experience")
    python_score: float = Field(0.0, description="Candidate's Python assessment score (0-10)")
    sql_score: float = Field(0.0, description="Candidate's SQL assessment score (0-10)")
    ml_score: float = Field(0.0, description="Candidate's ML assessment score (0-10)")
    statistics_score: float = Field(0.0, description="Candidate's Statistics assessment score (0-10)")
    javascript_score: float = Field(0.0, description="Candidate's JavaScript assessment score (0-10)")
    data_structures_score: float = Field(0.0, description="Candidate's Data Structures score (0-10)")
    expected_salary_inr: float = Field(..., description="Candidate's expected salary in INR")
    education_level: str = Field(..., description="Candidate's education level (e.g. UG, PG, PhD)")
    location_student: str = Field(..., description="Candidate's preferred/current location")
    
    # Job Features
    title: str = Field("", description="Job title")
    exp_required_years: float = Field(..., description="Job's required years of experience")
    python_required: float = Field(0.0, description="Job's required Python score")
    sql_required: float = Field(0.0, description="Job's required SQL score")
    ml_required: float = Field(0.0, description="Job's required ML score")
    statistics_required: float = Field(0.0, description="Job's required Statistics score")
    javascript_required: float = Field(0.0, description="Job's required JavaScript score")
    data_structures_required: float = Field(0.0, description="Job's required Data Structures score")
    salary_offered_inr: float = Field(..., description="Job's offered salary in INR")
    edu_minimum: str = Field(..., description="Job's minimum education level required")
    location_job: str = Field(..., description="Job's location")

class BatchMatchRequest(BaseModel):
    requests: List[MatchRequest] = Field(..., description="List of match requests to process in batch")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    req_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Store req_id in state so endpoints can access it
    request.state.req_id = req_id
    
    metrics_state["total_requests"] += 1
    
    try:
        response = await call_next(request)
        latency = time.time() - start_time
        metrics_state["latency_sum"] += latency
        
        # Log successful request
        logger.info(json.dumps({
            "request_id": req_id,
            "timestamp": time.time(),
            "method": request.method,
            "path": request.url.path,
            "latency_s": latency,
            "model_version": "1.0.0" if model_pkg else "not_loaded",
            "status_code": response.status_code,
            "error": None
        }))
        return response
    except Exception as e:
        latency = time.time() - start_time
        metrics_state["latency_sum"] += latency
        metrics_state["total_errors"] += 1
        
        # Log error
        logger.error(json.dumps({
            "request_id": req_id,
            "timestamp": time.time(),
            "method": request.method,
            "path": request.url.path,
            "latency_s": latency,
            "model_version": "1.0.0" if model_pkg else "not_loaded",
            "status_code": 500,
            "error": str(e)
        }))
        raise

@app.on_event("startup")
def load_model():
    global model_pkg
    model_path = "models/production_binary_classifier.pkl"
    threshold_path = "reports/operating_point.json"
    
    if not os.path.exists(model_path):
        print(f"WARNING: Model package not found at {model_path}!")
        return
        
    model = joblib.load(model_path)
    
    threshold = 0.5
    if os.path.exists(threshold_path):
        with open(threshold_path, 'r') as f:
            op_data = json.load(f)
            threshold = op_data.get("optimal_threshold", 0.5)
            
    model_pkg = {
        "model": model,
        "threshold": threshold
    }
    print(f"Successfully loaded model and threshold: {threshold}")

@app.get("/", include_in_schema=False)
def root():
    """Redirects the root URL to the Swagger UI."""
    return RedirectResponse(url="/docs")

@app.get("/health")
def health_check():
    """Basic health check endpoint to confirm API is running and model is loaded."""
    if model_pkg is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Server not ready.")
    return {"status": "ok", "model_loaded": True}

@app.get("/metadata")
def get_metadata():
    """Returns metadata about the deployed model."""
    if model_pkg is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Server not ready.")
    return {
        "model_type": "Logistic Regression (Calibrated)",
        "framework": "Scikit-Learn",
        "threshold_used": model_pkg["threshold"],
        "version": "1.0.0",
        "author": "Job-Catcher-System"
    }

@app.post("/predict")
def predict_match(request: MatchRequest, raw_request: Request):
    """
    Predicts whether a candidate is a good match for a job based on the provided features.
    Uses the pre-trained and calibrated model with cost-based thresholding.
    """
    if model_pkg is None:
        metrics_state["total_errors"] += 1
        raise HTTPException(status_code=503, detail="Model not loaded. Server not ready.")
        
    try:
        # Convert incoming JSON payload to a dictionary, then to a DataFrame (1 row)
        input_data = request.dict()
        df_input = pd.DataFrame([input_data])
        
        # Extract components from the package
        model = model_pkg["model"]
        threshold = model_pkg["threshold"]
        
        # Predict probability
        proba = model.predict_proba(df_input)[0, 1]
        
        # Make decision using bundled threshold
        decision = 1 if proba >= threshold else 0
        
        # Log prediction result
        logger.info(json.dumps({
            "request_id": getattr(raw_request.state, "req_id", "unknown"),
            "event": "prediction_result",
            "prediction": decision,
            "probability": float(proba)
        }))
        
        return {
            "probability": round(float(proba), 4),
            "is_good_match": int(decision),
            "threshold_used": float(threshold)
        }
    except Exception as e:
        metrics_state["total_errors"] += 1
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@app.post("/batch_predict")
def batch_predict_match(request: BatchMatchRequest, raw_request: Request):
    """
    Batch predicts whether candidates are good matches for jobs based on the provided features.
    """
    if model_pkg is None:
        metrics_state["total_errors"] += 1
        raise HTTPException(status_code=503, detail="Model not loaded. Server not ready.")
        
    if not request.requests:
        metrics_state["total_errors"] += 1
        raise HTTPException(status_code=400, detail="Empty batch payload")
        
    try:
        # Convert list of incoming JSON payloads to a DataFrame
        input_data = [req.dict() for req in request.requests]
        df_input = pd.DataFrame(input_data)
        
        # Extract components from the package
        model = model_pkg["model"]
        threshold = model_pkg["threshold"]
        
        # Predict probabilities
        probas = model.predict_proba(df_input)[:, 1]
        
        # Make decisions
        decisions = [1 if p >= threshold else 0 for p in probas]
        
        results = []
        for i, (p, d) in enumerate(zip(probas, decisions)):
            results.append({
                "index": i,
                "probability": round(float(p), 4),
                "is_good_match": int(d),
                "threshold_used": float(threshold)
            })
            
        # Log batch prediction result summary
        logger.info(json.dumps({
            "request_id": getattr(raw_request.state, "req_id", "unknown"),
            "event": "batch_prediction_result",
            "batch_size": len(results),
            "positive_matches": sum(decisions)
        }))
        
        return {"batch_results": results}
    except Exception as e:
        metrics_state["total_errors"] += 1
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {str(e)}")

@app.get("/metrics")
def get_metrics():
    """Returns simple API performance metrics."""
    avg_latency = 0.0
    if metrics_state["total_requests"] > 0:
        avg_latency = metrics_state["latency_sum"] / metrics_state["total_requests"]
        
    return {
        "total_requests": metrics_state["total_requests"],
        "total_errors": metrics_state["total_errors"],
        "average_latency_seconds": round(avg_latency, 4)
    }
