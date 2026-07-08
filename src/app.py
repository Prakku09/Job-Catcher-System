from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import sys
import os

# Ensure src module can be found if running from root
sys.path.append('.')

app = FastAPI(
    title="Job Match API",
    description="Live REST API for binary classification of Job/Candidate matches.",
    version="1.0.0"
)

# Global variables for model artifacts
model_pkg = None

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

@app.on_event("startup")
def load_model():
    global model_pkg
    model_path = "models/production_model_package.joblib"
    if not os.path.exists(model_path):
        print(f"WARNING: Model package not found at {model_path}!")
        return
    model_pkg = joblib.load(model_path)
    print(f"Successfully loaded model and threshold: {model_pkg['threshold']}")

@app.get("/health")
def health_check():
    """Basic health check endpoint to confirm API is running and model is loaded."""
    if model_pkg is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Server not ready.")
    return {"status": "ok", "model_loaded": True}

@app.post("/predict")
def predict_match(request: MatchRequest):
    """
    Predicts whether a candidate is a good match for a job based on the provided features.
    Uses the pre-trained and calibrated Stacking Ensemble with cost-based thresholding.
    """
    if model_pkg is None:
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
        
        return {
            "probability": round(float(proba), 4),
            "is_good_match": int(decision),
            "threshold_used": float(threshold)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

