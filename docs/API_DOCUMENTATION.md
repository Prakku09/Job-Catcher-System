# API Documentation

## Base URL
Local deployment: `http://127.0.0.1:8000`

## Endpoints

### 1. `GET /health`
Validates the API is running and the model is successfully loaded.
**Response (200 OK):**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

### 2. `GET /metadata`
Returns metadata about the deployed model architecture.
**Response (200 OK):**
```json
{
  "model_type": "Logistic Regression (Calibrated)",
  "framework": "Scikit-Learn",
  "threshold_used": 0.2,
  "version": "1.0.0",
  "author": "Job-Catcher-System"
}
```

### 3. `POST /predict`
Predicts if a candidate matches a job.
**Request Body:**
Expects a JSON payload matching the `MatchRequest` Pydantic model (candidate features and job features).

**Response (200 OK):**
```json
{
  "probability": 0.8542,
  "is_good_match": 1,
  "threshold_used": 0.2
}
```
