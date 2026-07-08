# Failure Handling Guide

This document outlines standard procedures for gracefully handling common failures when deploying `production_binary_classifier.pkl` into a live environment.

## 1. Missing Dataset or Model File
- **Error Condition:** The application throws a `FileNotFoundError` upon booting.
- **Handling Strategy:** Before loading the model into memory or starting the API, implement a startup check. If `production_binary_classifier.pkl` is missing, the API should refuse to start and log a `CRITICAL` alert to the monitoring system (e.g., Sentry, Datadog) prompting developers to mount the correct model volume or rebuild the container.

## 2. Invalid Input Payload
- **Error Condition:** The client submits a JSON payload with incorrect data types (e.g., string instead of integer).
- **Handling Strategy:** Use a strict validation framework (like Pydantic in FastAPI) at the API boundary. Reject malformed requests immediately with an HTTP 422 Unprocessable Entity, detailing exactly which fields are invalid. Do not pass malformed data to the model.

## 3. Missing Columns / Features
- **Error Condition:** The client payload is missing required predictive features (e.g., `python_score`).
- **Handling Strategy:** The prediction pipeline strictly expects certain columns. Missing columns should be caught by the API schema validation (returning HTTP 422). If a feature is truly optional, the backend must impute it with a standard baseline (like the median or zero) before passing the dataframe to the `Pipeline`.

## 4. Timeout / Latency Spikes
- **Error Condition:** The model inference takes too long due to resource starvation, causing client requests to timeout.
- **Handling Strategy:** Set an explicit timeout on the prediction endpoint (e.g., 500ms). If inference exceeds this, return an HTTP 503 Service Unavailable, and automatically trigger horizontal pod scaling to increase compute resources.

## 5. Unexpected Exceptions in Preprocessing
- **Error Condition:** An unexpected format bypasses validation but causes the `JobMatchPreprocessor` to crash (e.g., a division by zero error during feature engineering).
- **Handling Strategy:** Wrap the `model.predict()` call in a standard `try-except` block. Catch any `Exception`, log the stack trace along with the raw payload for debugging, and return a generic HTTP 500 Internal Server Error to the client, preventing the entire server from crashing.
