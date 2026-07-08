import requests
import time
import json
import pandas as pd
import subprocess
from datetime import datetime

API_URL = "http://127.0.0.1:8000"

def run_tests():
    metrics = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "endpoints": {}
    }
    
    results = []
    
    # 1. Test /health
    start = time.time()
    res = requests.get(f"{API_URL}/health")
    latency = (time.time() - start) * 1000
    metrics["endpoints"]["health"] = {"status_code": res.status_code, "latency_ms": round(latency, 2)}
    results.append({"endpoint": "/health", "status": res.status_code, "latency_ms": round(latency, 2), "response": json.dumps(res.json())})
    
    # 2. Test /metadata
    start = time.time()
    res = requests.get(f"{API_URL}/metadata")
    latency = (time.time() - start) * 1000
    metrics["endpoints"]["metadata"] = {"status_code": res.status_code, "latency_ms": round(latency, 2)}
    results.append({"endpoint": "/metadata", "status": res.status_code, "latency_ms": round(latency, 2), "response": json.dumps(res.json())})
    
    # 3. Test /predict
    payload = {
        "years_experience": 3,
        "python_score": 8,
        "sql_score": 7,
        "ml_score": 5,
        "statistics_score": 6,
        "javascript_score": 0,
        "data_structures_score": 7,
        "expected_salary_inr": 800000,
        "education_level": "UG",
        "location_student": "Bangalore",
        "title": "Data Scientist",
        "exp_required_years": 2,
        "python_required": 7,
        "sql_required": 6,
        "ml_required": 5,
        "statistics_required": 5,
        "javascript_required": 0,
        "data_structures_required": 5,
        "salary_offered_inr": 1000000,
        "edu_minimum": "UG",
        "location_job": "Bangalore"
    }
    
    # Run multiple predict requests to get average latency
    predict_latencies = []
    for _ in range(10):
        start = time.time()
        res = requests.post(f"{API_URL}/predict", json=payload)
        predict_latencies.append((time.time() - start) * 1000)
        
    avg_predict_latency = sum(predict_latencies) / len(predict_latencies)
    
    metrics["endpoints"]["predict"] = {
        "status_code": res.status_code,
        "avg_latency_ms": round(avg_predict_latency, 2),
        "test_iterations": 10
    }
    results.append({"endpoint": "/predict", "status": res.status_code, "latency_ms": round(avg_predict_latency, 2), "response": json.dumps(res.json())})
    
    # Save Metrics
    with open("reports/api_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Save Results CSV
    df = pd.DataFrame(results)
    df.to_csv("reports/api_test_results.csv", index=False)
    print("API tests completed successfully.")

if __name__ == "__main__":
    run_tests()
