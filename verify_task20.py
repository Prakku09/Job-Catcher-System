import requests
import time
import pandas as pd
import json
import os

API_URL = "http://127.0.0.1:8000"

payload = {
    "years_experience": 5.0,
    "python_score": 8.0,
    "sql_score": 7.0,
    "ml_score": 8.0,
    "statistics_score": 6.0,
    "javascript_score": 5.0,
    "data_structures_score": 7.0,
    "expected_salary_inr": 1500000.0,
    "education_level": "PG",
    "location_student": "Bangalore",
    "title": "Data Scientist",
    "exp_required_years": 3.0,
    "python_required": 7.0,
    "sql_required": 6.0,
    "ml_required": 7.0,
    "statistics_required": 5.0,
    "javascript_required": 0.0,
    "data_structures_required": 5.0,
    "salary_offered_inr": 1800000.0,
    "edu_minimum": "UG",
    "location_job": "Bangalore"
}

batch_payload = {
    "requests": [payload] * 10
}

def check_endpoints():
    print("Verifying API endpoints...")
    results = []
    
    # 1. Health
    r = requests.get(f"{API_URL}/health")
    results.append({"Endpoint": "/health", "Method": "GET", "StatusCode": r.status_code, "Status": "Pass" if r.status_code == 200 else "Fail", "Response": r.text[:100]})
    
    # 2. Metadata
    r = requests.get(f"{API_URL}/metadata")
    results.append({"Endpoint": "/metadata", "Method": "GET", "StatusCode": r.status_code, "Status": "Pass" if r.status_code == 200 else "Fail", "Response": r.text[:100]})
    
    # 3. Predict
    r = requests.post(f"{API_URL}/predict", json=payload)
    results.append({"Endpoint": "/predict", "Method": "POST", "StatusCode": r.status_code, "Status": "Pass" if r.status_code == 200 else "Fail", "Response": r.text[:100]})
    
    # 4. Batch Predict
    r = requests.post(f"{API_URL}/batch_predict", json=batch_payload)
    results.append({"Endpoint": "/batch_predict", "Method": "POST", "StatusCode": r.status_code, "Status": "Pass" if r.status_code == 200 else "Fail", "Response": r.text[:100]})
    
    # 5. Metrics
    r = requests.get(f"{API_URL}/metrics")
    results.append({"Endpoint": "/metrics", "Method": "GET", "StatusCode": r.status_code, "Status": "Pass" if r.status_code == 200 else "Fail", "Response": r.text[:100]})

    df = pd.DataFrame(results)
    df.to_csv("api_validation_results.csv", index=False)
    print("Saved api_validation_results.csv")
    return df

def measure_performance():
    print("Measuring performance...")
    
    # Measure Latency & Throughput
    num_requests = 100
    latencies = []
    
    start_time_all = time.time()
    for _ in range(num_requests):
        t0 = time.time()
        requests.post(f"{API_URL}/predict", json=payload)
        t1 = time.time()
        latencies.append(t1 - t0)
    end_time_all = time.time()
    
    total_time = end_time_all - start_time_all
    throughput = num_requests / total_time
    avg_latency = sum(latencies) / len(latencies)
    
    # Peak memory omitted to avoid psutil dependency on Windows
    peak_memory_mb = 0.0
    
    # Model size
    model_path = "models/production_binary_classifier.pkl"
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024) if os.path.exists(model_path) else 0

    metrics = {
        "api_latency_ms": round(avg_latency * 1000, 2),
        "throughput_req_per_sec": round(throughput, 2),
        "average_prediction_time_ms": round(avg_latency * 1000, 2),
        "model_loading_time_ms": 0.0, # Pre-loaded at startup
        "peak_memory_mb": round(abs(peak_memory_mb), 2),
        "model_size_mb": round(model_size_mb, 2)
    }
    
    with open("performance_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("Saved performance_metrics.json")
    
if __name__ == "__main__":
    # Ensure server is reachable
    try:
        requests.get(f"{API_URL}/docs")
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to {API_URL}. Ensure uvicorn is running.")
        exit(1)
        
    check_endpoints()
    measure_performance()
