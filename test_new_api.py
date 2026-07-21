import json
import requests

def test_api():
    base_url = "http://127.0.0.1:8000"
    
    payload = {
        "years_experience": 3.0,
        "python_score": 8.0,
        "sql_score": 7.0,
        "ml_score": 6.5,
        "statistics_score": 6.0,
        "javascript_score": 0.0,
        "data_structures_score": 7.0,
        "expected_salary_inr": 800000,
        "education_level": "PG",
        "location_student": "Bangalore",
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
        "location_job": "Bangalore",
        "thresholds": {
            "min_python": 7.5,
            "location_match_required": True
        }
    }
    
    print("Testing /generate_match_vector...")
    res = requests.post(f"{base_url}/generate_match_vector", json=payload)
    print(res.status_code)
    print(json.dumps(res.json(), indent=2))
    
    print("\nTesting /validate_thresholds...")
    res = requests.post(f"{base_url}/validate_thresholds", json=payload)
    print(res.status_code)
    print(json.dumps(res.json(), indent=2))
    
    print("\nTesting /predict...")
    res = requests.post(f"{base_url}/predict", json=payload)
    print(res.status_code)
    print(json.dumps(res.json(), indent=2))

if __name__ == "__main__":
    test_api()
