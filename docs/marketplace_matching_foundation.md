# Marketplace Matching Foundation (Phase 2)

## 1. Match Feature Space

The matching engine consumes features generated from the Candidate (Student) profile and the Job Description. The ML pipeline currently relies on engineered features that measure the gap or similarity between the candidate's attributes and the job's requirements.

### Candidate Features
- **Skills**: Python, SQL, ML, Statistics, JavaScript, Data Structures scores (0-10).
- **Experience**: Years of experience.
- **Education**: Education level (e.g., UG, PG, PhD).
- **Salary Expectation**: Expected salary in INR.
- **Location**: Preferred or current location.
- **NLP Features**: 
  - Resume SBERT Embeddings
  - TF-IDF vectors
  - Extracted skills keyword list.

### Job Features
- **Required Skills**: Minimum scores expected for Python, SQL, ML, Statistics, JavaScript, Data Structures (0-10).
- **Required Experience**: Minimum years of experience.
- **Required Education**: Minimum education level required (e.g., UG, PG, PhD).
- **Salary Offered**: Job's offered salary in INR.
- **Location**: Job's location.
- **NLP Features**:
  - Job Description SBERT Embeddings
  - TF-IDF vectors
  - Extracted skills keyword list.

### Matching Features (Used by ML Pipeline)
These features are engineered using the `JobMatchPreprocessor` and passed into the prediction model:
- **`diff_experience`**: Candidate Experience - Job Required Experience.
- **`diff_python`**, **`diff_ml`**, **`diff_sql`**, **`diff_stats`**, **`diff_js`**, **`diff_ds`**: Candidate Skill Score - Job Required Skill Score.
- **`domain_match`**: Extracted maximum relevant skill based on the job title.
- **`diff_salary`**: Job Salary Offered - Candidate Expected Salary.
- **`diff_education`**: Mapped Candidate Education Level - Job Required Education Level.
- **`location_match`**: Binary match (1 if exact match, else 0).

*Additional NLP overlap features (SBERT Similarity, TF-IDF Similarity, skill overlap count) exist in the feature space and can be used for thresholding.*

## 2. Matching API Contract

The system exposes REST API endpoints via FastAPI to interface between the Backend and the Matching Engine.

### Existing Endpoint: `POST /predict`
Predicts the probability of a match using the pre-trained Logistic Regression classifier.

**Request Schema:**
```json
{
  "years_experience": 2.5,
  "python_score": 8.0,
  "sql_score": 7.0,
  "ml_score": 6.0,
  "statistics_score": 7.5,
  "javascript_score": 0.0,
  "data_structures_score": 6.5,
  "expected_salary_inr": 800000,
  "education_level": "UG",
  "location_student": "Bangalore",
  "title": "Data Scientist",
  "exp_required_years": 2.0,
  "python_required": 7.0,
  "sql_required": 6.0,
  "ml_required": 5.0,
  "statistics_required": 6.0,
  "javascript_required": 0.0,
  "data_structures_required": 5.0,
  "salary_offered_inr": 1000000,
  "edu_minimum": "UG",
  "location_job": "Bangalore"
}
```

**Response Schema:**
```json
{
  "probability": 0.8542,
  "is_good_match": 1,
  "threshold_used": 0.5
}
```

### New Phase 2 Endpoints
To support marketplace matching thresholds, two new functionalities are exposed:
- `POST /generate_match_vector`: Returns the raw feature vector and matching features without making a prediction.
- `POST /validate_thresholds`: Applies configurable thresholds (e.g., minimum resume similarity, minimum experience) before proceeding to prediction, mapping results to competencies.

## 3. Architecture Flow

```
Student & Job Input
         ↓
Feature Engineering (JobMatchPreprocessor)
         ↓
NLP Processing (TF-IDF + SBERT)
         ↓
Match Feature Vector Generated
         ↓
Threshold Validation (Configurable limits)
         ↓
Competency Mapping (Technical, Experience, Education)
         ↓
ML Pipeline (Predictive Model)
         ↓
Threshold Decision (Cost-based Threshold)
         ↓
Prediction & Probability
         ↓
Backend Response (JSON)
```
