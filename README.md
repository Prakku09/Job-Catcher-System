# Job-Catcher-System

An end-to-end Machine Learning pipeline designed to intelligently match students to job openings. The system extracts domain requirements from job titles, computes skill and experience gaps, and utilizes a hyperparameter-tuned Logistic Regression model (wrapped in a robust Scikit-Learn Pipeline) to predict compatibility while strictly preventing data leakage.

## Live API Demonstration (Task 13)

### 🌐 Public Live API Demonstration
The production-grade calibrated model is live and accessible globally!

You can test the model interactively and view the Swagger UI documentation by navigating to our public Render deployment:
👉 **[https://job-catcher-system.onrender.com/docs](https://job-catcher-system.onrender.com/docs)**

### Local Deployment
1. Clone this repository: `git clone https://github.com/Prakku09/Job-Catcher-System.git`
2. Install the exact required dependencies: `pip install -r requirements.txt`
3. Start the live ASGI server:
   ```bash
   uvicorn src.app:app --host 127.0.0.1 --port 8000
   ```
4. Access the interactive API documentation and test the model live by navigating to:
   **http://127.0.0.1:8000/docs**
   
*Note: The API dynamically loads `models/production_binary_classifier.pkl` and uses the cost-optimized threshold derived from `reports/operating_point.json`.*
