# Clinical Readmission Risk Predictor & Explainable API

**Live API Endpoint (Swagger UI):** [https://clinical-readmission-api.onrender.com/docs](https://clinical-readmission-api.onrender.com/docs)

An end-to-end Machine Learning microservice built to predict 30-day hospital readmissions. The model analyzes over 98,000 Electronic Health Records (EHRs) using a full 44-feature set to flag high-risk diabetic patients at discharge, serving as a decision-support system to optimize healthcare resources.

## Project Architecture & Highlights

### 1. Data Engineering & Machine Learning
* **Zero Data Leakage:** Built a robust sklearn pipeline utilizing ColumnTransformer to ensure scaling, imputation, and encoding happen strictly within cross-validation folds.
* **Imbalance Mitigation:** The dataset had a severe minority class (<12% positive cases). Addressed this using XGBoost's dynamic class weighting (`scale_pos_weight`) and Precision-Recall curve threshold optimization, achieving a 0.55 Macro F1-score. The optimal decision threshold was pinned at 0.5349.
* **Local Explainability (XAI):** Integrated SHAP (TreeExplainer) directly into the inference pipeline. The API doesn't just return a score; it returns the top 5 clinical factors driving that specific patient's risk.

### 2. MLOps & Security-First Deployment
* **Asynchronous Backend:** The serialized model (`clinical_readmission_model.joblib`) is wrapped in an async FastAPI application, deployed on Render.
* **Strict Payload Validation:** Enforced rigid Pydantic schema validation. The API natively handles complex clinical mappings (e.g., mapping hyphenated column names like `glyburide-metformin` via Pydantic aliases).
* **Injection Prevention:** By enforcing strict string lengths, bounds, and data types at the schema layer, the API automatically drops malformed JSONs and mitigates basic SQLi and XSS injection vectors before they ever hit the ML model.

## Tech Stack
* **Machine Learning:** Python, XGBoost, Scikit-Learn (v1.6.1), Pandas, SHAP
* **API & Security:** FastAPI, Uvicorn, Pydantic
* **Deployment:** Render, GitHub Actions (CI/CD)

## Local Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/abhishekjoshi13/Clinical-Readmission-xai.git](https://github.com/abhishekjoshi13/Clinical-Readmission-xai.git)
   cd Clinical-Readmission-xai
   ```

2. **Install dependencies:** (It is recommended to use a virtual environment)
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the API Server:**
   ```bash
   uvicorn main:app --reload
   ```

4. **Test the Endpoint:**
   Open `http://localhost:8000/docs` in your browser. The Swagger UI will be pre-populated with default clinical values (covering all 44 required features). Hit **Execute** on the `POST /predict_risk` endpoint to see the real-time inference and SHAP breakdown.

## API Response Example

```json
{
  "probability": 0.4981,
  "risk_tier": "Standard Care",
  "needs_intervention": false,
  "threshold_used": 0.5349,
  "top_shap_factors": {
    "numbers__number_inpatient": 0.1193,
    "categories__discharge_disposition_id_1": -0.1156,
    "categories__diag_2_401": -0.1056,
    "numbers__number_emergency": -0.0405,
    "categories__diabetesMed_No": 0.0332
  }
}
```
