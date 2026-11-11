# Clinical Readmission Risk Predictor & API

A machine learning pipeline and REST API built to predict 30-day hospital readmissions. The model analyzes over 98,000+ Electronic Health Records (EHR) to flag high-risk diabetic patients at the time of discharge.

**Live API Endpoint:** [https://clinical-readmission-api.onrender.com/docs](https://clinical-readmission-api.onrender.com/docs)

## Project Highlights
* **Imbalance Handling:** The dataset had a heavy minority class (<12% positive cases). Handled this using XGBoost's dynamic `scale_pos_weight` and PR-curve threshold tuning, achieving a 0.55 Macro F1-score.
* **Zero Data Leakage:** Used `sklearn`'s `ColumnTransformer` inside a Pipeline to keep preprocessing steps (scaling, encoding) strictly within cross-validation folds.
* **Explainability:** Integrated `SHAP` (TreeExplainer) directly into the API response to show clinicians exactly which features are driving a patient's risk score.
* **Security:** The FastAPI backend uses strict `Pydantic` schemas. By enforcing exact types and max-lengths on the input payload, it inherently drops malformed data and mitigates basic SQLi/XSS injection vectors before they reach the model.

## Tech Stack
* **Python** (Pandas, NumPy)
* **ML:** XGBoost, Scikit-Learn, SHAP
* **API:** FastAPI, Uvicorn, Pydantic

## Local Setup

1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
