# Clinical Readmission Risk Predictor & Explainable ML API

An end-to-end machine learning research project that estimates the probability of **hospital readmission within 30 days** using patient encounter data. The project combines an XGBoost classification pipeline, a FastAPI prediction service, and SHAP-based explainability.

**Live API:** https://clinical-readmission-xai.onrender.com  
**Interactive API documentation:** https://clinical-readmission-xai.onrender.com/docs  
**Health endpoint:** https://clinical-readmission-xai.onrender.com/health

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Development](#model-development)
- [Model Evaluation](#model-evaluation)
- [Explainability with SHAP](#explainability-with-shap)
- [Technology Stack](#technology-stack)
- [Repository Structure](#repository-structure)
- [API Usage](#api-usage)
- [Running Locally](#running-locally)
- [Deployment](#deployment)
- [Limitations and Responsible Use](#limitations-and-responsible-use)

---

## Project Overview

Hospital readmission is an important healthcare quality and resource-management concern. Identifying patterns associated with readmission can help researchers study patient outcomes and healthcare utilization.

This project develops a machine learning pipeline that estimates the probability of a patient's **readmission within 30 days** based on structured hospital encounter information.

The trained model is exposed through a REST API built with FastAPI. SHAP is integrated to provide feature-level explanations of individual predictions.

The project is intended as a machine learning research and engineering prototype. It is not a clinically validated medical prediction system.

## Key Features

- **Machine Learning:** XGBoost classifier trained on structured hospital encounter data.
- **Patient-Grouped Validation:** Patient-level grouping is used to reduce leakage between training and test sets.
- **Preprocessing Pipeline:** Numeric imputation, categorical imputation, and one-hot encoding.
- **Probability Estimation:** Returns a readmission probability rather than only a class label.
- **Threshold-Based Classification:** Uses a threshold selected from training out-of-fold predictions.
- **Explainable Predictions:** SHAP TreeExplainer provides feature contributions for individual predictions.
- **REST API:** FastAPI endpoints for health checks and risk prediction.
- **Input Validation:** Pydantic validates incoming patient data and rejects unexpected fields.
- **Cloud Deployment:** Hosted on Render.

## System Architecture

```text
                 Hospital Encounter Data
                           |
                           v
                  Data Preprocessing
                           |
             +-------------+-------------+
             |                           |
       Numeric Features           Categorical Features
       Median Imputation          Missing Value Imputation
                                       |
                                 One-Hot Encoding
             |                           |
             +-------------+-------------+
                           |
                           v
                   XGBoost Classifier
                           |
                           v
                 Trained Model Pipeline
                           |
                    Saved Model Artifact
                           |
                           v
                    FastAPI Application
                           |
                 +---------+---------+
                 |                   |
                 v                   v
           Risk Prediction       SHAP Explanation
                 |                   |
                 +---------+---------+
                           |
                           v
                    JSON API Response
```

## Dataset

The project uses the **Diabetes 130-US Hospitals for Years 1999–2008** dataset from the UCI Machine Learning Repository.

Dataset source:

https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

The dataset contains:

- 101,766 hospital encounters
- 50 original columns
- 71,518 unique patients
- Demographic, admission, diagnosis, medication, and hospital-encounter information

### Target Definition

The original `readmitted` field is converted into a binary classification target:

| Original value | Target |
|---|---:|
| `NO` | 0 |
| `>30` | 0 |
| `<30` | 1 |

Therefore, the model specifically estimates **readmission within 30 days**, not readmission at any time.

## Data Preprocessing

The original dataset contains missing values, categorical variables, numeric variables, and identifier columns.

The following preprocessing steps were applied:

1. **Identifier handling:** `encounter_id` and `patient_nbr` were excluded from model features. Patient identifiers were retained only for grouped splitting.
2. **High-missingness handling:** `weight` was removed because approximately 96.86% of its values were missing.
3. **Constant-feature removal:** `examide` and `citoglipton` were removed because they were constant in the dataset.
4. **Missing-value normalization:** Question-mark markers were normalized as missing values. The literal category `"None"` in glucose and A1C result fields was retained as a dataset category.
5. **Numeric imputation:** Missing numeric values were replaced using the median.
6. **Categorical imputation:** Missing categorical values were replaced using a dedicated `__MISSING__` category.
7. **Categorical encoding:** OneHotEncoder was configured with `handle_unknown="ignore"` and `min_frequency=10`.

The final model uses **44 input features**:

- 8 numeric features
- 36 categorical features

The preprocessing transformations are included in the fitted machine learning pipeline.

## Model Development

An XGBoost classifier was used for the binary classification task.

### Validation Strategy

A patient-grouped splitting strategy was used with:

`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`

The final grouped split contained:

- Training encounters: 81,413
- Test encounters: 20,353
- Patient overlap between training and test sets: None

Grouping by patient helps prevent encounters belonging to the same patient from appearing in both training and test partitions.

**Validation caveat:** This is a random patient-grouped split, not a temporal validation. It does not establish how the model would perform on future hospital encounters.

### Hyperparameter Selection

An eight-candidate hyperparameter search was conducted using five grouped folds, with PR-AUC as the selection metric.

The selected configuration was:

| Parameter | Value |
|---|---:|
| `n_estimators` | 200 |
| `max_depth` | 4 |
| `learning_rate` | 0.1 |
| `subsample` | 0.9 |
| `colsample_bytree` | 0.9 |
| `min_child_weight` | 5 |
| `reg_lambda` | 1 |

The selected model was evaluated on a frozen holdout test set.

### Classification Threshold

The final classification threshold is:

`0.13268983364105225`

The threshold was selected using training out-of-fold predictions rather than the final holdout test predictions.

A prediction is classified as positive when its estimated probability meets or exceeds the configured threshold.

## Model Evaluation

The following metrics were obtained on the frozen holdout test set:

| Metric | Result |
|---|---:|
| F1-score | 0.2903 |
| Precision | 0.2118 |
| Recall | 0.4614 |
| PR-AUC | 0.2294 |
| ROC-AUC | 0.6793 |
| Balanced Accuracy | 0.6195 |
| Brier Score | 0.09644 |

### Confusion Matrix

| | Predicted Negative | Predicted Positive |
|---|---:|---:|
| Actual Negative | 14,010 | 4,009 |
| Actual Positive | 1,257 | 1,077 |

### Interpretation

- The model detected approximately 46.14% of the positive readmission cases in the holdout set.
- Approximately 53.86% of positive cases were missed.
- Approximately 21.18% of positive predictions were actually positive.
- The results indicate that the model has limited predictive performance and should not be interpreted as a reliable clinical decision system.

The Brier score was 0.09644, compared with an approximately 0.1015 prevalence-based baseline. This indicates a modest improvement over that baseline, not proof of clinical calibration.

## Explainability with SHAP

SHAP (SHapley Additive exPlanations) is integrated using `TreeExplainer`.

For each prediction, the API attempts to return the five features with the largest absolute SHAP contributions.

The explanation response includes:

- Feature name
- SHAP contribution
- Explanation method
- Contribution scale

The current contribution scale is:

`model_output_margin`

This means contributions are expressed in the model's raw output margin (log-odds) space. They are not probability percentages and do not establish causal relationships.

If explanation generation is unavailable, the API can still return the prediction with an explanation-unavailable response.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python 3.13 |
| Machine Learning | XGBoost |
| Data Processing | Pandas, NumPy |
| Model Pipeline | Scikit-learn |
| Explainable AI | SHAP |
| API Framework | FastAPI |
| Data Validation | Pydantic |
| ASGI Server | Uvicorn |
| Model Serialization | Joblib |
| Deployment | Render |

## Repository Structure

```text
Clinical-Readmission-xai/
├── main.py
├── schemas.py
├── requirements.txt
├── requirements-dev.txt
├── .python-version
├── .gitignore
├── readmission_model_v1.joblib
├── clinical_readmission_model.joblib
├── optimal_threshold.txt
├── readmission_predictor.ipynb
├── tests/
│   └── test_api.py
└── README.md
```

The active API entry point is the root-level `main.py`. The older `deployment/app.py` is retained for reference and is not used by the current Render deployment.

The deployed application loads `readmission_model_v1.joblib` from the repository root. The files `clinical_readmission_model.joblib` and `optimal_threshold.txt` are legacy artifacts and are not used by the current API.

## API Usage

The deployed API provides two endpoints.

### 1. Health Check

**Endpoint**

`GET /health`

**Live URL**

https://clinical-readmission-xai.onrender.com/health

This endpoint checks application health and reports model version and explainability availability.

Example response:

```json
{
  "status": "healthy",
  "model_version": "1.0.0",
  "explainability_available": true
}
```

### 2. Predict Readmission Risk

**Endpoint**

`POST /predict_risk`

**Live URL**

https://clinical-readmission-xai.onrender.com/predict_risk

The endpoint accepts a JSON object containing all 44 model input features.

#### Example Request

The following is a synthetic example for API testing only.

```json
{
  "race": "Caucasian",
  "gender": "Female",
  "age": "[70-80)",
  "admission_type_id": "1",
  "discharge_disposition_id": "1",
  "admission_source_id": "7",
  "time_in_hospital": 5,
  "payer_code": null,
  "medical_specialty": null,
  "num_lab_procedures": 40,
  "num_procedures": 1,
  "num_medications": 15,
  "number_outpatient": 0,
  "number_emergency": 0,
  "number_inpatient": 1,
  "diag_1": "250.83",
  "diag_2": "276",
  "diag_3": "414",
  "number_diagnoses": 9,
  "max_glu_serum": "None",
  "A1Cresult": "None",
  "metformin": "No",
  "repaglinide": "No",
  "nateglinide": "No",
  "chlorpropamide": "No",
  "glimepiride": "No",
  "acetohexamide": "No",
  "glipizide": "No",
  "glyburide": "No",
  "tolbutamide": "No",
  "pioglitazone": "No",
  "rosiglitazone": "No",
  "acarbose": "No",
  "miglitol": "No",
  "troglitazone": "No",
  "tolazamide": "No",
  "insulin": "No",
  "glyburide-metformin": "No",
  "glipizide-metformin": "No",
  "glimepiride-pioglitazone": "No",
  "metformin-rosiglitazone": "No",
  "metformin-pioglitazone": "No",
  "change": "No",
  "diabetesMed": "Yes"
}
```

#### Example Response

```json
{
  "model_version": "1.0.0",
  "readmission_probability": 0.1193,
  "decision_threshold": 0.13269,
  "predicted_class": 0,
  "predicted_label": "not_readmitted_within_30_days",
  "above_threshold": false,
  "explanation": {
    "available": true,
    "method": "SHAP",
    "contribution_scale": "model_output_margin",
    "top_factors": [
      {
        "feature": "cat__discharge_disposition_id_11",
        "contribution": -2.593858
      },
      {
        "feature": "cat__discharge_disposition_id_15",
        "contribution": 0.635844
      },
      {
        "feature": "cat__discharge_disposition_id_13",
        "contribution": -0.577486
      },
      {
        "feature": "cat__discharge_disposition_id_2",
        "contribution": 0.454315
      },
      {
        "feature": "cat__diag_1_V58",
        "contribution": 0.432769
      }
    ]
  },
  "notice": "Research prototype only. This output is not a clinical diagnosis or treatment recommendation."
}
```

The example response corresponds to the synthetic request above. Actual responses depend on the submitted input and loaded model artifact.

### Interactive API Documentation

Swagger UI:

https://clinical-readmission-xai.onrender.com/docs

OpenAPI specification:

https://clinical-readmission-xai.onrender.com/openapi.json

## Running Locally

### Prerequisites

- Python 3.13
- Git
- A trusted copy of the trained model artifact

### 1. Clone the Repository

```bash
git clone https://github.com/abhishekjoshi13/Clinical-Readmission-xai.git
cd Clinical-Readmission-xai
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**macOS / Linux**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the API

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 5. Open the API

- Swagger UI: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

Ensure `readmission_model_v1.joblib` is present in the repository root before starting the application.

## Deployment

The API is deployed using Render.

### Deployment Configuration

| Setting | Value |
|---|---|
| Service Type | Web Service |
| Runtime | Python |
| Branch | `new-branch` |
| Root Directory | Blank |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
| Python Version | `3.13.18` |
| Health Check Path | `/health` |

### Live Deployment

https://clinical-readmission-xai.onrender.com

**Note:** Render free instances can spin down after inactivity. The first request after a period of inactivity may take longer because the service needs to start again.

## Limitations and Responsible Use

This project is an educational and research prototype. It must not be used to diagnose patients, recommend treatment, determine discharge decisions, or make other clinical decisions.

Important limitations include:

1. **No clinical validation:** The model has not been prospectively or clinically validated.
2. **Historical dataset:** The dataset represents hospital encounters from 1999–2008 and may not reflect current clinical practices.
3. **No temporal validation:** The evaluation uses patient-grouped random splitting rather than a future-time holdout.
4. **Limited predictive performance:** The holdout precision, recall, and PR-AUC demonstrate substantial room for improvement.
5. **Potential bias:** Performance can vary across patient subgroups. The available analysis does not establish fairness or absence of bias.
6. **External validation required:** Testing on independent and more recent datasets is necessary.
7. **Privacy and security:** Real patient information must not be submitted to this research API. Any future clinical use would require appropriate privacy, security, governance, and regulatory review.

The model output is an estimate produced by a machine learning system, not a medical fact or clinical recommendation.

## Future Improvements

Potential future research directions include:

- Temporal and external validation on independent datasets.
- More extensive calibration and subgroup evaluation.
- Evaluation of alternative classification models.
- Improved handling of class imbalance and decision thresholds.
- Monitoring of model performance and data drift.
- Stronger API authentication, access control, and deployment security.
- A user-facing interface for visualizing predictions and explanations.

---

**Project Status:** Deployed research prototype  
**Model Version:** 1.0.0  
**API Framework:** FastAPI  
**Deployment Platform:** Render

