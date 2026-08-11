# Clinical Readmission Risk Predictor
An end-to-end machine learning pipeline predicting 30-day hospital readmissions for diabetic patients using Electronic Health Record(EHR) data.
# Business Problem
High 30-day readmission rates negatively impact patient outcomes and result in financial penalties for healthcare facilities. This project utilizes machine learning to proactively identify high-risk patients at the time of discharge, enabling targeted medical interventions.
## Architecture & Methodology
- **Zero Data Leakage:** Implemented `sklearn.pipeline` and `ColumnTransformer` to isolate preprocessing (scaling and encoding) within cross-validation folds.
- **Hyperparameter Tuning:** Applied `RandomizedSearchCV` across XGBoost architectures to maximize minority-class F1 performance.
- **Class Imbalance & Threshold Moving:** Utilized dynamic `scale_pos_weight` and optimized decision thresholds using Precision-Recall curves to prioritize clinical recall.
- **Explainable AI (XAI):** Integrated `SHAP` (SHapley Additive exPlanations) for coalitional game-theory feature attribution.
- **REST API Deployment:** Serialized model artifacts (`.joblib`) and wrapped inference logic in an asynchronous `FastAPI` application.

## API Endpoint Reference
- `POST /predict_risk`: Accepts patient clinical JSON payloads and returns calculated probability, risk tier (`Critical Risk`, `Elevated Risk`, `Standard Care`), and flag for mandatory intervention.
