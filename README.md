# Clinical Readmission Risk Predictor & API
**Live API Endpoint:** [https://clinical-readmission-api.onrender.com/docs](https://clinical-readmission-api.onrender.com/docs)
*(Click the link, select `POST /predict_risk`, click "Try it out", and hit "Execute" to see the real-time inference engine in action.)*
An end-to-end machine learning project comparing a baseline model against a production-grade pipeline to predict 30-day hospital readmissions using Electronic Health Record (EHR) data. 
## Business Problem
High 30-day readmission rates trigger severe financial penalties and indicate poor patient outcomes. This project identifies high-risk diabetic patients at discharge, allowing for targeted clinical interventions.
## Architecture & Methodology
- **Zero Data Leakage:** Implemented `sklearn.pipeline` and `ColumnTransformer` to isolate preprocessing (scaling and encoding) within cross-validation folds.
- **Hyperparameter Tuning:** Applied `RandomizedSearchCV` across XGBoost architectures to maximize minority-class F1 performance.
- **Class Imbalance & Threshold Moving:** Utilized dynamic `scale_pos_weight` and optimized decision thresholds using Precision-Recall curves to prioritize clinical recall.
- **Explainable AI (XAI):** Integrated `SHAP` (SHapley Additive exPlanations) for coalitional game-theory feature attribution.
- **REST API Deployment:** Serialized model artifacts (`.joblib`), resolved dimensionality and pickling mismatches, and wrapped inference logic in an asynchronous `FastAPI` application hosted on Render.
## API Endpoint Reference
- `POST /predict_risk`: Accepts patient clinical JSON payloads and returns calculated probability, risk tier (`Critical Risk`, `Elevated Risk`, `Standard Care`), and flag for mandatory intervention.
