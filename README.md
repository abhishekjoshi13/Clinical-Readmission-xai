# Clinical-Readmission-xai
An end-to-end machine learning pipeline predicting 30-day hospital readmissions for diabetic patients using Electronic Health Record(EHR) data.
# Business Problem
High 30-day readmission rates negatively impact patient outcomes and result in financial penalties for healthcare facilities. This project utilizes machine learning to proactively identify high-risk patients at the time of discharge, enabling targeted medical interventions.
## Architecture & Methodology
- **Data Pipeline:** Handled a highly imbalanced dataset of 100K+ clinical records. Implemented categorical encoding for ICD-9 diagnostic codes and managed high-null variables natively via Pandas.
- **Modeling:** Trained a gradient boosted decision tree ('XGBoost') optimized for tabular clinical data, capturing non-linear relationships between medications, lab results, and patient demographics.
- **Explainable AI (XAI):** Integrated 'SHAP' (SHapley Additive exPlanations) to ensure model transparency. The resulting SHAP summary plots provide clinicians with feature-level insights into why a specific prediction was made, prioritizing trust and interpretability in healthcare AI.
