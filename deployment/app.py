from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
production_pipeline = joblib.load('clinical_readmission_model.joblib')
with open('optimal_threshold.txt', 'r') as file:
    best_decision_threshold = float(file.read())
app = FastAPI(
    title="Clinical Readmission Risk API",
    description="Real-time 30-day hospital readmission risk scoring and clinical triaging.",
    version="1.0"
)
class PatientData(BaseModel):
    time_in_hospital: int
    num_lab_procedures: int
    num_procedures: int
    num_medications: int
    number_outpatient: int
    number_emergency: int
    number_inpatient: int
    number_diagnoses: int
    race: str
    gender: str
    age: str
    diag_1: str
    diag_2: str
    diag_3: str
@app.get("/")
def health_check():
    return {"status": "online", "model_loaded": True}
@app.post("/predict_risk")
def predict_risk(patient: PatientData):
    patient_data_dict = patient.dict()
    patient_df = pd.DataFrame([patient_data_dict])
    probability = float(production_pipeline.predict_proba(patient_df)[0, 1])
    is_high_risk = bool(probability >= best_decision_threshold)
    if probability >= 0.80:
        risk_tier = "Critical Risk"
    elif probability >= 0.50:
        risk_tier = "Elevated Risk"
    else:
        risk_tier = "Standard Care"
    return {
        "readmission_probability": round(probability, 4),
        "risk_tier": risk_tier,
        "requires_intervention": is_high_risk,
        "classification_threshold": round(best_decision_threshold, 4)
    }
