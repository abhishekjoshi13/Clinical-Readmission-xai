
import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_patient():
    return {
        "race": "Caucasian",
        "gender": "Female",
        "age": "[70-80)",
        "admission_type_id": "1",
        "discharge_disposition_id": "1",
        "admission_source_id": "7",
        "time_in_hospital": 5,
        "payer_code": None,
        "medical_specialty": None,
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


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_version"] == "1.0.0"
    assert "explainability_available" in data


def test_prediction_endpoint(client, valid_patient):
    response = client.post("/predict_risk", json=valid_patient)

    assert response.status_code == 200

    data = response.json()

    assert 0 <= data["readmission_probability"] <= 1
    assert 0 <= data["decision_threshold"] <= 1
    assert data["predicted_class"] in [0, 1]
    assert data["predicted_label"] in [
        "readmission_within_30_days",
        "not_readmitted_within_30_days"
    ]
    assert "explanation" in data
    assert "notice" in data


def test_shap_explanation(client, valid_patient):
    response = client.post("/predict_risk", json=valid_patient)

    assert response.status_code == 200

    explanation = response.json()["explanation"]

    assert explanation["available"] is True
    assert explanation["method"] == "SHAP"
    assert explanation["contribution_scale"] == "model_output_margin"
    assert len(explanation["top_factors"]) <= 5


def test_missing_field_rejected(client, valid_patient):
    invalid_patient = valid_patient.copy()
    invalid_patient.pop("race")

    response = client.post("/predict_risk", json=invalid_patient)

    assert response.status_code == 422


def test_extra_field_rejected(client, valid_patient):
    invalid_patient = valid_patient.copy()
    invalid_patient["unexpected_field"] = "invalid"

    response = client.post("/predict_risk", json=invalid_patient)

    assert response.status_code == 422


def test_invalid_numeric_value_rejected(client, valid_patient):
    invalid_patient = valid_patient.copy()
    invalid_patient["time_in_hospital"] = -1

    response = client.post("/predict_risk", json=invalid_patient)

    assert response.status_code == 422


def test_security_headers(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "no-referrer"
    assert "default-src 'none'" in response.headers[
        "Content-Security-Policy"
    ]
    assert response.headers["Cache-Control"] == "no-store"


def test_xss_payload_is_not_reflected(client, valid_patient):
    payload = "<script>alert('xss')</script>"

    malicious_patient = valid_patient.copy()
    malicious_patient["race"] = payload

    response = client.post(
        "/predict_risk",
        json=malicious_patient
    )

    assert response.status_code == 200
    assert payload not in response.text
    assert "<script>" not in response.text


def test_validation_error_does_not_expose_input(
    client,
    valid_patient
):
    malicious_patient = valid_patient.copy()
    malicious_patient["race"] = "<script>secret-value</script>"
    malicious_patient["time_in_hospital"] = -1

    response = client.post(
        "/predict_risk",
        json=malicious_patient
    )

    assert response.status_code == 422
    assert "secret-value" not in response.text
    assert "<script>" not in response.text

    assert response.json()["detail"] == (
        "Request validation failed. "
        "Check field names and value constraints."
    )


 
  
