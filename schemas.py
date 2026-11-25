
from pydantic import BaseModel, ConfigDict, Field


class PatientData(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
        str_strip_whitespace=True
    )

    race: str | None = Field(..., max_length=50)
    gender: str | None = Field(..., max_length=20)
    age: str | None = Field(..., max_length=15)

    admission_type_id: str | None = Field(..., max_length=10)
    discharge_disposition_id: str | None = Field(..., max_length=10)
    admission_source_id: str | None = Field(..., max_length=10)

    time_in_hospital: int | None = Field(..., ge=1, le=14)
    num_lab_procedures: int | None = Field(..., ge=0)
    num_procedures: int | None = Field(..., ge=0)
    num_medications: int | None = Field(..., ge=0)
    number_outpatient: int | None = Field(..., ge=0)
    number_emergency: int | None = Field(..., ge=0)
    number_inpatient: int | None = Field(..., ge=0)
    number_diagnoses: int | None = Field(..., ge=1, le=16)

    payer_code: str | None = Field(..., max_length=20)
    medical_specialty: str | None = Field(..., max_length=50)

    diag_1: str | None = Field(..., max_length=10)
    diag_2: str | None = Field(..., max_length=10)
    diag_3: str | None = Field(..., max_length=10)

    max_glu_serum: str | None = Field(..., max_length=20)
    A1Cresult: str | None = Field(..., max_length=20)

    metformin: str | None = Field(..., max_length=20)
    repaglinide: str | None = Field(..., max_length=20)
    nateglinide: str | None = Field(..., max_length=20)
    chlorpropamide: str | None = Field(..., max_length=20)
    glimepiride: str | None = Field(..., max_length=20)
    acetohexamide: str | None = Field(..., max_length=20)
    glipizide: str | None = Field(..., max_length=20)
    glyburide: str | None = Field(..., max_length=20)
    tolbutamide: str | None = Field(..., max_length=20)
    pioglitazone: str | None = Field(..., max_length=20)
    rosiglitazone: str | None = Field(..., max_length=20)
    acarbose: str | None = Field(..., max_length=20)
    miglitol: str | None = Field(..., max_length=20)
    troglitazone: str | None = Field(..., max_length=20)
    tolazamide: str | None = Field(..., max_length=20)
    insulin: str | None = Field(..., max_length=20)

    glyburide_metformin: str | None = Field(
        ..., alias="glyburide-metformin", max_length=20
    )
    glipizide_metformin: str | None = Field(
        ..., alias="glipizide-metformin", max_length=20
    )
    glimepiride_pioglitazone: str | None = Field(
        ..., alias="glimepiride-pioglitazone", max_length=20
    )
    metformin_rosiglitazone: str | None = Field(
        ..., alias="metformin-rosiglitazone", max_length=20
    )
    metformin_pioglitazone: str | None = Field(
        ..., alias="metformin-pioglitazone", max_length=20
    )

    change: str | None = Field(..., max_length=10)
    diabetesMed: str | None = Field(..., max_length=10)
